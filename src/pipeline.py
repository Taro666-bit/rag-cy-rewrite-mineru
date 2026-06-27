# Qwen-Turbo API的基础限流设置为每分钟不超过500次API调用（QPM）。同时，Token消耗限流为每分钟不超过500,000 Tokens
from dataclasses import dataclass
from pathlib import Path
from pyprojroot import here
import logging
import os
import json
import pandas as pd
import shutil
import time
from PyPDF2 import PdfReader

from src.pdf_parsing import PDFParser
from src import pdf_mineru
from src.parsed_reports_merging import PageTextPreparation
from src.text_splitter import TextSplitter
from src.ingestion import VectorDBIngestor
from src.ingestion import BM25Ingestor
from src.questions_processing import QuestionsProcessor
from src.tables_serialization import TableSerializer

@dataclass
class PipelineConfig:
    def __init__(self, root_path: Path, subset_name: str = "subset.csv", questions_file_name: str = "questions.json", pdf_reports_dir_name: str = "pdf_reports", serialized: bool = False, config_suffix: str = ""):
        # 路径配置，支持不同流程和数据目录
        self.root_path = root_path
        suffix = "_ser_tab" if serialized else ""

        self.subset_path = root_path / subset_name
        self.questions_file_path = root_path / questions_file_name
        self.pdf_reports_dir = root_path / pdf_reports_dir_name
        
        self.answers_file_path = root_path / f"answers{config_suffix}.json"       
        self.debug_data_path = root_path / "debug_data"
        self.databases_path = root_path / f"databases{suffix}"
        
        self.vector_db_dir = self.databases_path / "vector_dbs"
        self.documents_dir = self.databases_path / "chunked_reports"
        self.bm25_db_path = self.databases_path / "bm25_dbs"

        # self.parsed_reports_dirname = "01_parsed_reports"
        # self.parsed_reports_debug_dirname = "01_parsed_reports_debug"
        # self.merged_reports_dirname = f"02_merged_reports{suffix}"
        self.reports_markdown_dirname = f"03_reports_markdown{suffix}"

        #self.parsed_reports_path = self.debug_data_path / self.parsed_reports_dirname
        #self.parsed_reports_debug_path = self.debug_data_path / self.parsed_reports_debug_dirname
        #self.merged_reports_path = self.debug_data_path / self.merged_reports_dirname
        self.reports_markdown_path = self.debug_data_path / self.reports_markdown_dirname

@dataclass
class RunConfig:
    # 运行流程参数配置
    use_serialized_tables: bool = False
    parent_document_retrieval: bool = False
    use_vector_dbs: bool = True
    use_bm25_db: bool = False
    llm_reranking: bool = False
    llm_reranking_sample_size: int = 30
    top_n_retrieval: int = 10
    parallel_requests: int = 1 # 并行的数量，需要限制，否则glm-5会超出阈值
    pipeline_details: str = ""
    submission_file: bool = True
    full_context: bool = False
    use_neighbor_chunks: bool = True   # 默认：将每个检索到的chunk扩展为其±1邻居（共3个chunk）
    use_stream: bool = True            # 默认：DashScope启用流式输出
    api_provider: str = "dashscope" #openai
    answering_model: str = "glm-5" # gpt-4o-mini-2024-07-18 or "gpt-4o-2024-08-06"
    reranking_model: str = "gte-rerank-v2"
    config_suffix: str = ""

class Pipeline:
    def __init__(self, root_path: Path, subset_name: str = "subset.csv", questions_file_name: str = "questions.json", pdf_reports_dir_name: str = "pdf_reports", run_config: RunConfig = RunConfig()):
        # 初始化主流程，加载路径和配置
        self.run_config = run_config
        self.paths = self._initialize_paths(root_path, subset_name, questions_file_name, pdf_reports_dir_name)
        self._convert_json_to_csv_if_needed()

    def _initialize_paths(self, root_path: Path, subset_name: str, questions_file_name: str, pdf_reports_dir_name: str) -> PipelineConfig:
        """根据配置初始化所有路径"""
        return PipelineConfig(
            root_path=root_path,
            subset_name=subset_name,
            questions_file_name=questions_file_name,
            pdf_reports_dir_name=pdf_reports_dir_name,
            serialized=self.run_config.use_serialized_tables,
            config_suffix=self.run_config.config_suffix
        )

    def _convert_json_to_csv_if_needed(self):
        """
        检查是否存在subset.json且无subset.csv，若是则自动转换为CSV。
        """
        json_path = self.paths.root_path / "subset.json"
        csv_path = self.paths.root_path / "subset.csv"
        
        if json_path.exists() and not csv_path.exists():
            try:
                with open(json_path, 'r') as f:
                    data = json.load(f)
                
                df = pd.DataFrame(data)
                
                df.to_csv(csv_path, index=False)
                
            except Exception as e:
                print(f"Error converting JSON to CSV: {str(e)}")

    @staticmethod
    def download_docling_models(): 
        # 下载Docling所需模型，避免首次运行时自动下载
        logging.basicConfig(level=logging.DEBUG)
        parser = PDFParser(output_dir=here())
        parser.parse_and_export(input_doc_paths=[here() / "src/dummy_report.pdf"])

    def parse_pdf_reports_parallel(self, chunk_size: int = 2, max_workers: int = 10):
        """多进程并行解析PDF报告，提升处理效率
        参数：
            chunk_size: 每个worker处理的PDF数
            num_workers: 并发worker数
        """
        logging.basicConfig(level=logging.DEBUG)
        
        pdf_parser = PDFParser(
            output_dir=self.paths.parsed_reports_path,
            csv_metadata_path=self.paths.subset_path
        )
        pdf_parser.debug_data_path = self.paths.parsed_reports_debug_path

        input_doc_paths = list(self.paths.pdf_reports_dir.glob("*.pdf"))
        
        pdf_parser.parse_and_export_parallel(
            input_doc_paths=input_doc_paths,
            optimal_workers=max_workers,
            chunk_size=chunk_size
        )
        print(f"PDF reports parsed and saved to {self.paths.parsed_reports_path}")

    def export_reports_to_markdown(self, file_name):
        """
        使用 pdf_mineru.py，将指定 PDF 文件转换为 markdown，并放到 reports_markdown_dirname 目录下。
        :param file_name: PDF 文件名（如 '【财报】中芯国际：中芯国际2024年年度报告.pdf'）
        """
        # 调用 pdf_mineru 获取 task_id 并下载、解压
        print(f"开始处理: {file_name}")
        task_id = pdf_mineru.get_task_id(file_name)
        print(f"task_id: {task_id}")
        pdf_mineru.get_result(task_id)

        # 解压后目录名与 task_id 相同
        extract_dir = f"{task_id}"
        md_path = os.path.join(extract_dir, "full.md")
        if not os.path.exists(md_path):
            print(f"未找到 markdown 文件: {md_path}")
            return
        # 目标目录
        os.makedirs(self.paths.reports_markdown_path, exist_ok=True)
        # 目标文件名为原始 file_name，扩展名改为 .md
        base_name = os.path.splitext(file_name)[0]
        target_path = os.path.join(self.paths.reports_markdown_path, f"{base_name}.md")
        shutil.move(md_path, target_path)
        print(f"已将 {md_path} 移动到 {target_path}")

    def export_report_with_pages(self, file_name: str):
        """
        使用 MinerU API 分片解析大 PDF（200+ 页），从 content_list.json 提取逐页内容，
        构建 Docling 兼容的 JSON 并用 _split_report 分块（每个 chunk 带 page 字段），
        输出到 databases/chunked_reports/。

        流程：
        1. PyPDF2 获取总页数 → 计算分片（每片 ≤200 页）
        2. 逐片调用 MinerU API → 下载 zip → 解析 content_list.json → 提取每页文本
        3. 合并所有页 → 构建 Docling 格式 → _split_report 分块
        4. 保存到 chunked_reports/
        :param file_name: PDF 文件名
        """
        pdf_path = self.paths.pdf_reports_dir / file_name
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF 文件不存在: {pdf_path}")

        # 1. 获取总页数
        reader = PdfReader(str(pdf_path))
        total_pages = len(reader.pages)
        print(f"PDF 总页数: {total_pages}")

        PAGE_LIMIT = 200
        base_name = os.path.splitext(file_name)[0]
        all_pages = []  # 汇总所有分片的页面数据

        # 2. 分片处理
        for range_start in range(1, total_pages + 1, PAGE_LIMIT):
            range_end = min(range_start + PAGE_LIMIT - 1, total_pages)
            page_ranges = f"{range_start}-{range_end}"
            page_offset = range_start - 1  # 页码偏移（第二片起）

            print(f"\n{'='*60}")
            print(f"处理第 {range_start}-{range_end} 页（共 {total_pages} 页）")
            print(f"{'='*60}")

            # 提交任务
            task_id = pdf_mineru.submit_task(file_name, page_ranges=page_ranges)
            print(f"task_id: {task_id}")

            # 下载并解压
            extract_dir = pdf_mineru.wait_and_download(task_id)

            # 解析 content_list.json → 每页文本
            try:
                cl_path = pdf_mineru.find_content_list_json(extract_dir)
                pages = pdf_mineru.parse_content_list_to_pages(cl_path, page_offset=page_offset)
                print(f"本片提取到 {len(pages)} 页")
                all_pages.extend(pages)
            except FileNotFoundError as e:
                print(f"警告: {e}，跳过本片")

        if not all_pages:
            raise RuntimeError("未能从任何分片中提取到页面内容")

        print(f"\n合并完成，共 {len(all_pages)} 页")

        # 3. 构建 Docling 兼容的 JSON
        sha1 = "stock_10001"  # 使用固定 sha1，与现有 subset.csv 匹配
        company_name = "中芯国际"

        docling_json = {
            "metainfo": {
                "sha1": sha1,
                "company_name": company_name,
                "file_name": file_name,
            },
            "content": {
                "pages": all_pages,
                "chunks": None,
            },
        }

        # 4. 用 _split_report 分块（每 chunk 自动带 page 字段）
        text_splitter = TextSplitter()
        chunked = text_splitter._split_report(docling_json)

        # 5. 保存到 chunked_reports/
        self.paths.documents_dir.mkdir(parents=True, exist_ok=True)
        output_path = self.paths.documents_dir / f"{base_name}.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(chunked, f, indent=2, ensure_ascii=False)
        print(f"分块报告已保存到: {output_path}")
        print(f"共 {len(chunked['content']['chunks'])} 个 chunk")

    def chunk_reports(self, include_serialized_tables: bool = False):
        """
        将规整后 markdown 报告分块，便于后续向量化和检索
        """
        text_splitter = TextSplitter()
        # 只处理 markdown 文件，输入目录为 reports_markdown_path，输出目录为 documents_dir
        print(f"开始分割 {self.paths.reports_markdown_path} 目录下的 markdown 文件...")
        # 自动传入 subset.csv 路径，便于补充 company_name 字段
        text_splitter.split_markdown_reports(
            all_md_dir=self.paths.reports_markdown_path,
            output_dir=self.paths.documents_dir,
            subset_csv=self.paths.subset_path
        )
        print(f"分割完成，结果已保存到 {self.paths.documents_dir}")

    def create_vector_dbs(self):
        """从分块报告创建向量数据库"""
        input_dir = self.paths.documents_dir
        output_dir = self.paths.vector_db_dir
        
        vdb_ingestor = VectorDBIngestor()
        vdb_ingestor.process_reports(input_dir, output_dir)
        print(f"Vector databases created in {output_dir}")
    
    def create_bm25_db(self):
        """从分块报告创建BM25数据库"""
        input_dir = self.paths.documents_dir
        output_file = self.paths.bm25_db_path
        
        bm25_ingestor = BM25Ingestor()
        bm25_ingestor.process_reports(input_dir, output_file)
        print(f"BM25 database created at {output_file}")
    
    def parse_pdf_reports(self, parallel: bool = True, chunk_size: int = 2, max_workers: int = 10):
        # 解析PDF报告，支持并行处理
        if parallel:
            self.parse_pdf_reports_parallel(chunk_size=chunk_size, max_workers=max_workers)

    def process_parsed_reports(self):
        """
        处理已解析的PDF报告，主要流程：
        1. 对报告进行分块
        2. 创建向量数据库
        """
        print("开始处理报告流程...")
        
        print("步骤1：报告分块...")
        self.chunk_reports()
        
        print("步骤2：创建向量数据库...")
        self.create_vector_dbs()

        print("步骤3：创建BM25索引...")
        self.create_bm25_db()
        
        print("报告处理流程已成功完成！")
        
    def _get_next_available_filename(self, base_path: Path) -> Path:
        """
        获取下一个可用的文件名，如果文件已存在则自动添加编号后缀。
        例如：若answers.json已存在，则返回answers_01.json等。
        """
        if not base_path.exists():
            return base_path
            
        stem = base_path.stem
        suffix = base_path.suffix
        parent = base_path.parent
        
        counter = 1
        while True:
            new_filename = f"{stem}_{counter:02d}{suffix}"
            new_path = parent / new_filename
            
            if not new_path.exists():
                return new_path
            counter += 1

    def process_questions(self):
        # 处理所有问题，生成答案文件
        processor = QuestionsProcessor(
            vector_db_dir=self.paths.vector_db_dir,
            documents_dir=self.paths.documents_dir,
            bm25_db_dir=self.paths.bm25_db_path,
            questions_file_path=self.paths.questions_file_path,
            new_challenge_pipeline=True,
            subset_path=self.paths.subset_path,
            parent_document_retrieval=self.run_config.parent_document_retrieval,
            llm_reranking=self.run_config.llm_reranking,
            llm_reranking_sample_size=self.run_config.llm_reranking_sample_size,
            top_n_retrieval=self.run_config.top_n_retrieval,
            parallel_requests=self.run_config.parallel_requests,
            api_provider=self.run_config.api_provider,
            answering_model=self.run_config.answering_model,
            reranking_model=self.run_config.reranking_model,
            full_context=self.run_config.full_context,
            use_neighbor_chunks=self.run_config.use_neighbor_chunks,
            use_stream=self.run_config.use_stream
        )
        
        output_path = self._get_next_available_filename(self.paths.answers_file_path)
        
        _ = processor.process_all_questions(
            output_path=output_path,
            submission_file=self.run_config.submission_file,
            pipeline_details=self.run_config.pipeline_details
        )
        print(f"Answers saved to {output_path}")

    def answer_single_question(self, question: str, kind: str = "string", use_stream: bool = None, use_neighbor_chunks: bool = None):
        """
        单条问题即时推理，返回结构化答案（dict）。
        kind: 支持 'string'、'number'、'boolean'、'names' 等
        use_stream: 覆盖 RunConfig 的流式设置
        use_neighbor_chunks: 覆盖 RunConfig 的邻域chunk设置
        """
        t0 = time.time()
        _use_stream = use_stream if use_stream is not None else self.run_config.use_stream
        _use_neighbor = use_neighbor_chunks if use_neighbor_chunks is not None else self.run_config.use_neighbor_chunks
        processor = QuestionsProcessor(
            vector_db_dir=self.paths.vector_db_dir,
            documents_dir=self.paths.documents_dir,
            bm25_db_dir=self.paths.bm25_db_path,
            questions_file_path=None,
            new_challenge_pipeline=True,
            subset_path=self.paths.subset_path,
            parent_document_retrieval=self.run_config.parent_document_retrieval,
            llm_reranking=self.run_config.llm_reranking,
            llm_reranking_sample_size=self.run_config.llm_reranking_sample_size,
            top_n_retrieval=self.run_config.top_n_retrieval,
            parallel_requests=1,
            api_provider=self.run_config.api_provider,
            answering_model=self.run_config.answering_model,
            reranking_model=self.run_config.reranking_model,
            full_context=self.run_config.full_context,
            use_neighbor_chunks=_use_neighbor,
            use_stream=_use_stream
        )
        answer = processor.process_single_question(question, kind=kind)
        t2 = time.time()
        return {
            "answer": answer,
            "retrieval_info": processor.last_retrieval_info if hasattr(processor, 'last_retrieval_info') else None,
            "response_data": processor.response_data if hasattr(processor, 'response_data') else None
        }

    def answer_single_question_stream(self, question: str, kind: str = "string", use_stream: bool = None, use_neighbor_chunks: bool = None, top_n_retrieval: int = None):
        """
        流式版本：实时 yield LLM token 和最终结构化答案。
        Yields: {"type": "token", "content": delta}
                {"type": "done", "answer": answer_dict, "retrieval_info": info}
        """
        _use_neighbor = use_neighbor_chunks if use_neighbor_chunks is not None else self.run_config.use_neighbor_chunks
        _top_n = top_n_retrieval if top_n_retrieval is not None else self.run_config.top_n_retrieval
        processor = QuestionsProcessor(
            vector_db_dir=self.paths.vector_db_dir,
            documents_dir=self.paths.documents_dir,
            bm25_db_dir=self.paths.bm25_db_path,
            questions_file_path=None,
            new_challenge_pipeline=True,
            subset_path=self.paths.subset_path,
            parent_document_retrieval=self.run_config.parent_document_retrieval,
            llm_reranking=self.run_config.llm_reranking,
            llm_reranking_sample_size=self.run_config.llm_reranking_sample_size,
            top_n_retrieval=_top_n,
            parallel_requests=1,
            api_provider=self.run_config.api_provider,
            answering_model=self.run_config.answering_model,
            reranking_model=self.run_config.reranking_model,
            full_context=self.run_config.full_context,
            use_neighbor_chunks=_use_neighbor,
            use_stream=True
        )
        stream_gen = processor.process_single_question_stream(question, kind=kind)
        
        final_answer = None
        final_info = None
        for item in stream_gen:
            if item["type"] == "token":
                yield item
            elif item["type"] == "retrieval":
                yield item
            elif item["type"] == "done":
                final_answer = item["answer"]
                final_info = item.get("retrieval_info")
        
        yield {
            "type": "done",
            "answer": final_answer,
            "retrieval_info": final_info,
            "response_data": processor.response_data if hasattr(processor, 'response_data') else None
        }


preprocess_configs = {"ser_tab": RunConfig(use_serialized_tables=True),
                      "no_ser_tab": RunConfig(use_serialized_tables=False)}

base_config = RunConfig(
    parallel_requests=10,
    submission_file=True,
    pipeline_details="Custom pdf parsing + vDB + Router + SO CoT; llm = GPT-4o-mini",
    config_suffix="_base"
)

parent_document_retrieval_config = RunConfig(
    parent_document_retrieval=True,
    parallel_requests=20,
    submission_file=True,
    pipeline_details="Custom pdf parsing + vDB + Router + Parent Document Retrieval + SO CoT; llm = GPT-4o",
    answering_model="gpt-4o-2024-08-06",
    config_suffix="_pdr"
)

## 这里
max_config = RunConfig(
    use_serialized_tables=False,
    parent_document_retrieval=False,
    llm_reranking=True,
    parallel_requests=4,
    submission_file=True,
    pipeline_details="Custom pdf parsing + vDB + Router + Parent Document Retrieval + reranking + SO CoT; llm = glm-5",
    answering_model="glm-5",
    reranking_model="gte-rerank-v2",
    config_suffix="_glm_5"
)


configs = {"base": base_config,
           "pdr": parent_document_retrieval_config,
           "max": max_config}


# 你可以直接在本文件中运行任意方法：
# python .\src\pipeline.py
# 只需取消你想运行的方法的注释即可
# 你也可以修改 run_config 以尝试不同的配置
if __name__ == "__main__":
    # 设置数据集根目录（此处以 test_set 为例）
    root_path = here() / "data" / "stock_data"
    print('root_path:', root_path)
    # 初始化主流程，使用推荐的最佳配置
    pipeline = Pipeline(root_path, run_config=max_config)

    FILE_NAME = '【财报】中芯国际：中芯国际2024年年度报告.pdf'

    # 4. MinerU 分片解析 + 构建带 page 的分块 JSON → databases/chunked_reports/
    print('4. MinerU 分片解析，构建带 page 的分块报告')
    pipeline.export_report_with_pages(FILE_NAME)

    # 5. 从分块报告创建向量数据库，输出到 databases/vector_dbs
    print('5. 从分块报告创建向量数据库')
    pipeline.create_vector_dbs()

    # 6. 从分块报告创建BM25索引，输出到 databases/bm25_dbs
    print('6. 从分块报告创建BM25索引')
    pipeline.create_bm25_db()

    print('完成')
