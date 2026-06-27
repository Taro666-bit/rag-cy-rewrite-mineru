# RAG 年报智能问答系统

基于检索增强生成（RAG）实现的公司年报自动问答系统。支持对 PDF 年报进行深度解析，通过向量检索与 LLM 重排序精准定位相关内容，结合链式推理（Chain-of-Thought）生成结构化答案。

核心技术栈：

- 基于 Docling 的 PDF 结构化解析
- 向量检索 + 父文档召回
- LLM 重排序提升上下文相关性
- 结构化输出 + 链式推理（Chain-of-Thought）
- 多公司比较问题的自动拆分与并行处理

## 快速开始

环境配置：
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或 venv\Scripts\Activate.ps1  # Windows (PowerShell)
pip install -e . -r requirements.txt
```

将 `env` 重命名为 `.env` 并填入你的 API Key。

## 数据准备

将 PDF 年报放入 `data/stock_data/pdf_reports/` 目录，问题放入 `data/stock_data/questions.json`。

## 使用方式

运行完整流水线：
```bash
python ./src/pipeline.py
```

也可通过 `main.py` 单独执行某个阶段：
```bash
cd ./data/stock_data/
python ../../main.py process-questions --config max_nst_o3m
```

### CLI 命令

查看可用命令：
```bash
python main.py --help
```

可用命令：
- `download-models` - 下载 Docling 模型
- `parse-pdfs` - 解析 PDF 年报（支持并行处理）
- `serialize-tables` - 处理解析后的表格
- `process-reports` - 运行完整报告处理流水线
- `process-questions` - 使用指定配置处理问题

每个命令都有独立帮助，例如：
```bash
python main.py parse-pdfs --help
```

## 可选配置

- `max_nst_o3m` - 推荐配置，使用 OpenAI o3-mini 模型
- `gemini_thinking` - 利用 Gemini 超大上下文窗口直接回答（非 RAG 模式）

更多配置详见 `src/pipeline.py`。

## License

MIT
