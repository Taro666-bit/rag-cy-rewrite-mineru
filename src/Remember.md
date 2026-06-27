# 提示词模版

> 共 9 个 Prompt 类，定义在 `src/prompts.py`，由 `src/api_requests.py` 的 `_build_rag_context_prompts(schema)` 按 schema 分发调用。

---

## 整体调用流程

```
用户提问
  │
  ├─ 单公司问题 ──→ 按问题类型选 schema ──→ 6种 RAG答案Prompt 之一
  │
  └─ 多公司比较 ──→ ① RephrasedQuestionsPrompt（拆问题）
                         │
                    ② 并行调用 6种 RAG答案Prompt（每公司独立回答）
                         │
                    ③ ComparativeAnswerPrompt（汇总比较）
                         │
                    ④ AnswerSchemaFixPrompt（格式兜底修复，必要时）
```

---

## 九大 Prompt 详细提示词

### 1. `RephrasedQuestionsPrompt` — 问题重写

**Instruction（系统指令）：**
```
你是一个问题重写系统。
你的任务是将比较类问题拆解为针对每个公司独立的具体问题。
每个输出问题都必须自洽、保持原意和指标、针对对应公司，并用一致的表达方式。
```

**User Prompt：**
```
原始比较问题：'{question}'

涉及公司：{companies}
```

**Example：**
```
示例：
输入：
原始比较问题：'2022年哪家公司营收更高，"苹果"还是"微软"？'
涉及公司："苹果", "微软"

输出：
{
    "questions": [
        {
            "company_name": "苹果",
            "question": "苹果公司2022年营收是多少？"
        },
        {
            "company_name": "微软",
            "question": "微软公司2022年营收是多少？"
        }
    ]
}
```

**输出 Schema：** `RephrasedQuestion` 列表，每条含 `company_name`（公司名）和 `question`（重写问题）

---

### 2. `AnswerWithRAGContextNamePrompt` — 人名/实体名问答

**备注：** Instruction 和 User Prompt 继承自共享基类 `AnswerWithRAGContextSharedPrompt`

**Instruction（共享基类）：**
```
你是一个RAG（检索增强生成）问答系统。
你的任务是仅基于公司年报中RAG检索到的相关页面内容，回答给定问题。

在给出最终答案前，请详细分步思考，尤其关注问题措辞。
- 注意：答案可能与问题表述不同。
- 问题可能是模板生成的，有时对该公司不适用。
```

**User Prompt（共享基类）：**
```
以下是上下文:
"""
{context}
"""

---

以下是问题：
"{question}"
```

**Example：**
```
示例：
问题：
"'南方航空股份有限公司'的CEO是谁？"

答案：
{
  "step_by_step_analysis": "1. 问题询问'南方航空股份有限公司'的CEO。CEO通常是公司最高管理者，有时也称总裁或董事总经理。\n2. 信息来源为该公司的年报，将用来确认CEO身份。\n3. 年报中明确指出张三为公司总裁兼首席执行官。\n4. 因此，CEO为张三。",
  "reasoning_summary": "年报明确写明张三为总裁兼CEO，直接回答了问题。",
  "relevant_pages": [58],
  "final_answer": "张三"
}
```

**输出 Schema：**
| 字段 | 类型 | 说明 |
|---|---|---|
| `step_by_step_analysis` | `str` | 详细分步推理，至少5步、150字以上 |
| `reasoning_summary` | `str` | 简要总结，约50字 |
| `relevant_pages` | `List[int]` | 直接用于回答的页面编号 |
| `final_answer` | `str` 或 `"N/A"` | 公司名需与问题完全一致，人名需全名，产品名需与上下文一致。无信息返回 `N/A` |

---

### 3. `AnswerWithRAGContextNumberPrompt` — 数值型问答（最复杂）

**Instruction / User Prompt：** 同上（共享基类）

**额外内嵌规则（Schema 中的严格指标匹配要求）：**
```
1. 明确问题中指标的精确定义，它实际衡量什么？
2. 检查上下文中的所有可能指标。不要只看名称，要关注其实际衡量内容。
3. 仅当上下文指标的含义与目标指标*完全一致*时才接受。可接受同义词，但概念不同则不可。
4. 拒绝（并返回'N/A'）的情况：
    - 上下文指标范围大于或小于问题指标。
    - 上下文指标为相关但非*完全等价*的概念（如代理指标或更宽泛类别）。
    - 需要计算、推导或推断才能作答。
    - 聚合不匹配：问题要求单一值，但上下文仅有总计。
5. 不允许猜测：如对指标等价性有任何疑问，默认返回 N/A。
```

**数值格式处理规则（Schema description 内嵌）：**
- 百分比：`58,3%` → `58.3`
- 负数括号：`(2,124,837) CHF` → `-2124837`
- 千为单位：`4970,5（千美元）` → `4970500`
- 币种不匹配 → 返回 `N/A`
- 不可由其他指标计算/推导 → 返回 `N/A`

**Example 1（正常匹配）：**
```
问题：
"'万科企业股份有限公司'2022年总资产是多少？"

答案：
{
  "step_by_step_analysis": "1. 问题询问'万科企业股份有限公司'2022年总资产。'总资产'指公司拥有的全部资源。\n2. 年报第78页有'合并资产负债表'，列明2022年12月31日总资产。\n3. 该行数据为'总资产'，与问题完全匹配。\n4. 报告显示总资产为18500342000元。\n5. 无需计算，直接取值。",
  "reasoning_summary": "年报78页直接给出2022年总资产，无需推算。",
  "relevant_pages": [78],
  "final_answer": 18500342000
}
```

**Example 2（严格拒绝匹配）：**
```
问题：
"'某医药公司'年报期末研发设备原值是多少？"

答案：
{
  "step_by_step_analysis": "1. 问题询问研发设备原值。\n2. 年报35页有'固定资产净值'12500元，但为净值，非原值。\n3. 37页有'累计折旧'11万元，但未区分研发设备。\n4. 无法直接获得研发设备原值。\n5. 因此答案为'N/A'。",
  "reasoning_summary": "年报无研发设备原值，严格匹配应返回N/A。",
  "relevant_pages": [35, 37],
  "final_answer": "N/A"
}
```

**输出 Schema：** 同上（4字段），`final_answer` 为 `float | int | "N/A"`

---

### 4. `AnswerWithRAGContextBooleanPrompt` — 布尔型问答

**Instruction / User Prompt：** 同上（共享基类）

**Example：**
```
问题：
"'万科企业股份有限公司'年报是否宣布了分红政策变更？"

答案：
{
  "step_by_step_analysis": "1. 问题询问是否有分红政策变更。\n2. 年报12、18页提到年度分红金额增加，但政策未变。\n3. 45页有分红细节。\n4. 持续小幅增长，符合既定政策。\n5. 问题问的是政策变更，非金额变化。",
  "reasoning_summary": "年报显示分红金额变化但政策未变，答案为False。",
  "relevant_pages": [12, 18, 45],
  "final_answer": false
}
```

**输出 Schema：** 同上（4字段），`final_answer` 为 `bool`（`true` / `false`）

---

### 5. `AnswerWithRAGContextNamesPrompt` — 名单/多实体问答

**Instruction / User Prompt：** 同上（共享基类）

**Example：**
```
示例：
问题：
"公司有哪些新任高管？"

答案：
{
    "step_by_step_analysis": "1. 问题询问公司新任高管名单。\n2. 年报89页列出新高管签约信息。\n3. 10.9节说明张三为新任总法律顾问，10.10节李四为新任COO。\n4. 综上，张三和李四为新任高管。",
    "reasoning_summary": "年报10.9、10.10节明确列出张三、李四为新任高管。",
    "relevant_pages": [89],
    "final_answer": ["张三", "李四"]
}
```

**输出规则（按问题类型区分）：**
- 问职位 → 只返回职位名称（如 `['首席技术官', '董事', '首席执行官']`）
- 问姓名 → 返回上下文中的全名（如 `['张三', '李四']`）
- 问新产品 → 返回上下文中的产品名（候选/测试产品不算）
- 无信息 → `"N/A"`

**输出 Schema：** 同上（4字段），`final_answer` 为 `List[str] | "N/A"`

---

### 6. `AnswerWithRAGContextStringPrompt` — 文本总结型问答

**Instruction / User Prompt：** 同上（共享基类）

**Example：**
```
示例：
问题：
"请简要总结'万科企业股份有限公司'2022年主营业务的主要内容。"

答案：
{
  "step_by_step_analysis": "1. 问题要求总结2022年万科企业股份有限公司的主营业务。\n2. 年报第10-12页详细描述了公司主营业务，包括房地产开发、物业服务等。\n3. 结合上下文，归纳出主要业务板块。\n4. 重点突出房地产开发和相关服务。\n5. 形成简明扼要的总结。",
  "reasoning_summary": "年报10-12页明确列出主营业务，答案基于原文归纳。",
  "relevant_pages": [10, 11, 12],
  "final_answer": "万科企业股份有限公司2022年主营业务包括房地产开发、物业服务、租赁住房、物流仓储等，核心业务为住宅及商业地产开发与运营。"
}
```

**输出 Schema：** 同上（4字段），`final_answer` 为完整连贯的文本段落（`str`）

---

### 7. `ComparativeAnswerPrompt` — 比较类最终答案聚合

**Instruction（独立，不继承共享基类）：**
```
你是一个问答系统。
你的任务是基于各公司独立答案，给出原始比较问题的最终结论。
只能基于已给出的答案，不可引入外部知识。
请分步详细推理。

比较规则：
- 问题要求选出公司时，答案必须与原问题公司名完全一致
- 若某公司数据币种不符，需排除
- 若全部公司被排除，返回'N/A'
- 若仅剩一家，直接返回该公司名
```

**User Prompt：**
```
以下是单个公司的回答：
"""
{context}
"""

---

以下是原始比较问题：
"{question}"
```

**Example：**
```
示例：
问题：
"下列公司中，哪家2022年总资产最低："A公司", "B公司", "C公司"？若无数据则排除。"

答案：
{
  "step_by_step_analysis": "1. 问题要求比较多家公司2022年总资产。\n2. 各公司独立答案：A公司6,601,086,000元，B公司1,249,642,000元，C公司217,435,000元。\n3. 直接比较得C公司最低。\n4. 若有公司币种不符则排除。\n5. 因此答案为C公司。",
  "reasoning_summary": "独立答案显示C公司总资产最低，直接得出结论。",
  "relevant_pages": [],
  "final_answer": "C公司"
}
```

**输出 Schema：** 同上（4字段），`relevant_pages` 保持空，`final_answer` 为单个公司名或 `"N/A"`

---

### 8. `AnswerSchemaFixPrompt` — JSON 格式兜底修复

**System Prompt：**
```
你是一个JSON格式化助手。
你的任务是将大模型输出的原始内容格式化为合法的JSON对象。
你的回答必须以"{"开头，以"}"结尾。
你的回答只能包含JSON字符串，不要有任何前言、注释或三引号。
```

**User Prompt：**
```
下面是定义JSON对象Schema和示例的系统提示词:
"""
{system_prompt}
"""

---

下面是需要你格式化为合法JSON的LLM原始输出：
"""
{response}
"""
```

**说明：** 此类无 example——任务是纯格式化修复，不需要示范。

---

### 9. `RerankingPrompt` — RAG 检索重排序

**单块模式 `system_prompt_rerank_single_block`：**
```
你是一个RAG检索重排专家。
你将收到一个查询和一个检索到的文本块，请根据其与查询的相关性进行评分。

评分说明：
1. 推理：分析文本块与查询的关系，简要说明理由。
2. 相关性分数（0-1，步长0.1）：
   0 = 完全无关
   0.1 = 极弱相关
   0.2 = 很弱相关
   0.3 = 略有相关
   0.4 = 部分相关
   0.5 = 一般相关
   0.6 = 较为相关
   0.7 = 相关
   0.8 = 很相关
   0.9 = 高度相关
   1 = 完全匹配
3. 只基于内容客观评价，不做假设。
```

**多块模式 `system_prompt_rerank_multiple_blocks`：** 同上，仅将"一个检索到的文本块"改为"若干检索到的文本块，请分别对每个块进行相关性评分"。

**说明：** 此类无 example 和 user_prompt——由调用方自行构造 query + 文本块。

---

## 九大 Prompt 总览

| # | 类名 | 用途 | 有 Example? | final_answer 类型 |
|---|------|------|:---:|---|
| 1 | `RephrasedQuestionsPrompt` | 比较问题拆解 | ✅ | `[{公司名, 问题}]` |
| 2 | `AnswerWithRAGContextNamePrompt` | 人名/实体提取 | ✅ | `str \| N/A` |
| 3 | `AnswerWithRAGContextNumberPrompt` | 数值提取（最复杂） | ✅（2个） | `float \| int \| N/A` |
| 4 | `AnswerWithRAGContextBooleanPrompt` | 是非判断 | ✅ | `bool` |
| 5 | `AnswerWithRAGContextNamesPrompt` | 名单提取 | ✅ | `List[str] \| N/A` |
| 6 | `AnswerWithRAGContextStringPrompt` | 文本总结 | ✅ | `str` |
| 7 | `ComparativeAnswerPrompt` | 比较结论合并 | ✅ | `str \| N/A` |
| 8 | `AnswerSchemaFixPrompt` | JSON 格式修复 | ❌ | JSON |
| 9 | `RerankingPrompt` | 检索片段重排序 | ❌ | 相关性分数 0~1 |

---

## 输入输出规范的四层约束机制

1. **Pydantic Schema 强约束**：每个 Prompt 类内嵌 `AnswerSchema(BaseModel)`，用 `Field(description=...)` 精确定义字段含义和格式，并通过 `inspect.getsource()` 嵌入 system prompt
2. **类型 + Literal 联合做取值约束**：`final_answer` 用 `Union[float, Literal["N/A"]]` 等形式，限制 LLM 只能返回特定类型
3. **Few-shot Example 规范输出格式**：每个 Prompt 提供 JSON 示例，让 LLM 模仿标准输出
4. **Instruction 中嵌入推理规则**：如 number 类写了 7 条严格指标匹配规则，禁止推算、禁止猜测

---

## 一句话总结

> **9 个 Prompt = 1 拆问题 + 6 按类型回答 + 1 比较汇总 + 1 重排 + 1 格式修复**

核心设计思想是 **"按答案类型分治"**——不要让一个 prompt 同时处理人名、数字、布尔、列表、文本，每类问题各有各的陷阱和边界条件，独立约束反而更精准、幻觉更少。

---

# 项目架构深度分析（基于代码审查）

## 一、当前实际走的分块路径：MinerU（非 Docling）

### 结论
当前项目**实际运行的是 MinerU 路径**，不是 Docling 路径。两者产物完全不同，影响后续检索能力。

### 两条路径对比

| | MinerU 路径（当前在用） | Docling 路径（未启用） |
|---|---|---|
| 入口 | `pipeline.export_reports_to_markdown()` → `pdf_mineru.py` → 生成 `full.md` | `pipeline.parse_pdf_reports()` → `pdf_parsing.py` → 生成结构化 JSON |
| 分块函数 | `text_splitter.py` 的 `split_markdown_reports()` | `text_splitter.py` 的 `split_all_reports()` → `_split_page()` |
| 分块方式 | 按固定行数滑动窗口（30行/块，重叠5行） | 先按页分割，再在每页内用 `RecursiveCharacterTextSplitter` 按 token 分块 |
| **有无页码** | ❌ 无 `page` 字段，只有 `lines` | ✅ 有 `page` 字段，有 `pages` 数组 |
| 分块JSON产物 | `databases/chunked_reports/*.json` 中 `chunks[i]` 包含 `lines` 和 `text` | 同目录，但 `chunks[i]` 包含 `page`、`length_tokens`、`text` |

### 影响
- **父文档检索不生效**：`src/retrieval.py` 第 234-237 行，检索时尝试从 `pages` 数组中匹配父页面，但 MinerU 路径的 JSON 中 `pages` 为空数组，永远跳过。
- **页码校验不生效**：`src/questions_processing.py` 第 95-127 行的 `_validate_page_references()` 依赖 `result['page']`，MinerU 下该值为 `chunk.get("page", 0)` 即恒为 0，校验无实际意义。

### 关键文件
- 分块入口：`src/text_splitter.py` 第 129-172 行 `split_markdown_reports()` 和 `_split_markdown_file()`
- MinerU 路径：滑动窗口在 `src/text_splitter.py` 第 144-153 行
- Docling 路径（备用）：`src/text_splitter.py` 第 75-90 行 `_split_page()`
- PDF→Markdown 调用链：`src/pipeline.py` 第 135-158 行 → `src/pdf_mineru.py`

---

## 二、MinerU 路径下的上下文保全机制

由于没有页码和父文档，上下文保全**全靠以下三层**：

### 1. 滑动窗口重叠分块（防边界截断）
`src/text_splitter.py` 第 148-153 行：30行/块，步进 25 行（重叠 5 行，约 16.7%）。关键信息落在边界时，会同时出现在相邻两个块中。

### 2. CoT 强制分步推理（防幻觉）
每个 Prompt 类（定义在 `src/prompts.py`）都要求 LLM 先输出 `step_by_step_analysis`，再给 `final_answer`：
- number 类最严格：7 条规则（`src/prompts.py` 第 139-195 行的 `AnswerSchema` description），禁止推算、禁止猜测、指标定义必须完全一致、不确定就 N/A

### 3. 答案结构校验 + N/A 兜底
`src/api_requests.py` 第 425-461 行：LLM 返回非 JSON 时，兜底填充 N/A，不抛异常。

---

## 三、重排模块的致命问题：三个方案，唯一接线的那个有 Bug

### 项目中有三套重排方案

| 方案 | 代码位置 | 机制 | 是否被调用 |
|---|---|---|---|
| `LLMReranker` (dashscope) | `src/reranking.py` 第 38-98 行 | Chat API 调用 `qwen-turbo` 做自然语言评分 | ✅ 有调用 |
| `JinaReranker` | `src/reranking.py` 第 10-35 行 | Jina 专用重排 API（`jina-reranker-v2-base-multilingual`） | ❌ 从未 import/实例化 |
| DashScope `TextReRank` | SDK 自带 `dashscope.rerank.TextReRank`，模型 `gte-rerank` | 阿里云百炼专用重排 API（Cross-Encoder） | ❌ 项目代码未引用 |

### LLMReranker 的 Bug：LLM 调了但分数没用

当前实际调用链：
```
pipeline.py 第 313-322 行 (max_config: llm_reranking=True)
  → src/questions_processing.py 第 132-136 行 (创建 HybridRetriever)
    → src/retrieval.py 第 286-289 行 (self.reranker = LLMReranker())
      → src/reranking.py 第 329 行 (rerank_documents, batch_size=10)
```

**Bug 根因**：dashscope provider 下，`get_rank_for_multiple_blocks()`（第 122-142 行）调用 `qwen-turbo` 获取评分文本后，将 `relevance_score` **硬编码为 0.0** 返回，LLM 的实际判断内容被丢弃。

最终融合公式（`src/reranking.py` 第 168-174 行）：
```
combined_score = 0.7 × 0.0 + 0.3 × distance = 0.3 × distance
```
等价于**纯向量距离排序**，且每次查询白白消耗 qwen-turbo 的 token。

代码注释也承认了这一点：
- 第 77 行：`# dashscope 只返回字符串，暂不做结构化解析`
- 第 93 行：`# 这里只返回字符串，后续可按需解析`

### 其他重排方案的状态
- `JinaReranker`（`src/reranking.py` 第 10-35 行）：完整实现了 Jina API 调用（密钥认证、请求体组装、响应解析），但全项目 `src/` 下零引用。`src/retrieval.py` 第 12 行只 `from src.reranking import LLMReranker`，从未导入 `JinaReranker`。
- DashScope `TextReRank`（SDK：`venv/.../dashscope/rerank/text_rerank.py`，模型 `gte-rerank`）：SDK 已安装，API 已就绪，但项目代码从未 `import dashscope.TextReRank` 或调用 `TextReRank.call()`。

---

## 四、检索流程完整链路（MinerU + max_config）

### 粗排：向量检索 top-N（广撒网）
`src/retrieval.py` 第 317-324 行（`HybridRetriever.retrieve_by_company_name`）：
- 调用 `VectorRetriever.retrieve_by_company_name()` 取 `llm_reranking_sample_size`（默认 28）个候选
- 使用 FAISS IndexFlatIP（内积相似度），embedding 由 DashScope `text-embedding-v1` 生成
- 这一步的目标是**高召回**，宁可多捞

### 精排（名义上）：LLMReranker
`src/retrieval.py` 第 328-334 行 → `src/reranking.py` 第 146-224 行：
- 将 28 个候选按 batch_size=10 分批送入 LLM
- **但因 Bug，实际评分全为 0.0，精排退化为纯向量距离排序**

### 答案生成
`src/questions_processing.py` 第 160-168 行：
- 将 top-6 检索结果格式化为 RAG 上下文
- 调用 `APIProcessor.get_answer_from_rag_context()` → 按 schema 匹配对应 Prompt
- 输出结构化 JSON（含 `step_by_step_analysis`、`final_answer`、`relevant_pages`）

### 后处理
- `src/questions_processing.py` 第 95-127 行：页码校验（MinerU 下无效）
- `src/questions_processing.py` 第 385-436 行：提交格式转换（页码 1-based → 0-based）

---

## 五、Pipeline 配置体系

### 预定义 Config（`src/pipeline.py` 第 293-327 行）

| Config 名 | `llm_reranking` | `parent_document_retrieval` | 答题模型 | 用途 |
|---|---|---|---|---|
| `base_config` | False | False | qwen-turbo | 基础配置 |
| `parent_document_retrieval_config` | False | True | gpt-4o | 父文档检索（需 Docling 路径） |
| `max_config` | **True** | True | qwen-turbo | 当前默认配置 |

### 入口
- `pipeline.py` 第 334-340 行：`__main__` 使用 `max_config`
- `app_streamlit.py` 第 3-39 行：Streamlit 界面也使用 `max_config`
- `main.py` 第 50-58 行：CLI 入口 `python main.py process-questions --config max`

---

## 六、值得修复的问题清单

1. **`LLMReranker` dashscope 分支需解析 LLM 返回的评分**：在 `src/reranking.py` 第 88-94 行和第 137-142 行，需将 LLM 返回的 JSON 字符串解析出 `relevance_score`，而不是硬编码 0.0
2. **或者直接改用 DashScope `TextReRank`（`gte-rerank`）**：更合理，因为那是专用重排模型，不需要 prompt 工程，分数直接返回
3. **或者接线 `JinaReranker`**：代码已写好，只需在 `HybridRetriever.__init__` 中替换 `LLMReranker()` 为 `JinaReranker()`
4. **MinerU 路径可补充页码信息**：在 `src/text_splitter.py` 的 `_split_markdown_file()` 中，如果能从 `full.md` 解析出分页标记，可替代 Docling 路径提供页码能力
