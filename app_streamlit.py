import streamlit as st
from pathlib import Path
from src.pipeline import Pipeline, max_config
import json
import csv
import time


def live_timer_html(timer_id: str) -> str:
    """生成一个自增的 JS 计时器，在页面加载后自动开始计时。"""
    return f"""<span id='{timer_id}' style='color: #888; font-size: 12px;'>⏱ 0.0s</span>
<script>
(function() {{
    var start = Date.now();
    var el = document.getElementById('{timer_id}');
    if (el) {{
        var interval = setInterval(function() {{
            var s = ((Date.now() - start) / 1000).toFixed(1);
            el.textContent = '⏱ ' + s + 's';
        }}, 100);
    }}
}})();
</script>"""


def stream_text(text: str, delay: float = 0.02):
    """逐字输出文本的生成器，供 st.write_stream 使用。"""
    for char in text:
        yield char
        time.sleep(delay)


def parse_answer_payload(answer):
    """将 pipeline 返回结果规范为展示用 dict。"""
    if isinstance(answer, str):
        answer = json.loads(answer)
    if not isinstance(answer, dict):
        raise ValueError(f"不支持的返回类型: {type(answer)}")

    if "step_by_step_analysis" in answer or "final_answer" in answer:
        return answer

    nested = answer.get("content", answer)
    if isinstance(nested, dict) and (
        "step_by_step_analysis" in nested or "final_answer" in nested
    ):
        return nested

    final_answer = answer.get("final_answer", "")
    if isinstance(final_answer, str) and final_answer.strip().startswith("{"):
        try:
            return json.loads(final_answer)
        except json.JSONDecodeError:
            pass

    return {
        "step_by_step_analysis": answer.get("step_by_step_analysis", ""),
        "reasoning_summary": answer.get("reasoning_summary", ""),
        "relevant_pages": answer.get("relevant_pages", []),
        "final_answer": final_answer if isinstance(final_answer, str) else str(final_answer),
    }


def get_company_list():
    """从 subset.csv 读取公司列表"""
    companies = []
    try:
        with open("data/stock_data/subset.csv", "r", encoding="gbk") as f:
            reader = csv.DictReader(f)
            for row in reader:
                companies.append(row["company_name"])
    except Exception:
        pass
    return companies if companies else ["中芯国际"]


# ---------- 初始化 ----------
root_path = Path("data/stock_data")

@st.cache_resource
def get_pipeline():
    return Pipeline(root_path, run_config=max_config)

pipeline = get_pipeline()

st.set_page_config(page_title="RAG 年报问答", layout="wide")

# ---------- 顶部标题 ----------
st.markdown("""
<div style='background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%); padding: 24px 32px; border-radius: 12px; margin-bottom: 24px;'>
    <h1 style='color: white; margin: 0; font-size: 28px; font-weight: 600;'>RAG 财报智能问答</h1>
    <div style='color: rgba(255,255,255,0.9); font-size: 14px; margin-top: 8px;'>向量检索 + LLM 推理 | 基于年报 RAG | 默认开放性问题（文本回答）</div>
</div>
""", unsafe_allow_html=True)

# ---------- 页面布局：左右两栏 ----------
left_col, right_col = st.columns([1, 2])

# ---------- 左侧标题 ----------
with left_col:
    st.markdown("**查询设置**")
    st.markdown("<hr style='margin: 8px 0 16px 0; border-color: #e0e0e0;'>", unsafe_allow_html=True)

# ---------- 右侧标题 ----------
with right_col:
    st.markdown("**检索与回答**")
    st.markdown("<hr style='margin: 8px 0 16px 0; border-color: #e0e0e0;'>", unsafe_allow_html=True)

# ---------- 左侧：输入区 ----------
with left_col:
    # 选择公司
    st.markdown("**选择公司**")
    company_list = get_company_list()
    selected_company = st.selectbox(
        "选择公司",
        options=company_list,
        label_visibility="collapsed"
    )
    
    # 输入问题
    st.markdown("**输入问题**")
    user_question = st.text_area(
        "输入问题",
        "中芯国际在晶圆制造行业中的地位如何？其服务范围和全球布局是怎样的？",
        height=100,
        label_visibility="collapsed"
    )
    
    # 问题类型
    st.markdown("**问题类型**")
    question_type = st.radio(
        "问题类型",
        options=["text", "boolean", "number", "name"],
        horizontal=True,
        label_visibility="collapsed"
    )
    
    # 高级选项
    st.markdown("**高级选项**")
    use_llm_rerank = st.checkbox("启用 LLM 重排序", value=True)
    use_stream = st.checkbox("启用流式输出（降低首 token 延迟）", value=True)
    use_neighbor_chunks = st.checkbox("启用邻域 Chunk（检索结果 ±1 邻居扩展）", value=True)
    
    st.markdown("**检索文档数量**")
    doc_count = st.slider(
        "检索文档数量",
        min_value=1,
        max_value=15,
        value=10,
        label_visibility="collapsed"
    )
    
    # 按钮
    st.markdown("")
    col1, col2 = st.columns(2)
    with col1:
        search_btn = st.button("搜索文档", use_container_width=True)
    with col2:
        submit_btn = st.button("生成答案", use_container_width=True, type="primary")

# ---------- 右侧：结果展示区 ----------
with right_col:
    st.markdown("<h2 style='margin-top: 0;'>最终答案</h2>", unsafe_allow_html=True)
    
    # 初始化 session state 用于流式展示
    if "streaming" not in st.session_state:
        st.session_state.streaming = False
    if "final_answer_text" not in st.session_state:
        st.session_state.final_answer_text = ""
    if "step_by_step_text" not in st.session_state:
        st.session_state.step_by_step_text = ""
    if "reasoning_summary_text" not in st.session_state:
        st.session_state.reasoning_summary_text = ""
    if "retrieval_info" not in st.session_state:
        st.session_state.retrieval_info = None
    if "retrieval_summary" not in st.session_state:
        st.session_state.retrieval_summary = ""
    if "relevant_pages" not in st.session_state:
        st.session_state.relevant_pages = []
    if "references" not in st.session_state:
        st.session_state.references = []
    
    if submit_btn and user_question.strip():
        # 重置状态
        st.session_state.streaming = True
        st.session_state.stream_start_time = time.time()
        st.session_state.stream_elapsed = 0.0
        st.session_state.final_answer_text = ""
        st.session_state.step_by_step_text = ""
        st.session_state.reasoning_summary_text = ""
        st.session_state.retrieval_info = None
        st.session_state.relevant_pages = []
        st.session_state.references = []
        
        # 先显示空框（占位）- 带计时器
        st.markdown("**最终答案**")
        answer_container = st.empty()
        answer_container.markdown(
            f"""<div style='background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%); padding: 20px; border-radius: 12px; border: 1px solid #0f3460; min-height: 100px; display: flex; align-items: center; justify-content: center; flex-direction: column;'>
                <span style='color: #4a6fa5; font-size: 14px;'>🤔 正在思考中...</span>
                <span style='margin-top: 8px;'>{live_timer_html('timer-answer-loading')}</span>
            </div>""",
            unsafe_allow_html=True
        )
        
        st.markdown("**分步推理**")
        reasoning_container = st.empty()
        reasoning_container.markdown(
            f"""<div style='background: linear-gradient(135deg, #2d2d2d 0%, #1a1a1a 100%); padding: 16px; border-radius: 8px; border: 1px solid #3d3d3d; min-height: 80px; display: flex; align-items: center; justify-content: center; flex-direction: column;'>
                <span style='color: #666; font-size: 13px;'>📝 推理过程准备中...</span>
                <span style='margin-top: 6px;'>{live_timer_html('timer-reasoning-loading')}</span>
            </div>""",
            unsafe_allow_html=True
        )
        
        st.markdown("**推理摘要**")
        summary_container = st.empty()
        summary_container.markdown(
            f"""<div style='background: linear-gradient(135deg, #2d2d2d 0%, #1a1a1a 100%); padding: 16px; border-radius: 8px; border: 1px solid #3d3d3d; min-height: 60px; display: flex; align-items: center; justify-content: center; flex-direction: column;'>
                <span style='color: #666; font-size: 13px;'>📋 摘要生成中...</span>
                <span style='margin-top: 6px;'>{live_timer_html('timer-summary-loading')}</span>
            </div>""",
            unsafe_allow_html=True
        )
        
        # 检索统计（折叠，默认收起）
        retrieval_expander = st.expander("🔍 检索统计", expanded=False)
        with retrieval_expander:
            st.caption("正在检索...")
            st.markdown(live_timer_html('timer-retrieval-loading'), unsafe_allow_html=True)
        
        # 相关页码和引用
        col_pages, col_refs = st.columns(2)
        with col_pages:
            st.markdown("**相关页码**")
            pages_container = st.empty()
            pages_container.code("[ ]", language="json")
        with col_refs:
            st.markdown("**引用**")
            refs_container = st.empty()
            refs_container.code("[ ]", language="json")
        
        # 执行检索和生成（流式）
        with st.spinner("正在检索..."):
            try:
                full_question = f"{selected_company}：{user_question}"
                kind = "string" if question_type == "text" else question_type
                # 短暂延迟，让加载状态可见
                time.sleep(0.8)
                stream_gen = pipeline.answer_single_question_stream(
                    full_question,
                    kind=kind,
                    use_neighbor_chunks=use_neighbor_chunks,
                    top_n_retrieval=doc_count
                )
                
                # 实时接收并显示 LLM 流式输出
                full_text = ""
                for item in stream_gen:
                    if item["type"] == "retrieval":
                        # 检索完成，立即显示检索结果到分步推理和推理摘要框
                        retrieval_elapsed = time.time() - st.session_state.stream_start_time
                        summary = item.get("retrieval_summary", "")
                        info = item.get("retrieval_info", {})
                        
                        # 分步推理 → 显示检索到的相关文档片段
                        reasoning_container.markdown(
                            f"<div style='background: #f0f5ff; padding: 16px; border-radius: 8px; border: 1px solid #d6e4ff; font-size: 13px; line-height: 1.7; color: #333;'>"
                            f"<b>🔍 检索结果（共 {info.get('total_chunks', '?')} 条，涉及 {info.get('unique_pages', '?')} 页）</b><br><br>"
                            f"{summary.replace(chr(10), '<br>')}"
                            f"</div>"
                            f"<div style='text-align: right; color: #999; font-size: 12px; margin-top: 4px;'>⏱ {retrieval_elapsed:.1f}s 检索完成</div>",
                            unsafe_allow_html=True
                        )
                        
                        # 推理摘要 → 显示检索统计
                        summary_container.markdown(
                            f"<div style='background: #f0f5ff; padding: 16px; border-radius: 8px; border: 1px solid #d6e4ff; font-size: 13px; line-height: 1.6; color: #333;'>"
                            f"<b>📊 检索统计</b><br>"
                            f"配置 top_n = {info.get('top_n', '?')}<br>"
                            f"邻域扩展 = {'✅ 启用' if info.get('neighbor_expansion') else '❌ 关闭'}<br>"
                            f"实际 chunk 数 = {info.get('total_chunks', '?')}<br>"
                            f"涉及唯一页面 = {info.get('unique_pages', '?')}<br>"
                            f"<span style='color: #4a6fa5;'>🤔 等待 LLM 生成答案...</span>"
                            f"</div>"
                            f"<div style='text-align: right; color: #999; font-size: 12px; margin-top: 4px;'>⏱ {retrieval_elapsed:.1f}s</div>",
                            unsafe_allow_html=True
                        )
                        
                        # 检索统计折叠栏
                        with retrieval_expander:
                            st.caption(f"配置 top_n = {info['top_n']}  |  邻域扩展 = {'✅ 启用' if info['neighbor_expansion'] else '❌ 关闭'}")
                            st.caption(f"实际投喂 chunk 数 = {info['total_chunks']}  |  涉及唯一页面数 = {info['unique_pages']}")
                        
                         # 存储到 session state（供后续使用）
                        st.session_state.retrieval_info = info
                        st.session_state.retrieval_summary = summary
                    elif item["type"] == "token":
                        full_text += item["content"]
                        elapsed = time.time() - st.session_state.stream_start_time
                        st.session_state.stream_elapsed = elapsed
                        # 实时更新显示（带计时器）
                        display_text = full_text if full_text.strip() else "📝 正在生成回答..."
                        answer_container.markdown(
                            f"<div style='background: #f8f4ff; padding: 20px; border-radius: 12px; border: 1px solid #e0d4f0; font-size: 16px; line-height: 1.8; color: #333;'>"
                            f"{display_text}"
                            f"</div>"
                            f"<div style='text-align: right; color: #999; font-size: 12px; margin-top: 4px;'>⏱ {elapsed:.1f}s</div>",
                            unsafe_allow_html=True
                        )
                    elif item["type"] == "done":
                        answer = item["answer"]
                        retrieval_info = item.get("retrieval_info")
                
                # 解析结构化答案
                try:
                    content = parse_answer_payload(answer)
                except (json.JSONDecodeError, ValueError) as e:
                    st.error(f"返回内容无法解析为结构化答案：{e}")
                    content = {}

                # 存储到 session state
                st.session_state.final_answer_text = str(content.get("final_answer") or "-")
                st.session_state.step_by_step_text = str(content.get("step_by_step_analysis") or "-")
                st.session_state.reasoning_summary_text = str(content.get("reasoning_summary") or "-")
                st.session_state.relevant_pages = content.get("relevant_pages", [])
                st.session_state.references = content.get("references", [])
                st.session_state.retrieval_info = retrieval_info
                
            except Exception as e:
                st.error(f"生成答案时出错: {e}")
                st.session_state.streaming = False
        
        # 检索完成后更新统计信息
        if st.session_state.retrieval_info:
            with retrieval_expander:
                info = st.session_state.retrieval_info
                st.caption(f"配置 top_n = {info['top_n']}  |  邻域扩展 = {'✅ 启用' if info['neighbor_expansion'] else '❌ 关闭'}")
                st.caption(f"实际投喂 chunk 数 = {info['total_chunks']}  |  涉及唯一页面数 = {info['unique_pages']}")
                if info['neighbor_expansion'] and info['total_chunks'] > info['top_n']:
                    st.success(f"邻域扩展生效：原 {info['top_n']} 条 → 扩展为 {info['total_chunks']} 条")
                elif info['neighbor_expansion'] and info['total_chunks'] <= info['top_n']:
                    st.warning("邻域扩展可能未生效（chunk 数未增加）")
        
        # 流式结束后展示结构化结果
        if st.session_state.streaming:
            total_elapsed = st.session_state.get("stream_elapsed", time.time() - st.session_state.stream_start_time)
            
            # 最终答案
            answer_container.markdown(
                f"<div style='background: #f8f4ff; padding: 20px; border-radius: 12px; border: 1px solid #e0d4f0; font-size: 16px; line-height: 1.8; color: #333;'>"
                f"{st.session_state.final_answer_text}"
                f"</div>"
                f"<div style='text-align: right; color: #999; font-size: 12px; margin-top: 4px;'>⏱ 总耗时 {total_elapsed:.1f}s</div>",
                unsafe_allow_html=True
            )
            
            # 分步推理：替换为 LLM 的分步推理结果
            reasoning_container.markdown(
                f"<div style='background: #f0f5ff; padding: 16px; border-radius: 8px; border: 1px solid #d6e4ff; font-size: 13px; line-height: 1.7; color: #333;'>"
                f"<b>🔍 分步推理</b><br><br>"
                f"{st.session_state.step_by_step_text}"
                f"</div>"
                f"<div style='text-align: right; color: #999; font-size: 12px; margin-top: 4px;'>⏱ 总耗时 {total_elapsed:.1f}s</div>",
                unsafe_allow_html=True
            )
            # 推理摘要：显示 LLM 的推理摘要
            summary_container.markdown(
                f"<div style='background: #f5f5f5; padding: 16px; border-radius: 8px; font-size: 14px; line-height: 1.6; color: #666;'>"
                f"{st.session_state.reasoning_summary_text}"
                f"</div>"
                f"<div style='text-align: right; color: #999; font-size: 12px; margin-top: 4px;'>⏱ 总耗时 {total_elapsed:.1f}s</div>",
                unsafe_allow_html=True
            )
            
            # 更新页码和引用
            if st.session_state.relevant_pages:
                pages_str = ", ".join([str(p) for p in st.session_state.relevant_pages])
                pages_container.code(f"[ {pages_str} ]", language="json")
            else:
                pages_container.code("[ ]", language="json")
            
            if st.session_state.references:
                refs_container.code(json.dumps(st.session_state.references, indent=2, ensure_ascii=False), language="json")
            else:
                refs_container.code("[ ]", language="json")
            
            st.session_state.streaming = False
    
    elif st.session_state.get("final_answer_text"):
        # 显示上次的结果（非流式，直接展示）
        st.markdown("**最终答案**")
        st.markdown(
            f"<div style='background: #f8f4ff; padding: 20px; border-radius: 12px; border: 1px solid #e0d4f0; font-size: 16px; line-height: 1.8; color: #333;'>"
            f"{st.session_state.final_answer_text}"
            f"</div>",
            unsafe_allow_html=True
        )
        
        st.markdown("**分步推理**")
        if st.session_state.get("step_by_step_text"):
            st.markdown(
                f"<div style='background: #f0f5ff; padding: 16px; border-radius: 8px; border: 1px solid #d6e4ff; font-size: 13px; line-height: 1.7; color: #333;'>"
                f"<b>🔍 分步推理</b><br><br>"
                f"{st.session_state.step_by_step_text}"
                f"</div>",
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"<div style='background: #f5f5f5; padding: 16px; border-radius: 8px; font-size: 14px; line-height: 1.6; color: #666;'>"
                f"-</div>",
                unsafe_allow_html=True
            )
        
        st.markdown("**推理摘要**")
        st.markdown(
            f"<div style='background: #f5f5f5; padding: 16px; border-radius: 8px; font-size: 14px; line-height: 1.6; color: #666;'>"
            f"{st.session_state.reasoning_summary_text}"
            f"</div>",
            unsafe_allow_html=True
        )
        
        if st.session_state.retrieval_info:
            with st.expander(f"🔍 检索统计", expanded=False):
                info = st.session_state.retrieval_info
                st.caption(f"配置 top_n = {info['top_n']}  |  邻域扩展 = {'✅ 启用' if info['neighbor_expansion'] else '❌ 关闭'}")
                st.caption(f"实际投喂 chunk 数 = {info['total_chunks']}  |  涉及唯一页面数 = {info['unique_pages']}")
        
        col_pages, col_refs = st.columns(2)
        with col_pages:
            st.markdown("**相关页码**")
            if st.session_state.relevant_pages:
                pages_str = ", ".join([str(p) for p in st.session_state.relevant_pages])
                st.code(f"[ {pages_str} ]", language="json")
            else:
                st.code("[ ]", language="json")
        with col_refs:
            st.markdown("**引用**")
            if st.session_state.references:
                st.code(json.dumps(st.session_state.references, indent=2, ensure_ascii=False), language="json")
            else:
                st.code("[ ]", language="json")
    
    else:
        # 默认空状态
        st.markdown(
            """
            <div style='background: #f8f4ff; padding: 20px; border-radius: 12px; border: 1px solid #e0d4f0; margin-bottom: 24px; min-height: 100px;'>
                <div style='font-size: 14px; color: #999; text-align: center; padding-top: 30px;'>
                    在左侧输入问题并点击"生成答案"即可开始查询
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        st.markdown("**分步推理**")
        st.markdown(
            """
            <div style='background: #f5f5f5; padding: 16px; border-radius: 8px; margin-bottom: 16px; min-height: 60px;'>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        st.markdown("**推理摘要**")
        st.markdown(
            """
            <div style='background: #f5f5f5; padding: 16px; border-radius: 8px; margin-bottom: 16px; min-height: 60px;'>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        col_pages, col_refs = st.columns(2)
        with col_pages:
            st.markdown("**相关页码**")
            st.code("[ ]", language="json")
        with col_refs:
            st.markdown("**引用**")
            st.code("[ ]", language="json")
