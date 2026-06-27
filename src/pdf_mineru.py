"""
MinerU API v4 调用模块。
支持：
- 环境变量 MINERU_API_KEY 读取 Token
- page_ranges 参数分片（解决 200 页限制）
- content_list.json 解析，提取逐页内容
"""
import os
import json
import requests
import time
import zipfile
from pathlib import Path
from typing import Optional

# ---------- 配置 ----------
API_KEY = os.getenv("MINERU_API_KEY")

def _get_api_key() -> str:
    key = API_KEY or os.getenv("MINERU_API_KEY")
    if not key:
        raise RuntimeError("请在环境变量中设置 MINERU_API_KEY")
    return key

def _api_headers() -> dict:
    return {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {_get_api_key()}",
    }


def submit_task(file_name: str, page_ranges: Optional[str] = None) -> str:
    """
    提交 MinerU 解析任务。
    :param file_name: OSS 上的 PDF 文件名（如 '【财报】中芯国际：中芯国际2024年年度报告.pdf'）
    :param page_ranges: 页码范围，如 "1-200"，None 表示全部
    :return: task_id
    """
    url = "https://mineru.net/api/v4/extract/task"
    pdf_url = "https://vl-image.oss-cn-shanghai.aliyuncs.com/pdf/" + file_name

    data = {
        "url": pdf_url,
        "is_ocr": True,
        "enable_formula": False,
        "model_version": "pipeline",
        "no_cache": False,  # 允许使用缓存（之前解析过可直接返回）
    }
    if page_ranges:
        data["page_ranges"] = page_ranges

    res = requests.post(url, headers=_api_headers(), json=data)
    print(f"[submit_task] status={res.status_code}, body={res.json()}")
    task_id = res.json()["data"]["task_id"]
    return task_id


def wait_and_download(task_id: str, output_dir: Optional[Path] = None) -> Path:
    """
    轮询任务状态，完成后下载 zip 并解压。
    :return: 解压后的目录路径
    """
    query_url = f"https://mineru.net/api/v4/extract/task/{task_id}"

    while True:
        res = requests.get(query_url, headers=_api_headers())
        result = res.json()["data"]
        state = result.get("state")
        err_msg = result.get("err_msg", "")

        if state in ("pending", "running"):
            progress = result.get("extract_progress", {})
            extracted = progress.get("extracted_pages", "?")
            total = progress.get("total_pages", "?")
            print(f"[{task_id}] 状态={state}  已解析 {extracted}/{total} 页，等待 5s...")
            time.sleep(5)
            continue

        if err_msg:
            raise RuntimeError(f"任务 {task_id} 失败: {err_msg}")

        if state == "done":
            full_zip_url = result.get("full_zip_url")
            if not full_zip_url:
                raise RuntimeError(f"任务 {task_id} 完成但无 full_zip_url")

            zip_path = Path(f"{task_id}.zip")
            print(f"[{task_id}] 开始下载: {full_zip_url}")
            r = requests.get(full_zip_url, stream=True)
            with open(zip_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            print(f"[{task_id}] 下载完成: {zip_path}")

            extract_dir = output_dir or Path(task_id)
            _unzip(zip_path, extract_dir)
            return extract_dir

        raise RuntimeError(f"未知状态: {state}")


def _unzip(zip_path: Path, extract_dir: Path):
    extract_dir = Path(extract_dir)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_dir)
    print(f"已解压到: {extract_dir}")


def parse_content_list_to_pages(content_list_path: Path, page_offset: int = 0) -> list[dict]:
    """
    解析 MinerU 输出的 content_list.json，提取每页聚合文本。
    返回 [{"page": int, "text": str}, ...]，page 从 1 开始。
    :param content_list_path: content_list.json 路径
    :param page_offset: 页码偏移（分片时使用，如第二片起始页从 page_offset+1 开始）
    """
    with open(content_list_path, "r", encoding="utf-8") as f:
        content_list = json.load(f)

    pages_dict: dict[int, list[str]] = {}

    for item in content_list:
        page_idx = item.get("page_idx", 0)  # 0-based
        item_type = item.get("type", "")

        # 提取文本内容
        text = ""
        if "text" in item and item["text"]:
            text = item["text"]
        elif "content" in item and item["content"]:
            text = item["content"]

        # 表格特殊处理：用 html table 表示
        if item_type == "table" and "table_body" in item:
            text = _table_to_html(item)

        if not text.strip():
            continue

        if page_idx not in pages_dict:
            pages_dict[page_idx] = []
        pages_dict[page_idx].append(text)

    # 按页码排序，聚合每页文本
    pages = []
    for page_idx in sorted(pages_dict.keys()):
        page_text = "\n\n".join(pages_dict[page_idx])
        actual_page = page_idx + 1 + page_offset
        pages.append({"page": actual_page, "text": page_text})

    return pages


def _table_to_html(table_item: dict) -> str:
    """将 MinerU 的 table_body 转为 HTML 表格字符串"""
    table_body = table_item.get("table_body", "")
    if not table_body:
        return ""

    # table_body 可能是 HTML 字符串（如 "<table>...</table>" 或 "<td>...</td>"）
    if isinstance(table_body, str):
        return table_body.strip()

    # 也可能是 list of lists of dicts/cells
    if isinstance(table_body, list):
        rows_html = []
        for row in table_body:
            if isinstance(row, str):
                rows_html.append(row)
            else:
                cells = ""
                for cell in row:
                    if isinstance(cell, str):
                        cells += f"<td>{cell}</td>"
                    elif isinstance(cell, dict):
                        cells += f"<td>{cell.get('content', '')}</td>"
                rows_html.append(f"<tr>{cells}</tr>")
        return f"<table>{''.join(rows_html)}</table>"

    return str(table_body)


def find_content_list_json(extract_dir: Path) -> Path:
    """在解压目录中查找 _content_list.json 文件"""
    candidates = list(extract_dir.glob("*_content_list.json"))
    if not candidates:
        candidates = list(extract_dir.rglob("*_content_list.json"))
    if not candidates:
        raise FileNotFoundError(f"在 {extract_dir} 中未找到 content_list.json")
    return candidates[0]


def find_full_md(extract_dir: Path) -> Path:
    """在解压目录中查找 full.md 文件"""
    candidates = list(extract_dir.glob("**/full.md"))
    if not candidates:
        raise FileNotFoundError(f"在 {extract_dir} 中未找到 full.md")
    return candidates[0]
