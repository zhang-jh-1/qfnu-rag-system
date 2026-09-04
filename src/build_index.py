"""
索引构建脚本 (build_index.py)

功能描述：
1. 自动扫描本地 data/raw/ 目录下的所有规章制度原始文档（PDF/Word/TXT等）。
2. 调用 RAGFlow API 将文档批量上传至指定的知识库 (Dataset)。
3. 触发 RAGFlow 的异步文档解析任务（建议在 RAGFlow 后台预先将该知识库的解析模板设置为 Laws）。
4. 在 data/processed/ 目录下生成文档处理与上传日志，满足实训交付物要求。
"""

import os
import json
import time
from datetime import datetime
from dotenv import load_dotenv
from ragflow_sdk import RAGFlow


def build_index():
    """
    执行文档的批量上传、解析触发以及日志记录的主流程函数。
    """
    # 1. 加载环境变量
    load_dotenv()
    api_key = os.getenv("RAGFLOW_API_KEY")
    base_url = os.getenv("RAGFLOW_BASE_URL", "https://demo.ragflow.io/v1")
    dataset_id = os.getenv("RAGFLOW_DATASET_ID")

    if not all([api_key, dataset_id]):
        print("❌ 错误：请检查 .env 文件，确保已配置 RAGFLOW_API_KEY 和 RAGFLOW_DATASET_ID")
        return

    # 2. 初始化 RAGFlow 客户端并连接知识库
    print("⏳ 正在连接 RAGFlow 服务...")
    try:
        rag_client = RAGFlow(api_key=api_key, base_url=base_url)
        dataset = rag_client.get_dataset(id=dataset_id)
        print(f"✅ 成功连接到知识库: {dataset.name} (ID: {dataset.id})")
    except Exception as e:
        print(f"❌ 无法连接到知识库，请检查网络或 API 配置: {e}")
        return

    # 3. 扫描原始数据目录
    raw_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw')
    processed_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed')

    # 确保输出目录存在
    os.makedirs(processed_dir, exist_ok=True)

    if not os.path.exists(raw_dir):
        print(f"❌ 错误：未找到原始数据目录 {raw_dir}")
        return

    valid_extensions = ('.pdf', '.doc', '.docx', '.txt')
    files_to_upload = []

    for filename in os.listdir(raw_dir):
        if filename.lower().endswith(valid_extensions):
            files_to_upload.append(os.path.join(raw_dir, filename))

    if not files_to_upload:
        print("⚠️ data/raw/ 目录下未找到有效的文档，请先将拆分好的学生手册放入该目录。")
        return

    print(f"📂 扫描到 {len(files_to_upload)} 份待处理文档，准备上传...")

    # 4. 执行上传并触发解析
    upload_log = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "dataset_id": dataset_id,
        "total_files": len(files_to_upload),
        "documents": []
    }

    try:
        # 调用 API 批量上传文件
        documents = dataset.upload_documents(files_to_upload)
        print(f"✅ 成功将 {len(documents)} 份文档上传至 RAGFlow 知识库。")

        # 记录上传成功的文档元数据
        doc_ids = []
        for doc in documents:
            doc_ids.append(doc.id)
            upload_log["documents"].append({
                "doc_id": doc.id,
                "doc_name": doc.name,
                "status": "uploaded"
            })

        # 触发云端的异步文档解析（切块与向量化）
        dataset.async_parse_documents(document_ids=doc_ids)
        print("🚀 异步解析任务已触发！云端正在进行智能切块与向量化...")

    except Exception as e:
        print(f"❌ 文档上传或触发解析失败: {e}")
        upload_log["error"] = str(e)

    # 5. 保存处理日志 (满足实训要求的 data/processed/ 产出物)
    log_path = os.path.join(processed_dir, 'upload_log.json')
    try:
        with open(log_path, 'w', encoding='utf-8') as f:
            json.dump(upload_log, f, ensure_ascii=False, indent=4)
        print(f"📝 文档处理日志已保存至: {log_path}")
    except Exception as e:
        print(f"⚠️ 保存日志失败: {e}")


if __name__ == "__main__":
    build_index()