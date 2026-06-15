"""
Vector RAG - Phase 2：向量檢索
=================================
retrieval.py 負責將使用者問題轉成問題向量，並從 Milvus 找出最相關的書籍資料。

此模組提供 retrieve() 函式供 main.py 呼叫。
"""

# ── 載入套件與環境變數 ──────────────────────────────
import os
from dotenv import load_dotenv
from langchain_milvus import Milvus
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
load_dotenv()

# ── 使用問題向量從 Milvus 向量資料庫檢索書籍 ────────────
def retrieve(query: str, top_k: int = 3) -> list[dict]:
    # 建立 Milvus 向量資料庫的連線物件
    vector_store = Milvus(
        # 問題向量需與建索引時使用相同的 Embedding Model
        embedding_function=NVIDIAEmbeddings(
            # 從環境變數取得 Embedding 模型名稱
            model=os.environ.get("EMBEDDING_MODEL"),
            # 從環境變數取得 NVIDIA API 金鑰
            api_key=os.environ.get("NVIDIA_LLM_API_KEY"),
        ),
        # 指定要查詢的 Milvus collection 名稱
        collection_name="library_books",
        # Milvus 服務的連線位址
        connection_args={"uri": "http://localhost:19530"},
        # 與建索引時一致開啟動態欄位，搜尋才會把 metadata 一起帶回
        enable_dynamic_field=True,
    )

    # 同一本書可能有多個 chunks，先多撈 search_k 個再去重，才能湊出 top_k 本不同書
    # search_k 至少為 10，或 top_k 的 3 倍，以免去重後書籍數量不足
    search_k = max(10, top_k * 3)

    # 將使用者問題轉成問題向量，在 Milvus 中搜尋最相近的 search_k 個 chunks，同時回傳相似度分數
    results = vector_store.similarity_search_with_score(query, k=search_k)

    # 用 set 記錄已出現的 book_id，避免同一本書重複加入結果
    seen_book_ids = set()
    # 初始化結果 list，存放最終回傳的書籍資訊
    docs = []
    # results 已依相似度由高到低排序，每本書取第一次出現的（即最相關的 chunk）
    for doc, score in results:
        # 從 metadata 取得此 chunk 對應的書籍 ID
        book_id = doc.metadata.get("book_id")
        # 若 book_id 不存在或該書已加入結果，則跳過此 chunk
        if not book_id or book_id in seen_book_ids:
            continue

        # 將此書的 book_id 加入 seen 集合，後續相同書籍的 chunk 會被跳過
        seen_book_ids.add(book_id)
        # 將書籍的 metadata、命中的 chunk 內容、相似度分數加入結果 list
        docs.append({
            # metadata 包含書名、作者、借閱狀態等書籍資訊
            "metadata": doc.metadata,
            # 命中的 chunk 原文，可用來檢視為何此書被檢索到
            "matched_page_content": doc.page_content,
            # 相似度分數，四捨五入至小數點後四位
            "score": round(float(score), 4),
        })

        # 已收集到 top_k 本不同書籍時提前結束迴圈
        if len(docs) >= top_k:
            break

    # 回傳最多 top_k 本書籍的檢索結果
    return docs
