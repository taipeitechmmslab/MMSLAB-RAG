"""
Agentic RAG - 向量檢索 Tool（自由式 ReAct 版本）
=================================
vector_tool.py 提供 vector_retrieve() 這個正式的 LangChain Tool，
由 agent.py 中的 LLM 透過 bind_tools() 自主決定要不要呼叫、呼叫幾次。
因此這個 Tool 的 docstring 本身就是 LLM 選擇工具的依據——
docstring 寫得夠清楚，LLM 才能判斷這個工具適不適合目前的問題。

使用 response_format="content_and_artifact"：
  - content  ：格式化過的文字，回傳給 LLM 作為推理依據
  - artifact ：原始資料（list[dict]），保留給 agent.py 還原成結構化結果供 main.py 顯示
"""

# ── 載入套件與環境變數 ──────────────────────────────
from dotenv import load_dotenv
from langchain_core.tools import tool
from clients import get_vector_store
from generation import format_vector_context
load_dotenv()


# ── 使用問題向量從 Milvus 向量資料庫檢索書籍 ────────────
@tool(response_format="content_and_artifact")
def vector_retrieve(query: str, top_k: int = 5) -> tuple[str, list[dict]]:
    """向量相似度檢索書籍，適用於語意相似、模糊需求推薦、內容描述型問題，
    例如：讀者描述情境或需求、想找主題相近的書、書籍講述什麼樣的內容。
    不適合精確實體查詢、結構化過濾或統計型問題（這類問題請改用 graph_retrieve）。

    Args:
        query: 讀者的原始問題或需求描述。
        top_k: 最多回傳幾本書，預設 5。
    """
    # 建立 Milvus 向量資料庫的連線物件
    vector_store = get_vector_store()

    # top_k 代表最後要回傳的書籍數量；search_k 代表先從 Milvus 取回的 chunks 數量
    # 因為同一本書可能有多個 chunks 出現在搜尋結果中，所以先取回較多 chunks
    # 再從中整理出最多 top_k 本不同書籍
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
        # 依 book_id 整理搜尋結果，避免同一本書重複出現在推薦結果中
        if not book_id or book_id in seen_book_ids:
            continue

        # 將此書的 book_id 加入 seen 集合，後續相同書籍的 chunk 會被跳過
        seen_book_ids.add(book_id)
        # 將書籍的 metadata、命中的 chunk 內容、相似度分數加入結果 list
        docs.append(
            {
                # metadata 包含書名、作者、借閱狀態等書籍資訊
                "metadata": doc.metadata,
                # 命中的 chunk 原文，可用來檢視為何此書被檢索到
                "matched_page_content": doc.page_content,
                # 相似度分數，四捨五入至小數點後四位
                "score": round(float(score), 4),
            }
        )

        # 已收集到 top_k 本不同書籍時提前結束迴圈
        if len(docs) >= top_k:
            break
    # content 給 LLM 閱讀；docs（artifact）保留原始資料供 agent.py 還原結構化結果
    return format_vector_context(docs), docs
