"""
Agentic RAG - 向量檢索 Tool（ReAct 版本）
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
load_dotenv()

# 模組載入時建立一次連線並重複使用，避免每次呼叫都重新連線
VECTOR_STORE = get_vector_store()


# ── 將向量檢索結果格式化為可讀的書籍清單 ────────────
# 作為 Tool 回傳給 LLM 的 content
def format_vector_context(vector_docs: list[dict] | None) -> str:
    # vector_docs 為 None 代表 Agent 判斷此問題不需要向量檢索，未執行查詢
    if vector_docs is None:
        return "（本次問題經 Agent 判斷不需要向量檢索，未執行查詢）"

    # 無檢索結果時給 LLM 一個明確提示，避免 LLM 憑空捏造
    if not vector_docs:
        return "（查無相關書籍資料）"

    # lines 暫存每本書格式化後的文字，最後再合併成完整 Context
    lines = []
    # 用計數器 i 為每本書標上編號，從 1 開始
    for i, doc in enumerate(vector_docs, 1):
        # metadata 保存書名、分類、作者、價格與借閱狀態等書籍資訊
        metadata = doc.get("metadata", {})
        # 取出借閱者姓名，用於組合借閱狀態文字
        borrower_name = metadata.get("borrower_name", "")
        # 根據 is_borrowed 產生對應的借閱狀態文字
        if metadata.get("is_borrowed"):
            # 已借出時若有借閱者姓名則一併顯示
            borrowed_text = f"目前已借出（借閱者：{borrower_name}）" if borrower_name else "目前已借出"
        else:
            borrowed_text = "目前可借閱"

        # 將單本書的欄位整理成固定格式，作為 LLM 回答時的參考資料
        lines.append(
            f"【書籍 {i}】{metadata.get('book', '')}\n"
            f"  類別：{metadata.get('category', '')}　作者：{metadata.get('authors', '')}　定價：{metadata.get('price', 0.0)} 元\n"
            f"  借閱狀態：{borrowed_text}\n"
            f"  相似度分數：{doc.get('score', '')}\n"
            f"  相關段落：{doc.get('matched_page_content', '')}"
        )

    # 每本書之間以空行分隔，讓 LLM 能清楚區分不同書籍資料
    return "\n\n".join(lines)


# ── 使用問題向量從 Milvus 向量資料庫檢索書籍 ────────────
# @tool 讓 LLM 能透過 tool_calls 自主呼叫這個函式；例如讀者問「想找主題相近、能讓人放鬆的書」這種語意模糊的推薦需求時，
# LLM 會讀到下面 docstring 寫著「適用於語意相似、模糊需求推薦」而決定呼叫 vector_retrieve；
# response_format="content_and_artifact" 讓回傳拆成兩份：content 給 LLM 讀，artifact 是給程式顯示用的結構化資料，LLM 不會看到
@tool(response_format="content_and_artifact")
def vector_retrieve(query: str, top_k: int = 5) -> tuple[str, list[dict]]:
    """向量相似度檢索書籍，適用於語意相似、模糊需求推薦、內容描述型問題，
    例如：讀者描述情境或需求、想找主題相近的書、書籍講述什麼樣的內容。
    不適合精確實體查詢、結構化過濾或統計型問題（這類問題請改用 graph_retrieve）。

    Args:
        query: 讀者的原始問題或需求描述，若已呼叫過 hyde_query，可改傳其回傳的假想文件。
        top_k: 最多回傳幾本書，預設 5。
    """
    # 同一本書可能有多個 chunks，先多撈 search_k 個再去重，才能湊出 top_k 本不同書
    search_k = max(10, top_k * 3)
    # 將使用者問題轉成問題向量，在 Milvus 中搜尋最相近的 search_k 個 chunks，同時回傳相似度分數
    results = VECTOR_STORE.similarity_search_with_score(query, k=search_k)

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
