"""
Hybrid RAG - Phase 3：生成整合回答
=================================
generation.py 將 Milvus 向量檢索結果與 Neo4j 知識圖譜查詢結果合併成單一 context，
並呼叫 LLM 產生單一繁體中文答案。

此模組提供 generate() 函式供 main.py 呼叫。
"""

# ── 載入套件與環境變數 ──────────────────────────────
import os
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_nvidia_ai_endpoints import ChatNVIDIA
load_dotenv()


# ── 將知識圖譜查詢結果格式化為 LLM 可讀的文字 ────────────
def format_graph_context(graph_result: dict) -> str:
    # 從 graph_result 取出 LLM 產生的 Cypher
    cypher = graph_result.get("cypher", "")
    # 取出 Neo4j 查詢結果（dict list）
    results = graph_result.get("results", [])
    # 取出錯誤訊息；若查詢成功則為空字串
    error = graph_result.get("error", "")

    # 查詢失敗時，附上失敗原因與嘗試執行的 Cypher，讓 LLM 知道查無結果的緣由
    if error:
        return (
            f"【Cypher 執行狀況】失敗，原因：{error}\n"
            f"【嘗試執行的 Cypher】\n{cypher}"
        )

    # 查詢成功時，附上執行的 Cypher 與知識圖譜查詢結果作為回答依據
    return (
        f"【執行的 Cypher】\n{cypher}\n\n"
        f"【知識圖譜查詢結果】\n{results}"
    )


# ── 將向量檢索結果格式化為可讀的書籍清單 ────────────
def format_vector_context(vector_docs: list[dict]) -> str:
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
            f"  相似距離分數：{doc.get('score', '')}\n"
            f"  相關段落：{doc.get('matched_page_content', '')}"
        )

    # 每本書之間以空行分隔，讓 LLM 能清楚區分不同書籍資料
    return "\n\n".join(lines)


# ── 整合向量檢索與知識圖譜結果呼叫 LLM 生成回答 ────────────
def generate(query: str, vector_docs: list[dict], graph_result: dict) -> str:
    # 將知識圖譜查詢結果格式化為 LLM 可讀的文字
    graph_context = format_graph_context(graph_result)
    # 將向量檢索結果格式化為可讀的書籍清單
    vector_context = format_vector_context(vector_docs)
    # 合併兩組檢索結果為單一 context，並以標題清楚區分來源
    combined_context = (
        "===== 知識圖譜查詢結果 =====\n"
        f"{graph_context}\n\n"
        "===== 向量檢索結果 =====\n"
        f"{vector_context}"
    )

    # 初始化 NVIDIA NIM LLM
    llm = ChatNVIDIA(
        # 從環境變數取得 LLM 模型名稱
        model=os.environ.get("LLM_MODEL"),
        # 從環境變數取得 NVIDIA NIM API 金鑰
        api_key=os.environ.get("NVIDIA_NIM_API_KEY"),
    )

    # System Prompt 放入回答規則，限制 LLM 只能根據提供的兩組資料回答
    system_prompt = (
        "你是一位圖書館智慧助理，專門整合 Neo4j 知識圖譜與 Milvus 向量檢索結果回答讀者問題。\n"
        "請嚴格依據提供的兩組資料回答，不可憑空捏造書籍、作者、分類、價格或借閱資訊。\n"
        "判斷證據優先順序時請遵守：\n"
        "1. 作者、借閱狀態、分類、價格、統計數字等精確事實型問題，以知識圖譜查詢結果為主，向量結果只作描述性補充。\n"
        "2. 模糊語意、情境建議、內容描述、推薦類問題，以向量檢索結果為主，知識圖譜提供結構資訊補充。\n"
        "3. 若兩邊結果互相矛盾，請優先採用知識圖譜中的結構化事實，並避免使用無法確認的資訊。\n"
        "若回答包含書籍，請針對每本書明確列出：書名、作者、分類、價格、借閱狀態、推薦理由。\n"
        "若兩邊皆無相關資料，請明確回覆「目前資料庫中無相關書籍」。\n"
        "若 Cypher 執行失敗，請說明目前無法由知識圖譜查得，並僅依向量結果回答。\n"
        "請使用繁體中文回答，語氣親切、條理清晰。"
    )

    # Human Prompt 放入兩組檢索結果與使用者問題
    human_prompt = (
        f"以下是 Hybrid RAG 檢索到的兩組資料：\n\n"
        f"{combined_context}\n\n"
        f"讀者問題：{query}\n\n"
        f"請根據以上資料產生單一整合答案。"
    )

    # 組合 System Prompt 與 Human Prompt，呼叫 LLM 生成有資料依據的回答
    response = llm.invoke(
        [
            # System Prompt 設定 LLM 的角色與回答規則
            SystemMessage(content=system_prompt),
            # Human Prompt 傳入兩組檢索結果與使用者問題
            HumanMessage(content=human_prompt),
        ]
    )

    # 回傳 LLM 生成的回答內容
    return response.content
