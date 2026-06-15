"""
Graph RAG - Phase 3：生成回答
=================================
generation.py 負責接收 retrieval.py 回傳的知識圖譜查詢結果，
再請 LLM 根據 Cypher 查詢結果生成繁體中文回答。

此模組提供 generate() 函式供 main.py 呼叫。
"""

# ── 載入套件與環境變數 ──────────────────────────────
import os
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_nvidia_ai_endpoints import ChatNVIDIA
load_dotenv()


# ── 根據知識圖譜檢索結果呼叫 LLM 生成回答 ────────────
def generate(query: str, retrieval_result: dict) -> str:
    # 初始化 NVIDIA NIM LLM
    llm = ChatNVIDIA(
        # 從環境變數取得 LLM 模型名稱
        model=os.environ.get("LLM_MODEL"),
        # 從環境變數取得 NVIDIA API 金鑰
        api_key=os.environ.get("NVIDIA_LLM_API_KEY"),
    )

    # 從 retrieval_result 取出 LLM 產生的 Cypher
    cypher = retrieval_result.get("cypher", "")
    # 取出 Neo4j 查詢結果（dict list）
    results = retrieval_result.get("results", [])
    # 取出錯誤訊息；若查詢成功則為空字串
    error = retrieval_result.get("error", "")

    # 將 Cypher 與查詢結果整理成 LLM 可讀的內容
    if error:
        # 查詢失敗時，附上失敗原因與嘗試執行的 Cypher，讓 LLM 知道查無結果的緣由
        context_block = (
            f"【Cypher 執行狀況】失敗，原因：{error}\n"
            f"【嘗試執行的 Cypher】\n{cypher}"
        )
    else:
        # 查詢成功時，附上執行的 Cypher 與知識圖譜查詢結果作為回答依據
        context_block = (
            f"【執行的 Cypher】\n{cypher}\n\n"
            f"【知識圖譜查詢結果】\n{results}"
        )

    # System Prompt 放入回答規則，限制 LLM 只能根據知識圖譜查詢結果回答
    system_prompt = (
        "你是一位圖書館智慧助理，專門根據 Neo4j 知識圖譜查詢結果回答讀者問題。\n"
        "請嚴格依據提供的知識圖譜查詢結果回答，不可憑空捏造書籍、作者、分類或借閱資訊。\n"
        "若查詢結果包含書籍資料，請根據已提供欄位整理下列資訊：\n"
        "  1. 書名\n"
        "  2. 作者\n"
        "  3. 分類\n"
        "  4. 價格\n"
        "  5. 借閱狀態（是否可借閱；若已借出，請說明是被誰借走）\n"
        "若查詢結果是統計數字、作者列表或其他知識圖譜關係，請直接整理成清楚的繁體中文答案。\n"
        "若查詢結果為空，請明確回覆「目前知識圖譜中查無相關資料」。\n"
        "若 Cypher 執行失敗，請說明目前無法根據知識圖譜查得答案。\n"
        "請使用繁體中文回答，語氣親切、條理清晰。"
    )

    # Human Prompt 放入知識圖譜查詢結果與使用者問題
    human_prompt = (
        f"以下是從 Neo4j 知識圖譜中查詢到的結果：\n\n"
        f"{context_block}\n\n"
        f"讀者問題：{query}\n\n"
        f"請根據以上知識圖譜查詢結果回答讀者的問題。"
    )

    # 組合 System Prompt 與 Human Prompt，呼叫 LLM 生成有資料依據的回答
    response = llm.invoke([
        # System Prompt 設定 LLM 的角色與回答規則
        SystemMessage(content=system_prompt),
        # Human Prompt 傳入知識圖譜查詢結果與使用者問題
        HumanMessage(content=human_prompt),
    ])

    # 回傳 LLM 生成的回答內容
    return response.content
