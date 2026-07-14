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
        # 從環境變數取得 NVIDIA NIM API 金鑰
        api_key=os.environ.get("NVIDIA_NIM_API_KEY"),
        # 最多等待 LLM 回應 60 秒，逾時後會拋出 Timeout 例外
        timeout=60,
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
        "你是圖書館智慧助理，只能根據提供的 Neo4j 知識圖譜查詢結果回答，不得捏造資訊。\n"
        "請先判斷問題類型，且只套用一種回答格式。\n"
        "【書籍清單】\n"
        "先用一至兩句直接回答問題。接著列出所有已提供且符合條件的書籍，每本以阿拉伯數字編號，並各自換行列出已有資料中的書名、作者、分類、價格與借閱狀態。每本書之間保留一個空行；沒有提供的欄位不可自行補寫。\n"
        "【統計、關係或單一事實】\n"
        "直接用短句整理統計數字、作者列表或書籍關係；若有多筆結果，可用阿拉伯數字編號列出。不要套用書籍欄位格式，也不要加入無資料依據的推薦理由。\n"
        "【查無資料或查詢失敗】\n"
        "查無資料時回答「目前知識圖譜中查無相關資料」；Cypher 執行失敗時回答「目前無法根據知識圖譜查得答案」。\n"
        "使用繁體中文，語氣親切、條理清晰。回答直接從內容開始，不得輸出「AI 回答：」或其他固定前綴。\n"
        "只能使用純文字、換行與阿拉伯數字編號；不得使用 Markdown、粗體、斜體、標題、程式碼區塊、表格、分隔線或項目符號。"
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
