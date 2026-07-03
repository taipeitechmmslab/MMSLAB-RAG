"""
Agentic RAG - HyDE 假想文件 Tool
=================================
hyde_tool.py 提供 hyde_query() 這個正式的 LangChain Tool，
由 agent.py 中的 LLM 透過 bind_tools() 自主決定要不要呼叫、呼叫幾次。
因此這個 Tool 的 docstring 本身就是 LLM 選擇工具的依據——
docstring 寫得夠清楚，LLM 才能判斷這個工具適不適合目前的問題。

hyde_query() 是 pre-retrieval 工具，只生成假想文件文字，本身不查詢資料庫；
回傳的文字會作為 vector_retrieve 的 query 參數使用。

使用 response_format="content_and_artifact"：
  - content  ：格式化過的文字，回傳給 LLM 作為推理依據
  - artifact ：原始資料（str），保留給 agent.py 還原成結構化結果供 main.py 顯示
"""

# ── 載入套件與環境變數 ──────────────────────────────
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from clients import get_llm
load_dotenv()


# ── 生成 HyDE 假想文件工具 ────────────
@tool(response_format="content_and_artifact")
def hyde_query(query: str) -> tuple[str, str]:
    """HyDE（假設性文件生成），適用於讀者問題較抽象、籠統，直接用原始問題檢索
    效果可能不佳的情境。生成一段可能出現在相關書籍簡介中的內容，
    再將回傳的文字作為 vector_retrieve 的 query 參數使用，取代原始問題去做向量檢索。
    本身不查詢資料庫、不回答問題，也不會編造真實書名或館藏資料。

    Args:
        query: 讀者的原始問題。
    """
    llm = get_llm()
    # System Prompt 限制 LLM 只生成適合向量檢索的描述，避免直接回答問題
    system_prompt = (
        "你是圖書館館藏檢索助理。"
        "請根據讀者問題，生成一段可能出現在相關書籍資料中的內容介紹。"
        "不要回答問題，不要列出真實書名，不要編造館藏資料。"
        "只需要生成一段適合向量檢索的繁體中文描述。"
    )
    human_prompt = f"讀者問題：{query}"

    response = llm.invoke(
        [
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt),
        ]
    )
    hypothetical_document = response.content.strip()
    # content 給 LLM 閱讀；artifact 保留同一份文字供 main.py 顯示
    return hypothetical_document, hypothetical_document
