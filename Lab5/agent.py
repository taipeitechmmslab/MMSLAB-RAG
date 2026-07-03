"""
Agentic RAG - Agent 控制迴圈
=================================
agent.py 使用 LangChain 的 create_agent() 組出 ReAct 風格的 Agentic RAG：
LLM 透過 bind_tools() 拿到 hyde_query()、vector_retrieve()、graph_retrieve() 三個正式的
LangChain Tool，每一輪自己決定要不要呼叫工具、呼叫哪一個、呼叫幾次，
直到它認為已經蒐集到足夠資料，才輸出不帶 tool_calls 的最終答案。

這裡沒有固定的節點順序：要不要檢索、檢索幾次、何時停止全部由 LLM 自主決定，
路由與品質判斷的知識已寫進 Tool 的 docstring 與下方 SYSTEM_PROMPT，供 LLM 參考。

此模組提供 run_agentic_rag() 函式供 main.py 呼叫：
main.py 需要的 route／retrievals／answer 皆由 run_agentic_rag()
從 ReAct 迴圈產生的訊息紀錄（messages）重新整理還原；retrievals 依實際執行順序全域編號，
每一筆都把該次工具呼叫的參數、LLM 判斷原因與檢索結果放在同一筆記錄裡。
"""

# ── 載入套件與環境變數 ──────────────────────────────
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from clients import get_llm
from tools import TOOLS
load_dotenv()


# ── System Prompt：涵蓋工具選擇原則、資料不足時的處理、回答規則 ──────────
SYSTEM_PROMPT = """你是一位圖書館智慧助理，可以自主使用 hyde_query、vector_retrieve、graph_retrieve 三個工具檢索資料，
並整合檢索結果回答讀者問題。

【工具選擇原則】
各工具適合的問題型態已寫在它們的 docstring 中，請依問題性質判斷要呼叫哪一個；
若問題同時包含結構化過濾條件與語意判斷（例如：某分類目前可借閱的書中，最適合新手的是哪一本），
vector_retrieve、graph_retrieve 都可以呼叫，整合兩邊結果後再回答。
hyde_query 是 vector_retrieve 之前的輔助步驟，只在讀者問題較抽象、籠統時才需要呼叫，
呼叫後把回傳的假想文件當作 vector_retrieve 的 query 參數；多數情況下可以直接呼叫 vector_retrieve，不必每次都先跑 hyde_query。

【資料不足時的處理】
若呼叫其中一個工具後查無相關資料，請不要直接回覆查無資料，應嘗試改呼叫另一個工具，
確認兩種檢索都查無結果後，才可以判定資料庫中沒有相關資料。

【回答規則】
請嚴格依據工具回傳的資料回答，不可憑空捏造書籍、作者、分類、價格或借閱資訊。
判斷證據優先順序時請遵守：
1. 作者、借閱狀態、分類、價格、統計數字等精確事實型問題，以知識圖譜查詢結果為主，向量結果只作描述性補充。
2. 模糊語意、情境建議、內容描述、推薦類問題，以向量檢索結果為主，知識圖譜提供結構資訊補充。
3. 若兩邊結果互相矛盾，請優先採用知識圖譜中的結構化事實，並避免使用無法確認的資訊。
若回答包含書籍，請針對每本書明確列出：書名、作者、分類、價格、借閱狀態、推薦理由。
若確認所有已嘗試過的檢索皆無相關資料，請明確回覆「目前資料庫中無相關書籍」。
若 Cypher 執行失敗，請說明目前無法由知識圖譜查得，並僅依向量結果回答（或改用向量檢索）。
請使用繁體中文回答，語氣親切、條理清晰。當你已經蒐集到足夠資料，請直接輸出最終答案，不要再呼叫工具。
你在決定是否呼叫工具時的推理過程，也請使用繁體中文，方便使用者理解你的判斷依據。"""


# ── 執行一次完整的 Agentic RAG 問答流程 ────────────
def run_agentic_rag(query: str) -> dict:
    # 組成初始對話：SYSTEM_PROMPT 已透過 create_agent(system_prompt=...) 固定加在最前面，這裡只需放入讀者的問題
    initial_state = {"messages": [HumanMessage(content=query)]}

    # model 傳未綁定工具的 LLM，create_agent 內部會自行 bind_tools(TOOLS)；
    # 讓 LLM 在 agent_node ⇄ tool_node 之間反覆執行，直到它輸出最終答案；
    # recursion_limit 避免 LLM 判斷失準時陷入無止盡的工具呼叫迴圈
    try:
        agent = create_agent(get_llm(), TOOLS, system_prompt=SYSTEM_PROMPT)
        result = agent.invoke(initial_state, {"recursion_limit": 15})
    except Exception as e:
        # agent_node 呼叫 LLM 本身失敗時（例如逾時、限流），沒有訊息紀錄可還原，
        # 回傳友善錯誤訊息而非讓例外一路往外炸到 main.py，error 保留原始例外供 main.py 顯示除錯資訊
        return {
            "route": "none",
            "retrievals": [],
            "error": str(e),
            "answer": "抱歉，目前系統暫時無法處理您的問題，請稍後再試。",
        }
    messages = result["messages"]

    # retrievals 不是圖的固定欄位，而是從訊息紀錄重新整理還原，每一筆對應一次實際的工具呼叫，
    # 依執行順序全域編號（不分輪次），並把該次呼叫的原因與結果放在同一筆記錄，供 main.py 完整顯示
    retrievals = []
    # pending_calls：tool_call_id → 該次呼叫的 tool 名稱／參數／LLM 判斷原因，等對應的 ToolMessage 回來時取用
    pending_calls = {}

    for message in messages:
        # AIMessage 帶 tool_calls 時，代表這一輪 LLM 決定呼叫某個（或多個）工具
        if isinstance(message, AIMessage) and message.tool_calls:
            # reasoning_content 是 NVIDIA NIM 推理模型呼叫工具前的思考過程（content 欄位在有 tool_calls 時通常是空的）
            reason = (message.additional_kwargs.get("reasoning_content") or "").strip()
            for tool_call in message.tool_calls:
                pending_calls[tool_call["id"]] = {
                    "tool": tool_call["name"],
                    "args": tool_call["args"],
                    "reason": reason,
                }

        # ToolMessage 是工具執行後的回傳結果，用 tool_call_id 對回發起呼叫時記下的參數與原因，
        # 兩者合成一筆完整的 retrieval 記錄，並依實際完成順序給予全域編號
        elif isinstance(message, ToolMessage):
            call_info = pending_calls.get(message.tool_call_id, {})
            retrievals.append(
                {
                    "index": len(retrievals) + 1,
                    "tool": call_info.get("tool", message.name),
                    "args": call_info.get("args", {}),
                    "reason": call_info.get("reason", ""),
                    # artifact 為 None 代表這次 Tool 呼叫本身拋出未預期的例外（例如網路逾時），
                    # 而不是 Tool 正常回傳的結果，需與正常結果分開處理，避免後續存取欄位時崩潰
                    "artifact": message.artifact,
                    "error": message.content if message.artifact is None else None,
                }
            )

    # route 欄位在 ReAct 風格下不再是預先決定的路由結果，改用「這次問答實際呼叫過的工具」呈現
    called_tools = {r["tool"] for r in retrievals}
    route = "+".join(sorted(called_tools)) if called_tools else "none"

    # 最後一則訊息是 LLM 判斷資料已足夠後輸出的最終答案；
    # 推理模型偶爾會把下一步該呼叫的工具寫進推理過程卻沒有真的送出 tool_call，
    # 導致以空白 content 結束，這裡視同未能產出有效答案，改用友善訊息避免顯示空白
    answer = messages[-1].content or "抱歉，系統已完成檢索，但未能整理出完整回答，請重新輸入問題或稍後再試。"

    # 回傳這次問答的路由、依序編號的檢索記錄與最終回答，供 main.py 顯示
    return {
        "route": route,
        "retrievals": retrievals,
        "answer": answer,
    }
