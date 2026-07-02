"""
Agentic RAG - Agent 控制迴圈（LangGraph 版本，自由式 ReAct）
=================================
agent.py 使用 LangGraph 的 StateGraph 實作自由式（ReAct 風格）的 Agentic RAG：
LLM 透過 bind_tools() 拿到 vector_retrieve()、graph_retrieve() 兩個正式的
LangChain Tool，每一輪自己決定要不要呼叫工具、呼叫哪一個、呼叫幾次，
直到它認為已經蒐集到足夠資料，才輸出不帶 tool_calls 的最終答案。

流程走向（標準 ReAct 迴圈）：
  START → agent_node ⇄ tool_node → agent_node → ... → END
    agent_node：呼叫 LLM，若回傳 tool_calls 就走向 tool_node，否則走向 END
    tool_node ：實際執行 LLM 選擇的工具，把結果以 ToolMessage 加回對話，再交還 agent_node

這裡沒有固定的節點順序：要不要檢索、檢索幾次、何時停止全部由 LLM 自主決定，
路由與品質判斷的知識已寫進 Tool 的 docstring 與下方 SYSTEM_PROMPT，供 LLM 參考。

此模組提供 run_agentic_rag() 函式供 main.py 呼叫，回傳格式與呼叫方式維持不變：
main.py 需要的 route／vector_docs／graph_result／steps／answer 皆由 run_agentic_rag()
從 ReAct 迴圈產生的訊息紀錄（messages）重新整理還原。
"""

# ── 載入套件與環境變數 ──────────────────────────────
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode
from clients import get_llm
from tools import TOOLS
load_dotenv()


# ── System Prompt：涵蓋工具選擇原則、資料不足時的處理、回答規則 ──────────
SYSTEM_PROMPT = """你是一位圖書館智慧助理，可以自主使用 vector_retrieve、graph_retrieve 兩個工具檢索資料，
並整合檢索結果回答讀者問題。

【工具選擇原則】
兩個工具各自適合的問題型態已寫在它們的 docstring 中，請依問題性質判斷要呼叫哪一個；
若問題同時包含結構化過濾條件與語意判斷（例如：某分類目前可借閱的書中，最適合新手的是哪一本），
兩個工具都可以呼叫，整合兩邊結果後再回答。

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
請使用繁體中文回答，語氣親切、條理清晰。當你已經蒐集到足夠資料，請直接輸出最終答案，不要再呼叫工具。"""


# ── agent_node：呼叫 LLM，由 LLM 自主決定是否呼叫工具、呼叫哪一個 ────────────
def agent_node(state: MessagesState) -> dict:
    # 初始化 NVIDIA NIM LLM，並綁定工具清單，讓 LLM 知道有哪些工具可以呼叫
    llm = get_llm().bind_tools(TOOLS)

    # 將目前累積的對話（含歷次工具呼叫與回傳結果）交給 LLM，由它決定下一步
    response = llm.invoke(state["messages"])
    # 回傳這個節點對 State 的更新：把 LLM 的回應（可能帶 tool_calls）加入對話紀錄
    return {"messages": [response]}


# ── 判斷 agent_node 執行後，該走向 tool_node 還是結束 ────────────
def route_after_agent(state: MessagesState) -> str:
    # 取得 LLM 最新一次的回應
    last_message = state["messages"][-1]
    # 若 LLM 這次回應帶有 tool_calls，代表它決定要呼叫工具，走向 tool_node
    if getattr(last_message, "tool_calls", None):
        return "tool_node"
    # 否則代表 LLM 認為資料已足夠，這則回應就是最終答案，流程結束
    return END


# ── 組裝 StateGraph：agent_node 與 tool_node 互相往返，形成 ReAct 迴圈 ──────────
def build_agent_graph():
    # 建立 StateGraph，使用 LangGraph 內建的 MessagesState 作為狀態結構
    # （state["messages"] 是一份會自動累加的訊息清單，新訊息會附加在後面而非覆蓋）
    graph = StateGraph(MessagesState)

    # 將 agent_node 註冊為節點；tool_node 使用 LangGraph 內建的 ToolNode，
    # 會自動依 LLM 回應中的 tool_calls 執行對應工具，並把結果包成 ToolMessage 加回對話
    graph.add_node("agent_node", agent_node)
    graph.add_node("tool_node", ToolNode(TOOLS))

    # 固定流程：進入圖後先呼叫一次 agent_node
    graph.add_edge(START, "agent_node")
    # agent_node 之後依 route_after_agent() 的判斷，走向 tool_node 或結束
    graph.add_conditional_edges("agent_node", route_after_agent)
    # 工具執行完後，把結果交還 agent_node，讓 LLM 決定下一步（再檢索或直接回答）
    graph.add_edge("tool_node", "agent_node")

    # 編譯成可以直接呼叫 invoke() 的圖
    return graph.compile()

# ── 執行一次完整的 Agentic RAG 問答流程 ────────────
def run_agentic_rag(query: str) -> dict:
    # 組成初始對話：System Prompt 設定角色與規則，接著放入讀者的問題
    initial_state = {
        "messages": [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=query),
        ]
    }

    # 呼叫編譯好的圖，讓 LLM 在 agent_node ⇄ tool_node 之間反覆執行，直到它輸出最終答案
    # recursion_limit 避免 LLM 判斷失準時陷入無止盡的工具呼叫迴圈
    final_state = build_agent_graph().invoke(initial_state, {"recursion_limit": 15})
    messages = final_state["messages"]

    # steps／vector_docs／graph_result 皆不是圖的固定欄位，而是從訊息紀錄重新整理還原，
    # 供 main.py 顯示 Agent 實際的思考與檢索過程
    steps = []
    vector_docs = None
    graph_result = None

    for message in messages:
        # AIMessage 帶 tool_calls 時，代表這一輪 LLM 決定呼叫某個工具
        if isinstance(message, AIMessage) and message.tool_calls:
            for tool_call in message.tool_calls:
                steps.append(f"Agent 決定呼叫工具：{tool_call['name']}（參數：{tool_call['args']}）")

        # ToolMessage 是工具執行後的回傳結果，artifact 存放未格式化的原始資料
        elif isinstance(message, ToolMessage):
            if message.name == "vector_retrieve":
                vector_docs = message.artifact
                steps.append(f"向量檢索完成，取得 {len(vector_docs)} 筆結果")
            elif message.name == "graph_retrieve":
                graph_result = message.artifact
                note = "知識圖譜檢索完成"
                # retries > 0 代表過程中曾發生 Cypher 生成或執行失敗並重新生成
                if graph_result["retries"] > 0:
                    note += f"（Cypher 生成或執行失敗，已重新生成並重試 {graph_result['retries']} 次）"
                steps.append(note)

    # route 欄位在自由式設計下不再是預先決定的路由結果，改用「這次問答實際呼叫過的工具」呈現
    called_tools = []
    if vector_docs is not None:
        called_tools.append("vector")
    if graph_result is not None:
        called_tools.append("graph")
    route = "+".join(called_tools) if called_tools else "none"

    # 最後一則訊息是 LLM 判斷資料已足夠後輸出的最終答案
    answer = messages[-1].content

    # 回傳這次問答的路由、檢索結果、決策過程與最終回答，供 main.py 顯示
    return {
        "route": route,
        "vector_docs": vector_docs,
        "graph_result": graph_result,
        "steps": steps,
        "answer": answer,
    }
