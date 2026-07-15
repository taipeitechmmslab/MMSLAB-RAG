"""
Agentic RAG - Agent 動態檢索流程
=================================
agent.py 負責建立 deepagents Agent，讓 Agent 依讀者問題自主選擇檢索工具，
並將 Agent 串流中的工具呼叫、工具結果與最終回答整理成事件，交給 main.py 顯示。

程式流程：
  1. 建立包含 LLM、檢索工具、skill 與對話記憶的 Agent。
  2. 將讀者問題送入 Agent，逐步取得各個節點產生的訊息。
  3. 將訊息整理成「檢索過程」、「錯誤」或「最終回答」事件。
"""

import re
from pathlib import Path
from typing import Iterator

from dotenv import load_dotenv
from deepagents import create_deep_agent
from deepagents.backends.filesystem import FilesystemBackend
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver

from clients import get_llm
from tools import TOOLS

load_dotenv()


# ── Agent 設定 ───────────────────────────────────────────────

# 這個物件必須在多次問答間共用，才能保留對話歷史。
CHECKPOINTER = InMemorySaver()

# 從 tools/ 匯入的三個工具名稱；只有這些工具需要將檢索結果顯示給使用者。
RAG_TOOL_NAMES = set()
# 逐一取出每個工具的名稱，加入 set 後可快速判斷某個名稱是否屬於 RAG 工具。
for rag_tool in TOOLS:
    RAG_TOOL_NAMES.add(rag_tool.name)

# deepagents 以虛擬路徑讀取 skill 說明檔，例如 /skills/booklist-markdown-exporter/SKILL.md。
# 這個規則用來從路徑中取出 skill 名稱。
SKILL_FILE_PATTERN = re.compile(r"/skills/([^/]+)/SKILL\.md$")

# 少數模型會把「我要呼叫工具」誤輸出成一般文字，例如 vector_retrieve(...)。
# 這個規則用來辨識這種不是正式回答的文字。
escaped_tool_names = []
# 逐一跳脫工具名稱中的正規表示式特殊字元，再組成「名稱 A 或名稱 B」的規則。
for tool_name in RAG_TOOL_NAMES:
    escaped_tool_names.append(re.escape(tool_name))

FAKE_TOOL_CALL_PATTERN = re.compile(
    r"\b(" + "|".join(escaped_tool_names) + r")\s*\("
)


# ── Agent 指示 ───────────────────────────────────────────────

SYSTEM_PROMPT = """你是一位圖書館智慧助理，可以自主使用 hyde_query、vector_retrieve、graph_retrieve 三個工具檢索資料，
並在讀者明確要求把書單匯出成檔案時，使用 booklist-markdown-exporter 這個 skill 決定檔名與寫檔格式。

【工具選擇原則】
各工具適合的問題型態已寫在它們的 docstring 中，請依問題性質判斷要呼叫哪一個；
若問題只包含結構化過濾條件（分類、借閱狀態、作者、借閱者等），沒有語意層面的判斷或情境推薦需求，
應優先呼叫 graph_retrieve，只有在圖譜查詢結果不足以回答問題時，才輔以 vector_retrieve 補充。
若問題同時包含結構化過濾條件與語意判斷（例如：某分類目前可借閱的書中，最適合新手的是哪一本），
vector_retrieve、graph_retrieve 都可以呼叫，整合兩邊結果後再回答。
hyde_query 是 vector_retrieve 之前的輔助步驟，只在讀者問題較抽象、籠統時才需要呼叫，
呼叫後把回傳的假想文件當作 vector_retrieve 的 query 參數使用。

【資料不足時的處理】
若呼叫其中一個工具後查無相關資料，請不要直接判定查無資料，應嘗試改呼叫另一個工具，
確認兩種檢索都查無結果後，才可以判定資料庫中沒有相關資料。

【使用 Skill 的時機】
只有在讀者明確要求把書單存成檔案、匯出、整理成 Markdown 文件時，才使用 booklist-markdown-exporter
決定檔名與寫檔內容格式；單純的問答不套用，仍依下方規則直接在對話中回覆。
這個 skill 不是可以直接呼叫的工具，使用方式是呼叫 read_file 讀取
/skills/booklist-markdown-exporter/SKILL.md，自行依內容套用。

【最終回答格式】
請嚴格依據已蒐集到的證據回答，不可憑空捏造書籍、作者、分類、價格或借閱資訊。
精確事實（作者、分類、價格、借閱狀態、統計數字）以知識圖譜結果為主；情境建議與推薦理由以向量檢索的命中段落為主。兩邊衝突時，以知識圖譜為準。
請先判斷問題類型，且只套用一種回答格式。

【一般書籍推薦或清單】
先用一至兩句直接回答問題。接著列出所有已蒐集且符合條件的書籍，每本以阿拉伯數字編號，並各自換行列出：
書名、作者、分類、價格、借閱狀態、推薦理由。
每個欄位都必須獨占一行，絕對不可在同一行以全形空格、頓號或其他方式串接兩個欄位。嚴格使用下列純文字版型（方括號內替換為實際內容）：
1. 書名：[書名]
   作者：[作者]
   分類：[分類]
   價格：[價格] 元
   借閱狀態：[借閱狀態]
   推薦理由：[推薦理由]
編號只放在「書名」那一行；每本書的六個欄位必須完整依此順序輸出，且每本書之間保留一個空行。
推薦理由必須根據命中段落，以白話且具體的方式說明此書能幫助讀者處理哪個問題或情境；不得只堆砌框架、矩陣等術語。每本書之間保留一個空行。

【借閱數量統計】
只可輸出下列兩組內容，不可加入前言、說明或結語：
目前被借出的書籍有 N 本
接著以阿拉伯數字編號列出該組每本書，並各自換行寫出「書名」與「借閱狀態」。
沒有被借出的書籍有 M 本
接著以阿拉伯數字編號列出該組每本書，並各自換行寫出「書名」與「借閱狀態」。
每組為 0 本時，只輸出該組本數標籤，不列書籍。本數僅依已蒐集的證據計算，不代表資料庫完整總數。

【其他統計、單一事實或查無資料】
其他統計或單一事實直接用短句回答。兩種檢索都無相關結果時，回答「目前資料庫中無相關書籍」。

請使用繁體中文，語氣親切、條理清晰。終端對話回答直接從內容開始，不得輸出「AI 回答：」或其他固定前綴。
只能使用純文字、換行與阿拉伯數字編號；不得使用 Markdown、粗體、斜體、標題、程式碼區塊、表格、分隔線或項目符號。
讀者要求匯出檔案時，檔案內容仍依 booklist-markdown-exporter skill 的 Markdown 規則處理；此例外不適用於終端對話回答。"""


# ── 呼叫 Agent ─────────────────────────────────────────────────

def stream_agent(query: str):
    # 設定本次 Agent 執行參數：
    #   - thread_id 固定為 cli-session，使後續提問能保留同一段對話歷史。
    #   - recursion_limit 限制 Agent 最多可反覆推理與呼叫工具的次數。
    config = {
        "configurable": {"thread_id": "cli-session"},
        "recursion_limit": 40,
    }

    # 建立 deepagents Agent：
    #   - model：使用 clients.py 設定好的 LLM。
    #   - tools：提供 HyDE、向量檢索與知識圖譜檢索工具。
    #   - system_prompt：告訴 Agent 如何選工具與回答讀者。
    #   - backend / skills：讓 Agent 能讀取 skills/ 下的 SKILL.md。
    #   - checkpointer：保存同一段對話的歷史訊息。
    agent = create_deep_agent(
        model=get_llm(),
        tools=TOOLS,
        system_prompt=SYSTEM_PROMPT,
        backend=FilesystemBackend(root_dir=Path(__file__).parent, virtual_mode=True),
        skills=["skills/"],
        checkpointer=CHECKPOINTER,
    )

    # 建立 Agent 需要的輸入資料；messages list 內放入使用者這次輸入的問題。
    input_data = {"messages": [HumanMessage(content=query)]}

    # 回傳 Agent 的原始串流資料，不在這裡處理任何教學 Log。
    # stream_mode="updates" 會在每個節點執行後回傳該節點更新的資料。
    return agent.stream(input_data, config, stream_mode="updates")


# ── 從 Agent 串流資料中取出訊息 ──────────────────────────────────────────────

def _iter_messages(agent_updates) -> Iterator[AIMessage | ToolMessage]:
    # 逐一讀取 Agent 每個節點執行後的更新資料。
    for update in agent_updates:
        # update 的 key 是節點名稱，value 是該節點這一步更新的資料。
        for node_result in update.values():
            # 有些節點只更新其他 state；只處理包含 messages 欄位的節點結果。
            if isinstance(node_result, dict) and "messages" in node_result:
                # messages 可能包含 AIMessage 或 ToolMessage，逐筆交給呼叫端處理。
                for message in node_result["messages"]:
                    yield message


# ── 將 Agent 訊息整理成教學 Log ──────────────────────────────────────────────

def _iter_teaching_events(messages) -> Iterator[dict]:
    # 暫存尚未收到工具結果的工具呼叫，key 為 tool_call_id。
    pending_calls = {}
    # 記錄要顯示給使用者的活動編號，從 1 開始。
    activity_index = 0
    # 保存最後一則沒有 tool_calls 的 AIMessage，串流結束後即為最終回答。
    final_answer = None

    # 逐筆處理 Agent 執行過程中的 AIMessage 與 ToolMessage。
    for message in messages:
        # AIMessage 有 tool_calls，代表 Agent 決定呼叫工具。
        if isinstance(message, AIMessage) and message.tool_calls:
            # 收集 Agent 呼叫工具前的 reasoning，作為畫面上的「判斷原因」。
            reasoning_texts = []
            for block in message.content_blocks:
                if block.get("type") == "reasoning":
                    reasoning_texts.append(block.get("reasoning", ""))
            reason = " ".join(reasoning_texts).strip()

            # AIMessage 先提供工具名稱與參數，稍後 ToolMessage 才會提供執行結果。
            # 先用 tool_call_id 記住這些資料，收到結果時才能組成完整 Log。
            for tool_call in message.tool_calls:
                pending_calls[tool_call["id"]] = {
                    "tool": tool_call["name"],
                    "args": tool_call["args"],
                    "reason": reason,
                }
            continue

        # 沒有 tool_calls 的 AIMessage 就是目前的最終回答。
        if isinstance(message, AIMessage):
            if message.content:
                final_answer = message.content
            continue

        # ToolMessage 是工具執行完成後的回傳資料。
        if isinstance(message, ToolMessage):
            # 依 tool_call_id 取回發起工具呼叫時記住的名稱、參數與原因。
            call_info = pending_calls.get(message.tool_call_id, {})
            tool_name = call_info.get("tool", message.name)
            args = call_info.get("args", {})
            reason = call_info.get("reason", "")

            # 先設定為 None；只有需要顯示的工具才會建立 Log 事件。
            event = None

            # read_file 只有在讀取 SKILL.md 時，才需要顯示使用了哪個 skill。
            if tool_name == "read_file":
                skill_match = SKILL_FILE_PATTERN.search(args.get("file_path", ""))
                if skill_match is not None:
                    event = {
                        "kind": "skill",
                        "reason": reason,
                        "skill": skill_match.group(1),
                        "content": message.content,
                    }

            # write_file 用來輸出書單，因此顯示寫入路徑、內容與錯誤訊息。
            elif tool_name == "write_file":
                event = {
                    "kind": "file_write",
                    "reason": reason,
                    "file_path": args.get("file_path", ""),
                    "content": args.get("content", ""),
                    "error": message.content if message.status == "error" else None,
                }

            # 專案定義的 RAG 工具需要顯示名稱、參數與檢索結果。
            elif tool_name in RAG_TOOL_NAMES:
                event = {
                    "kind": "tool",
                    "reason": reason,
                    "tool": tool_name,
                    "args": args,
                    "artifact": message.artifact,
                    "error": message.content if message.artifact is None else None,
                }

            # deepagents 的其他內建工具不需要顯示，因此略過。
            if event is None:
                continue

            # 每產生一個活動事件就加上編號，再交給 main.py 顯示。
            activity_index += 1
            yield {"type": "retrieval", "index": activity_index, **event}

    # 模型偶爾會把假的工具呼叫語法當成一般回答，這種回答視為無效。
    if final_answer and FAKE_TOOL_CALL_PATTERN.search(final_answer):
        final_answer = None

    # 串流正常結束後，回傳最終回答；若沒有有效回答則 answer 為 None。
    yield {"type": "answer", "answer": final_answer}


# ── 對外入口 ─────────────────────────────────────────────────

def run_agentic_rag(query: str) -> Iterator[dict]:
    try:
        # 核心 Agent 只負責執行問題並回傳原始串流資料。
        agent_updates = stream_agent(query)

        # 將原始串流資料中的訊息取出。
        messages = _iter_messages(agent_updates)

        # 教學 Log 是額外的顯示功能，不會放進 Agent 的執行函式中。
        teaching_events = _iter_teaching_events(messages)
        for event in teaching_events:
            yield event

    # Agent 執行期間若發生連線、模型或工具錯誤，轉成 error 事件交給 main.py 顯示。
    except Exception as error:
        yield {"type": "error", "error": str(error)}
