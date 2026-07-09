"""
Agentic RAG - Agent 控制迴圈
=================================
agent.py 使用 deepagents 的 create_deep_agent() 組出 ReAct 風格的 Agentic RAG：
  - Tool：hyde_query()、vector_retrieve()、graph_retrieve() 三個 LangChain Tool，
    綁定給 Agent，每一輪自己決定要不要呼叫、呼叫哪一個、呼叫幾次
  - Skill：booklist-markdown-exporter 存成 skills/ 底下的 SKILL.md，由 deepagents
    掃描後注入 system prompt，讀者要求匯出書單時 Agent 才會用 read_file 讀取並自行套用
  - 是否檢索、檢索幾次、何時套用 skill，全部由 LLM 依 Tool docstring／SKILL.md／
    SYSTEM_PROMPT 自主判斷，沒有固定的節點順序

此模組提供 run_agentic_rag() 函式供 main.py 呼叫，是一個 generator：
用 agent.stream(..., stream_mode="updates") 執行 ReAct 迴圈，每個節點跑完就逐步
yield 出一筆 retrieval 事件，main.py 不必等整個流程跑完就能即時印出。

透過 checkpointer 保留多輪對話歷史，讓 Agent 能理解「他」「那本書」這類追問。
"""

# ── 載入套件與環境變數 ──────────────────────────────
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

# BASE_DIR 是 Lab5 目錄本身，FilesystemBackend 才能從真實磁碟讀到 skills/ 底下的 SKILL.md
BASE_DIR = Path(__file__).parent

# 這個 process 只服務單一互動式對話，用固定值識別同一個 thread
THREAD_ID = "cli-session"
# 建在模組層級、跨輪共用同一份，才能保留多輪對話歷史（純 in-memory，重啟即重置）
CHECKPOINTER = InMemorySaver()

# 只有這幾個是我們自訂的檢索工具，用來從 messages 篩掉 deepagents 內建的 read_file／ls 等工具呼叫
RETRIEVAL_TOOL_NAMES = {tool.name for tool in TOOLS}

# 用來從 read_file 的 file_path 參數判斷這次讀取的是不是某個 skill 的 SKILL.md，抓出 skill 名稱
SKILL_PATH_PATTERN = re.compile(r"/skills/([^/]+)/SKILL\.md$")

# 推理模型偶爾把「呼叫某工具」的意圖寫成假的函式呼叫語法留在回答文字裡，用「工具名稱＋左括號」偵測這種無效答案
FAKE_TOOL_CALL_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(name) for name in RETRIEVAL_TOOL_NAMES) + r")\s*\("
)


# ── System Prompt：工具選擇原則、資料不足時的處理、何時使用 skill ──────────
SYSTEM_PROMPT = """你是一位圖書館智慧助理，可以自主使用 hyde_query、vector_retrieve、graph_retrieve 三個工具檢索資料，
並在讀者明確要求把書單匯出成檔案時，使用 booklist-markdown-exporter 這個 skill 決定檔名與寫檔格式。

【工具選擇原則】
各工具適合的問題型態已寫在它們的 docstring 中，請依問題性質判斷要呼叫哪一個；
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

【生成最終回答時的規則】
請嚴格依據已蒐集到的證據回答，不可憑空捏造書籍、作者、分類、價格或借閱資訊。
若回答包含書籍（非數量統計類問題，見下方另一格式），每本書獨立條列、逐一標明欄位名稱呈現：
書名、作者、分類、價格、借閱狀態、推薦理由，不要把多個欄位擠在同一行、省略欄位名稱。
判斷優先順序與寫法時請遵守：
1. 精確事實型欄位（作者、分類、價格、借閱狀態、統計數字）以知識圖譜結果為主：陣列欄位（如 collect() 產生的
   authors）轉成頓號列舉，空陣列代表無資料、勿臆測填寫；統計欄位（如 count()）轉成完整中文敘述，若是排序取前幾名
   的查詢需保留名次或比較關係；其他關聯欄位轉成通順中文子句，不要原樣附上英文欄位名稱。
2. 模糊語意、情境建議、推薦類問題以向量結果為主，依相似度分數（越小越相關）排序，推薦理由從命中段落萃取這本書
   符合語意或情境需求的重點。
3. 兩邊都查到同一本書時，精確事實欄位一律採知識圖譜版本，向量獨有的欄位仍照常採用，不因知識圖譜沒提供而省略；
   兩邊矛盾時優先採知識圖譜；若其中一種來源這次查無資料，僅依另一來源整理即可。

【數量統計類問題的格式】
若問題屬於數量統計類型（例如「有幾本」），不套用上述欄位格式，改用以下規則：
你可以自行在心中排除不符合問題所問類別或條件的書籍，但輸出時只能包含以下兩行標籤與其後的清單，
禁止輸出任何其他文字（不可有前言、排除過程說明、開場白或結語）：
\n目前被借出的書籍有 N 本\n
\n沒有被借出的書籍有 M 本\n
此時每本書只需列出書名與借閱狀態，不需要推薦理由。
N、M 請填入實際本數；若某一組本數為 0，仍須保留完整標籤，只是後面不要再接清單或其他文字。
此分類僅根據已蒐集到的證據，不代表該類別在資料庫中的完整總數，結尾請勿再另外總結出整體本數。

若確認所有已嘗試過的檢索皆無相關資料，請明確回覆「目前資料庫中無相關書籍」。

請使用繁體中文回答，語氣親切、條理清晰。"""


# ── 執行一次完整的 Agentic RAG 問答流程（generator，逐步 yield 決策過程） ────────
def run_agentic_rag(query: str) -> Iterator[dict]:
    # thread_id 讓 checkpointer 接續同一場對話的歷史訊息
    config = {"configurable": {"thread_id": THREAD_ID}, "recursion_limit": 40}
    # backend 指向 Lab5 目錄，讓 deepagents 讀到真實磁碟上的 skills/SKILL.md；
    # virtual_mode 用 /skills/... 虛擬路徑存取，避免 Windows 磁碟機代號路徑的相容性問題
    agent = create_deep_agent(
        model=get_llm(),
        tools=TOOLS,
        system_prompt=SYSTEM_PROMPT,
        backend=FilesystemBackend(root_dir=BASE_DIR, virtual_mode=True),
        skills=["skills/"],
        checkpointer=CHECKPOINTER,
    )

    # 用 tool_call_id 記住尚未配對到 ToolMessage 的工具呼叫（名稱／參數／原因）
    pending_calls = {}
    retrieval_count = 0
    # 記錄目前最新的 AIMessage 文字內容，串流結束時的版本即為最終答案
    last_answer = None

    try:
        # stream_mode="updates" 讓每個節點跑完就吐出這一步新增的訊息
        for chunk in agent.stream({"messages": [HumanMessage(content=query)]}, config, stream_mode="updates"):
            for node_update in chunk.values():
                if not isinstance(node_update, dict) or "messages" not in node_update:
                    continue

                for message in node_update["messages"]:
                    # AIMessage 帶 tool_calls 時，代表這一輪 LLM 決定呼叫某個（或多個）工具
                    if isinstance(message, AIMessage) and message.tool_calls:
                        # 從 content_blocks 篩出 type 為 reasoning 的區塊
                        reasoning_blocks = [b for b in message.content_blocks if b.get("type") == "reasoning"]
                        # 把各段 reasoning 文字接起來，即為 LLM 呼叫工具前的判斷原因
                        reason = " ".join(b.get("reasoning", "") for b in reasoning_blocks).strip()
                        # 一輪可能同時呼叫多個工具，逐一處理每個 tool_call
                        for tool_call in message.tool_calls:
                            pending_calls[tool_call["id"]] = {
                                "tool": tool_call["name"],
                                "args": tool_call["args"],
                                "reason": reason,
                            }
                        continue

                    # 沒有 tool_calls 的 AIMessage 代表這輪對話目前的最終回答，記下來供串流結束時 yield
                    if isinstance(message, AIMessage):
                        if message.content:
                            last_answer = message.content
                        continue

                    # ToolMessage 是工具執行後的回傳結果
                    if isinstance(message, ToolMessage):
                        # 用 tool_call_id 取回發起呼叫時記下的名稱、參數、原因
                        call_info = pending_calls.get(message.tool_call_id, {})
                        tool_name = call_info.get("tool", message.name)

                        # read_file 讀到 SKILL.md 時視為一次 skill 使用紀錄，跟 tool 呼叫共用 retrievals 編號
                        if tool_name == "read_file":
                            skill_match = SKILL_PATH_PATTERN.search(call_info.get("args", {}).get("file_path", ""))
                            if skill_match:
                                retrieval_count += 1
                                yield {
                                    "type": "retrieval",
                                    "index": retrieval_count,
                                    "kind": "skill",
                                    "skill": skill_match.group(1),
                                    "reason": call_info.get("reason", ""),
                                    # SKILL.md 全文（cat -n 格式）
                                    "content": message.content,
                                }
                            # 不是讀 SKILL.md 就略過
                            continue

                        # write_file 是 skill 實際落地寫檔的一步，同樣併入 retrievals 編號
                        if tool_name == "write_file":
                            retrieval_count += 1
                            yield {
                                "type": "retrieval",
                                "index": retrieval_count,
                                "kind": "file_write",
                                "file_path": call_info.get("args", {}).get("file_path", ""),
                                "content": call_info.get("args", {}).get("content", ""),
                                "reason": call_info.get("reason", ""),
                                # 沒有 artifact 可判斷成敗，改看 ToolMessage 的 status 欄位
                                "error": message.content if message.status == "error" else None,
                            }
                            continue

                        # 其他 deepagents 內建工具（ls／write_todos 等）沒有 artifact 可用，略過
                        if tool_name not in RETRIEVAL_TOOL_NAMES:
                            continue

                        # 合併發起呼叫的資訊與這次的執行結果，組成一筆完整的 retrieval 記錄並立即 yield
                        retrieval_count += 1
                        yield {
                            "type": "retrieval",
                            "index": retrieval_count,
                            "kind": "tool",
                            "tool": tool_name,
                            "args": call_info.get("args", {}),
                            "reason": call_info.get("reason", ""),
                            # artifact 為 None 代表 Tool 呼叫本身拋出例外，需與正常結果分開處理
                            "artifact": message.artifact,
                            # artifact 為 None 時，改用 content（錯誤訊息）填入 error 欄位
                            "error": message.content if message.artifact is None else None,
                        }
    except Exception as e:
        # LLM 呼叫本身失敗時（例如逾時、限流），吐出 error 事件後結束，不再產生 answer
        yield {"type": "error", "error": str(e)}
        return

    # 回答文字裡出現假的工具呼叫語法，視同無效答案
    if last_answer and FAKE_TOOL_CALL_PATTERN.search(last_answer):
        last_answer = None

    # answer 為 None 代表這輪沒有拿到有效答案，訊息文字交給 main.py 決定
    yield {"type": "answer", "answer": last_answer}
