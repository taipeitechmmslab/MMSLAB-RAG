"""
Agentic RAG - Agent 控制迴圈
=================================
agent.py 使用 deepagents 的 create_deep_agent() 組出 ReAct 風格的 Agentic RAG：
LLM 透過 tools 參數拿到 hyde_query()、vector_retrieve()、graph_retrieve() 三個正式的
LangChain Tool，每一輪自己決定要不要呼叫工具、呼叫哪一個、呼叫幾次。

Skill 與 Tool 並存但走不同機制：三個 Tool 直接綁定給 Agent 選用；booklist-markdown-exporter
這個 Skill 存成 skills/ 底下的 SKILL.md，由 deepagents 在啟動時掃描、把 name／description
注入 system prompt，只有在讀者明確要求把書單匯出成 Markdown 檔案時，Agent 才會判斷情境相關、
用內建的 read_file 讀取完整指示、自行套用（不是明確的函式呼叫）；單純問答不會觸發。
書籍格式、排序依據、欄位轉譯、數量統計格式這些每次回答都要套用的規則，不算情境判斷，
直接寫進下方 SYSTEM_PROMPT，不做成 skill。

這裡沒有固定的節點順序：要不要檢索、檢索幾次、何時停止、何時套用 skill，
全部由 LLM 自主決定，判斷依據已寫進 Tool 的 docstring、各 SKILL.md 與下方 SYSTEM_PROMPT。

此模組提供 run_agentic_rag() 函式供 main.py 呼叫，是一個 generator：
用 agent.stream(..., stream_mode="updates") 執行 ReAct 迴圈，每個節點（LLM 決策／
工具執行）跑完就吐出這一步新增的訊息，run_agentic_rag() 逐步配對還原成 retrieval
事件並立即 yield，不必等整個迴圈跑完才一次回傳；main.py 因此能在每一步完成的當下
就印出，不會在等待期間毫無輸出。retrievals 依實際執行順序全域編號，每一筆都把
該次工具呼叫的參數、LLM 判斷原因與檢索結果放在同一筆事件裡。

透過 checkpointer 保留同一次執行過程中的多輪對話歷史：每次呼叫只需帶入這一輪新的
HumanMessage，LangGraph 會依 thread_id 自動接續先前輪次已存進 checkpointer 的訊息，
Agent 因此能理解「他」「那本書」這類指涉前幾輪內容的追問。
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
# 建在模組層級、整個程式執行期間共用同一份，才能跨輪保留對話歷史
# （重新 create_deep_agent() 不影響它，歷史存在 checkpointer 而非 graph 實例裡；
# 純 in-memory，程式重啟即重置，不做跨執行的長期持久化）
CHECKPOINTER = InMemorySaver()

# 只有這幾個是我們自訂的檢索工具，用來從 messages 篩掉 deepagents 內建的 read_file／ls 等工具呼叫
RETRIEVAL_TOOL_NAMES = {tool.name for tool in TOOLS}

# 用來從 read_file 的 file_path 參數判斷這次讀取的是不是某個 skill 的 SKILL.md，抓出 skill 名稱
SKILL_PATH_PATTERN = re.compile(r"/skills/([^/]+)/SKILL\.md$")


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

若回答包含書籍，不論證據來源為何，每本書一律列出以下六個共同欄位：書名、作者、分類、價格、借閱狀態、推薦理由，
順序以此規則為準。此格式適用於一般書籍清單類問題；若問題屬於數量統計類型，改依下方【數量統計類問題的格式】
規則輸出，不套用此格式。

排序依據與推薦理由怎麼寫，依證據來源而定：
1. 向量證據：依讀者問題與相似度分數（score）判斷優先順序，分數越小代表語意越相近，越應排在前面；
   推薦理由從 matched_page_content（命中的段落原文）萃取這本書為何符合讀者「語意或情境」需求的重點，
   用一句精簡文字說明，例如「適合完全沒接觸過的初學者」「內容涵蓋在家自我練習的技巧」。
2. 知識圖譜證據：
   a. 陣列欄位（collect() 產生，例如 authors、coauthor 相關清單）：轉成用「、」連接的中文列舉，
      例如 authors=["Jason", "Mary"] 要寫成「作者：Jason、Mary」；若陣列為空清單，視為該欄位無資料，
      不要自行推測寫成「無作者」等文字。
   b. 統計欄位（count() 產生，例如 book_count）：轉成完整中文敘述而非直接貼數字，
      例如 book_count=12 要寫成「共著作 12 本書」；若查詢是排序取前幾名（例如「哪位作者寫的書最多」），
      敘述時要保留名次或比較關係（例如「著作數最多的是…」）。
   c. 其他關聯性欄位（例如 borrowed_at、category、coauthor）：轉成通順的中文子句嵌入書籍描述中，
      不要原樣附上英文欄位名稱。
   d. 推薦理由聚焦事實面向，說明這本書「為何符合查詢條件」，例如「屬於旅遊分類」「目前可借閱」
      「由指定作者所著」。

【數量統計類問題的格式】
若問題屬於數量統計類型（例如「有幾本」），不套用上述六欄位格式，改用以下規則：
你可以自行在心中排除不符合問題所問類別或條件的書籍（例如問題只問「商業管理類」，就排除其他類別的書籍），
但輸出時只能包含以下兩行標籤與其後的清單，禁止輸出任何其他文字
（不可有前言、排除過程說明、分類依據、開場白或結語）：
\n目前被借出的書籍有 N 本\n
\n沒有被借出的書籍有 M 本\n
此時每本書只需列出書名與借閱狀態，不需要推薦理由。
N、M 請填入實際本數；若某一組本數為 0，仍須保留完整標籤（例如「沒有被借出的書籍有 0 本」），只是後面不要再接清單或其他文字。
此分類僅根據已蒐集到的證據，不代表該類別在資料庫中的完整總數；
結尾請勿再另外總結出一個代表整體類別的本數（例如「該類別書籍共有 X 本」）。

【證據來源與最終格式的對應關係】
1. 只用到向量證據時，整份清單依向量證據的排序依據、推薦理由寫法整理。
2. 只用到知識圖譜證據時，整份清單依知識圖譜證據的欄位轉譯、推薦理由寫法整理。
3. 兩種證據都用到時，先各自依對應規則整理，再合併成單一清單（不要分成兩段呈現），合併時遵守：
   a. 精確事實型問題（作者、借閱狀態、分類、價格、統計數字）以知識圖譜認定的書籍與欄位為主，向量證據只作補充；
      模糊語意、情境建議、內容描述、推薦類問題以向量證據認定的書籍與欄位為主，知識圖譜提供結構補充。
   b. 同一本書兩邊都查到時，凡知識圖譜有提供的精確事實欄位（作者、分類、價格、借閱狀態）一律採用知識圖譜版本；
      知識圖譜沒有涵蓋、只有向量證據提供的欄位，仍應直接採用向量證據的資訊，不要因知識圖譜沒提供就標示為「資料缺乏」；
      推薦理由則綜合兩邊角度，同時反映「為何符合查詢條件」與「語意或情境為何相關」。
   c. 若兩邊證據對同一欄位互相矛盾，優先採用知識圖譜中的結構化事實。
   d. 若其中一種來源這次查詢失敗或查無資料，最終答案改僅依另一種來源的證據整理，
      不需要因此分成兩段呈現，但可視情況簡短說明另一來源這次不可用的原因。

若確認所有已嘗試過的檢索皆無相關資料，請明確回覆「目前資料庫中無相關書籍」。

請使用繁體中文回答，語氣親切、條理清晰。"""


# ── 執行一次完整的 Agentic RAG 問答流程（generator，逐步 yield 決策過程） ────────
def run_agentic_rag(query: str) -> Iterator[dict]:
    # thread_id 讓 checkpointer 認得這是同一場對話，才能接續先前輪次存下的訊息歷史
    config = {"configurable": {"thread_id": THREAD_ID}, "recursion_limit": 40}
    # backend 指向 Lab5 目錄本身，deepagents 才能從真實磁碟讀到 skills/ 底下的 SKILL.md
    # （預設是記憶體內的 StateBackend，讀不到本機檔案）
    # virtual_mode=True 讓 Agent 用 /skills/... 這種虛擬路徑存取，對應到 root_dir 底下的實際檔案，
    # 避免 Windows 磁碟機代號路徑（如 D:/...）造成的相容性問題
    # skills=["skills/"] 讓 deepagents 掃描這個資料夾，把每個 SKILL.md 的 name／description 注入 system prompt
    agent = create_deep_agent(
        model=get_llm(),
        tools=TOOLS,
        system_prompt=SYSTEM_PROMPT,
        backend=FilesystemBackend(root_dir=BASE_DIR, virtual_mode=True),
        skills=["skills/"],
        checkpointer=CHECKPOINTER,
    )

    # AIMessage 決定呼叫工具時還沒有結果，要等對應的 ToolMessage 出現才能配對合併成完整記錄，
    # pending_calls 就是用 tool_call_id 當 key，記住 tool 名稱／參數／原因
    pending_calls = {}
    retrieval_count = 0
    # 每次遇到「沒有 tool_calls 的 AIMessage」就更新，串流結束時的版本即為最終答案
    last_answer = None

    try:
        # stream_mode="updates" 讓每個節點（LLM 決策／工具執行）跑完就吐出這一步新增的訊息，
        # 天生就是本輪新增內容，不必再靠切片比對「這一輪新增訊息」
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

                        # read_file 讀取的是某個 SKILL.md 時，視為一次 skill 使用紀錄，跟 tool 呼叫共用同一份
                        # retrievals 編號，才能反映 Agent 實際交錯呼叫 tool／skill 的順序
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
                                    # read_file 回傳的 SKILL.md 全文（cat -n 格式），讓使用者看到 Agent 實際讀到的指示內容
                                    "content": message.content,
                                }
                            # 不是讀 SKILL.md（例如讀到其他檔案）就不記錄，繼續處理下一則訊息
                            continue

                        # write_file 是 booklist-markdown-exporter 這個 skill 實際落地成檔案的那一步，
                        # 跟 tool／skill 呼叫共用同一份 retrievals 編號，才能讓使用者看到「套用 skill 規則
                        # → 實際寫檔」的完整過程
                        if tool_name == "write_file":
                            retrieval_count += 1
                            yield {
                                "type": "retrieval",
                                "index": retrieval_count,
                                "kind": "file_write",
                                "file_path": call_info.get("args", {}).get("file_path", ""),
                                "content": call_info.get("args", {}).get("content", ""),
                                "reason": call_info.get("reason", ""),
                                # write_file 不是我們自訂的 content_and_artifact 工具，沒有 artifact 可判斷成敗，
                                # 改看 ToolMessage 的 status 欄位
                                "error": message.content if message.status == "error" else None,
                            }
                            continue

                        # deepagents 其他內建工具（ls／write_todos／task 等，write_file 已在上面另外處理）不是
                        # 我們自訂的檢索工具，且沒有 artifact 可用，混進 retrievals 會被誤判成呼叫失敗，故略過
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
        # LLM 呼叫本身失敗時（例如逾時、限流），先前已 yield 的 retrieval 不受影響，
        # 額外吐出 error 事件後結束，不再產生 answer
        yield {"type": "error", "error": str(e)}
        return

    # 推理模型偶爾只把下一步寫進推理過程卻沒真的送出內容，導致 last_answer 為 None，視同無效答案
    yield {
        "type": "answer",
        "answer": last_answer or "抱歉，這次 LLM 未能產生完整回答內容（並非系統故障），請重新輸入問題或稍後再試。",
    }
