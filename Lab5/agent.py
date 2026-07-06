"""
Agentic RAG - Agent 控制迴圈
=================================
agent.py 使用 deepagents 的 create_deep_agent() 組出 ReAct 風格的 Agentic RAG：
LLM 透過 tools 參數拿到 hyde_query()、vector_retrieve()、graph_retrieve() 三個正式的
LangChain Tool，每一輪自己決定要不要呼叫工具、呼叫哪一個、呼叫幾次。

Skill 與 Tool 並存但走不同機制：三個 Tool 直接綁定給 Agent 選用；三個 Skill
（vector-result-organizer／graph-result-organizer／final-answer-synthesizer）
存成 skills/ 底下的 SKILL.md，由 deepagents 在啟動時掃描、把 name／description
注入 system prompt，Agent 判斷情境相關時才會用內建的 read_file 讀取完整指示、
自行套用（不是明確的函式呼叫）。

這裡沒有固定的節點順序：要不要檢索、檢索幾次、何時停止、何時套用哪個 skill，
全部由 LLM 自主決定，判斷依據已寫進 Tool 的 docstring、各 SKILL.md 與下方 SYSTEM_PROMPT。

此模組提供 run_agentic_rag() 函式供 main.py 呼叫：
main.py 需要的 route／retrievals／answer 皆由 run_agentic_rag()
從 ReAct 迴圈產生的訊息紀錄（messages）重新整理還原；retrievals 依實際執行順序全域編號，
每一筆都把該次工具呼叫的參數、LLM 判斷原因與檢索結果放在同一筆記錄裡。
"""

# ── 載入套件與環境變數 ──────────────────────────────
import re
from pathlib import Path
from dotenv import load_dotenv
from deepagents import create_deep_agent
from deepagents.backends.filesystem import FilesystemBackend
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from clients import get_llm
from tools import TOOLS
load_dotenv()

# BASE_DIR 是 Lab5 目錄本身，FilesystemBackend 才能從真實磁碟讀到 skills/ 底下的 SKILL.md
BASE_DIR = Path(__file__).parent

# 只有這幾個是我們自訂的檢索工具，用來從 messages 篩掉 deepagents 內建的 read_file／ls 等工具呼叫
RETRIEVAL_TOOL_NAMES = {tool.name for tool in TOOLS}

# 用來從 read_file 的 file_path 參數判斷這次讀取的是不是某個 skill 的 SKILL.md，抓出 skill 名稱
SKILL_PATH_PATTERN = re.compile(r"/skills/([^/]+)/SKILL\.md$")


# ── System Prompt：工具選擇原則、資料不足時的處理、何時使用 skill ──────────
SYSTEM_PROMPT = """你是一位圖書館智慧助理，可以自主使用 hyde_query、vector_retrieve、graph_retrieve 三個工具檢索資料，
並在適當時機使用 vector-result-organizer、graph-result-organizer、final-answer-synthesizer 這幾個 skill。

【工具選擇原則】
各工具適合的問題型態已寫在它們的 docstring 中，請依問題性質判斷要呼叫哪一個；
若問題同時包含結構化過濾條件與語意判斷（例如：某分類目前可借閱的書中，最適合新手的是哪一本），
vector_retrieve、graph_retrieve 都可以呼叫，整合兩邊結果後再回答。
hyde_query 是 vector_retrieve 之前的輔助步驟，只在讀者問題較抽象、籠統時才需要呼叫，
呼叫後把回傳的假想文件當作 vector_retrieve 的 query 參數。

【資料不足時的處理】
若呼叫其中一個工具後查無相關資料，請不要直接判定查無資料，應嘗試改呼叫另一個工具，
確認兩種檢索都查無結果後，才可以判定資料庫中沒有相關資料。

【使用 Skill 的時機】
查完 vector_retrieve 後，使用 vector-result-organizer 整理結果；
查完 graph_retrieve 後，使用 graph-result-organizer 整理結果；
確認已蒐集足夠證據、準備輸出最終答案前，使用 final-answer-synthesizer 套用證據優先順序規則。

請使用繁體中文回答，語氣親切、條理清晰。"""


# ── 執行一次完整的 Agentic RAG 問答流程 ────────────
def run_agentic_rag(query: str) -> dict:
    try:
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
        )
        # recursion_limit 避免 LLM 判斷失準時陷入無止盡的工具呼叫迴圈；
        # deepagents 內建的 read_file／ls 等工具與讀取 skill 都會多佔用幾輪，門檻抓得比較高一些
        result = agent.invoke({"messages": [HumanMessage(content=query)]}, {"recursion_limit": 40})
    except Exception as e:
        # LLM 呼叫本身失敗時（例如逾時、限流）沒有訊息紀錄可還原，回傳友善訊息，error 保留原始例外供除錯
        return {
            "route": "none",
            "retrievals": [],
            "error": str(e),
            "answer": "抱歉，目前系統暫時無法處理您的問題，請稍後再試。",
        }
    # result["messages"] 是這次 ReAct 迴圈完整的訊息紀錄，包含使用者問題、每輪的 tool_calls 與工具回傳結果
    messages = result["messages"]

    # retrievals 從訊息紀錄重新整理還原，依實際執行順序全域編號，彙整每次工具呼叫的原因與結果
    retrievals = []
    # AIMessage 決定呼叫工具時還沒有結果，要等對應的 ToolMessage 出現才能配對合併成完整記錄，
    # pending_calls 就是用 tool_call_id 當 key，記住 tool 名稱／參數／原因
    pending_calls = {}

    # 依序走訪每則訊息，把 AIMessage 的 tool_calls 與對應 ToolMessage 的結果配對還原成 retrievals 過程
    for message in messages:
        # AIMessage 帶 tool_calls 時，代表這一輪 LLM 決定呼叫某個（或多個）工具
        if isinstance(message, AIMessage) and message.tool_calls:
            # 從 content_blocks 篩出 type 為 reasoning 的區塊
            reasoning_blocks = [b for b in message.content_blocks if b.get("type") == "reasoning"]
            # 把各段 reasoning 文字接起來，即為 LLM 呼叫工具前的判斷原因
            reason = " ".join(b.get("reasoning", "") for b in reasoning_blocks).strip()
            # 一輪可能同時呼叫多個工具，逐一處理每個 tool_call
            for tool_call in message.tool_calls:
                # 用 tool_call 的 id 當 key，先記下名稱、參數、原因，等 ToolMessage 回來時取用
                pending_calls[tool_call["id"]] = {
                    "tool": tool_call["name"],
                    "args": tool_call["args"],
                    "reason": reason,
                }

        # ToolMessage 是工具執行後的回傳結果
        elif isinstance(message, ToolMessage):
            # 用 tool_call_id 取回發起呼叫時記下的名稱、參數、原因
            call_info = pending_calls.get(message.tool_call_id, {})
            tool_name = call_info.get("tool", message.name)

            # read_file 讀取的是某個 SKILL.md 時，視為一次 skill 使用紀錄，跟 tool 呼叫共用同一份
            # retrievals 清單、同一組全域編號，才能反映 Agent 實際交錯呼叫 tool／skill 的順序
            if tool_name == "read_file":
                skill_match = SKILL_PATH_PATTERN.search(call_info.get("args", {}).get("file_path", ""))
                if skill_match:
                    retrievals.append(
                        {
                            "index": len(retrievals) + 1,
                            "kind": "skill",
                            "skill": skill_match.group(1),
                            "reason": call_info.get("reason", ""),
                            # read_file 回傳的 SKILL.md 全文（cat -n 格式），讓使用者看到 Agent 實際讀到的指示內容
                            "content": message.content,
                        }
                    )
                # 不是讀 SKILL.md（例如讀到其他檔案）就不記錄，繼續處理下一則訊息
                continue

            # deepagents 其他內建工具（ls／write_todos／task 等）不是我們自訂的檢索工具，且沒有
            # artifact 可用，混進 retrievals 會被誤判成呼叫失敗，故略過
            if tool_name not in RETRIEVAL_TOOL_NAMES:
                continue

            # 合併發起呼叫的資訊與這次的執行結果，組成一筆完整的 retrieval 記錄
            retrievals.append(
                {
                    # 依實際完成順序編號
                    "index": len(retrievals) + 1,
                    "kind": "tool",
                    "tool": tool_name,
                    "args": call_info.get("args", {}),
                    "reason": call_info.get("reason", ""),
                    # artifact 為 None 代表 Tool 呼叫本身拋出例外，需與正常結果分開處理
                    "artifact": message.artifact,
                    # artifact 為 None 時，改用 content（錯誤訊息）填入 error 欄位
                    "error": message.content if message.artifact is None else None,
                }
            )

    # route 欄位在 ReAct 風格下不再是預先決定的路由結果，改用「這次問答實際呼叫過的工具」呈現；
    # 只計算 kind 為 tool 的項目，skill 使用紀錄不算進路由
    called_tools = {r["tool"] for r in retrievals if r["kind"] == "tool"}
    route = "+".join(sorted(called_tools)) if called_tools else "none"

    # 最後一則訊息是 Agent 套用 final-answer-synthesizer skill 後產出的最終答案；
    # 推理模型偶爾只把下一步寫進推理過程卻沒真的送出 tool_call，導致空白 content，視同無效答案
    answer = messages[-1].content or "抱歉，系統已完成檢索，但未能整理出完整回答，請重新輸入問題或稍後再試。"

    # 回傳這次問答的路由、依序編號的檢索記錄與最終回答，供 main.py 顯示
    return {
        "route": route,
        "retrievals": retrievals,
        "answer": answer,
    }
