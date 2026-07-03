"""
Agentic RAG - 知識圖譜檢索 Tool
=================================
graph_tool.py 提供 graph_retrieve() 這個正式的 LangChain Tool，
由 agent.py 中的 LLM 透過 bind_tools() 自主決定要不要呼叫、呼叫幾次。
因此這個 Tool 的 docstring 本身就是 LLM 選擇工具的依據——
docstring 寫得夠清楚，LLM 才能判斷這個工具適不適合目前的問題。

使用 response_format="content_and_artifact"：
  - content  ：格式化過的文字，回傳給 LLM 作為推理依據
  - artifact ：原始資料（dict），保留給 agent.py 還原成結構化結果供 main.py 顯示

  - validate_cypher() ：在真正查詢 Neo4j 前，先檢查 Cypher 是否安全、語法是否正確（內部函式，非 Tool）

知識圖譜設計：
  Nodes：
    - Book(book_id, title, price, description)
    - Category(name)
    - Author(name)
    - Borrower(name)
  Relationships：
    - (Book)-[:BELONGS_TO]->(Category)
    - (Author)-[:WROTE]->(Book)
    - (Borrower)-[:BORROWED {borrowed_at}]->(Book)
"""

# ── 載入套件與環境變數 ──────────────────────────────
import re
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from clients import get_llm, get_neo4j_driver
load_dotenv()

# 模組載入時建立一次連線並重複使用，避免每次呼叫都重新連線
NEO4J_DRIVER = get_neo4j_driver()


# ── 將知識圖譜查詢結果格式化為 LLM 可讀的文字 ────────────
# 作為 Tool 回傳給 LLM 的 content
def format_graph_context(graph_result: dict | None) -> str:
    # graph_result 為 None 代表 Agent 判斷此問題不需要知識圖譜檢索，未執行查詢
    if graph_result is None:
        return "（本次問題經 Agent 判斷不需要知識圖譜檢索，未執行查詢）"

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


# ── Graph Schema：描述知識圖譜結構 ──────────
# LLM 會依照這份 Schema 決定可以查詢哪些節點、屬性與關係
GRAPH_SCHEMA = """
【Node Labels 與屬性】
  (:Book     {book_id, title, price, description})
  (:Category {name})
  (:Author   {name})
  (:Borrower {name})

【Relationship Types】
  (:Book)-[:BELONGS_TO]->(:Category)
  (:Author)-[:WROTE]->(:Book)
  (:Borrower)-[:BORROWED {borrowed_at}]->(:Book)
""".strip()


# ── Few-shot 範例：示範不同類型的問題要如何寫成 Cypher ──────────
# 共同作者不需要額外建立關係，可透過「兩位作者寫過同一本書」推導
FEW_SHOT_EXAMPLES = """
範例 1 — 作者查詢（精確實體匹配）
問題：Jason 寫了哪些書？
Cypher：
MATCH (a:Author {name: "Jason"})-[:WROTE]->(b:Book)
OPTIONAL MATCH (b)-[:BELONGS_TO]->(c:Category)
OPTIONAL MATCH (p:Borrower)-[:BORROWED]->(b)
RETURN b.title AS title, b.price AS price, c.name AS category, p.name AS borrower

範例 2 — 多跳關係推理（合著者的其他作品）
問題：跟 Mary 合著過的作者還寫了什麼書？
Cypher：
MATCH (:Author {name: "Mary"})-[:WROTE]->(:Book)<-[:WROTE]-(coauthor:Author)
MATCH (coauthor)-[:WROTE]->(b:Book)
WHERE NOT (:Author {name: "Mary"})-[:WROTE]->(b)
OPTIONAL MATCH (b)-[:BELONGS_TO]->(c:Category)
RETURN DISTINCT coauthor.name AS coauthor, b.title AS title, c.name AS category

範例 3 — 結構化過濾（類別 + 借閱狀態）
問題：旅遊類有哪些書目前可以借？
Cypher：
MATCH (b:Book)-[:BELONGS_TO]->(:Category {name: "旅遊"})
WHERE NOT (:Borrower)-[:BORROWED]->(b)
OPTIONAL MATCH (a:Author)-[:WROTE]->(b)
RETURN b.title AS title, b.price AS price, collect(a.name) AS authors

範例 4 — 借閱者反向查詢
問題：陳大偉借了哪些書？
Cypher：
MATCH (:Borrower {name: "陳大偉"})-[r:BORROWED]->(b:Book)
OPTIONAL MATCH (b)-[:BELONGS_TO]->(c:Category)
OPTIONAL MATCH (a:Author)-[:WROTE]->(b)
RETURN b.title AS title, c.name AS category, collect(a.name) AS authors, r.borrowed_at AS borrowed_at

範例 5 — 聚合統計（作者著作數）
問題：哪位作者寫的書最多？
Cypher：
MATCH (a:Author)-[:WROTE]->(b:Book)
RETURN a.name AS author, count(b) AS book_count
ORDER BY book_count DESC
LIMIT 5
""".strip()


# ── 只允許出現在 Cypher 中的 READ 關鍵字 ──────────
# 其餘視為禁止的寫入 / 管理指令
FORBIDDEN_KEYWORDS = ("CREATE", "MERGE", "DELETE", "SET", "REMOVE", "LOAD", "CALL", "FOREACH", "DROP")


# ── 呼叫 LLM 將問題轉成 Cypher，若帶入前次錯誤則視為修正重試 ────────────
def generate_cypher(query: str, previous_cypher: str = "", previous_error: str = "") -> str:
    # 建立 LLM，準備將使用者問題轉成 Cypher 查詢，降低隨機性讓相同問題盡量產生一致的 Cypher
    llm = get_llm(temperature=0.0)

    # System Prompt 放入 Graph Schema、Few-shot 範例與嚴格規則，引導 LLM 產生安全且正確的 Cypher
    system_prompt = (
        "你是 Neo4j 圖資料庫的 Cypher 查詢專家，專責將使用者的中文自然語言問題"
        "轉譯為精確可執行的 Cypher 查詢。\n\n"
        "【Graph Schema】\n"
        f"{GRAPH_SCHEMA}\n\n"
        "【範例】\n"
        f"{FEW_SHOT_EXAMPLES}\n\n"
        "【嚴格規則】\n"
        "1. 只能使用 READ 操作：MATCH / OPTIONAL MATCH / WHERE / RETURN / WITH / ORDER BY / LIMIT / collect / count。\n"
        "2. 嚴禁使用 CREATE、MERGE、DELETE、SET、REMOVE、LOAD、CALL、FOREACH 等寫入或管理指令。\n"
        "3. 必須使用 Schema 中定義的 Label 與 Relationship Type，不可自創。\n"
        "4. 字串比對使用完整實體名稱（如 \"Jason\"、\"旅遊\"），若需模糊匹配請用 CONTAINS。\n"
        "5. 回傳欄位應使用可讀性高的別名（AS title、AS author 等）。\n"
        "6. 只輸出純 Cypher 語句，不要加上任何說明、Markdown 標記、反引號或前後綴文字。"
    )

    # 若帶有前次失敗的 Cypher 與錯誤訊息，將其附加到 Human Prompt，讓 LLM 依錯誤修正後重新生成
    if previous_cypher and previous_error:
        human_prompt = (
            f"使用者問題：{query}\n\n"
            f"上一次生成的 Cypher 執行失敗：\n{previous_cypher}\n\n"
            f"失敗原因：{previous_error}\n\n"
            "請修正上述問題，重新輸出可正確執行的 Cypher 查詢："
        )
    else:
        # 一般情況下的 Human Prompt，只放入使用者問題
        human_prompt = f"使用者問題：{query}\n\n請輸出對應的 Cypher 查詢："

    # 將使用者問題、Graph Schema 與 Few-shot 範例傳給 LLM
    response = llm.invoke(
        [
            # System Prompt 設定 LLM 的角色與 Cypher 產生規則
            SystemMessage(content=system_prompt),
            # Human Prompt 傳入使用者問題（必要時附上前次錯誤）
            HumanMessage(content=human_prompt),
        ]
    )
    # 回傳 LLM 生成的 Cypher，並去除頭尾空白
    return response.content.strip()


# ── 驗證 Cypher 是否安全、語法是否正確 ────────────
def validate_cypher(cypher: str) -> tuple[bool, str]:
    # 統一轉大寫後逐一檢查是否包含禁止的寫入 / 管理關鍵字，作為 System Prompt 規則之外的第二道防線
    cypher_upper = cypher.upper()
    for keyword in FORBIDDEN_KEYWORDS:
        # \b 確保是完整關鍵字比對，避免誤判到欄位名稱裡剛好包含的子字串
        if re.search(rf"\b{keyword}\b", cypher_upper):
            return False, f"Cypher 包含禁止的關鍵字：{keyword}"

    try:
        with NEO4J_DRIVER.session() as session:
            # EXPLAIN 只會編譯查詢計畫、不會真正執行，可在查詢前先驗證語法是否正確
            session.run(f"EXPLAIN {cypher}")
        # EXPLAIN 未拋出例外，代表語法正確
        return True, ""
    except Exception as e:
        # 語法錯誤時回傳 Neo4j 提供的錯誤訊息，供 generate_cypher() 修正重試
        return False, str(e)


# ── 在 Neo4j 中實際執行已驗證過的 Cypher ────────────
def run_cypher(cypher: str) -> list[dict]:
    with NEO4J_DRIVER.session() as session:
        # 執行 Cypher 查詢
        result = session.run(cypher)
        # data() 會把查詢結果轉成 dict list，方便 agent.py 與 format_graph_context() 使用
        return result.data()


# ── 將問題轉成 Cypher，驗證通過後查詢 Neo4j，失敗時自動重試 ────────────
@tool(response_format="content_and_artifact")
def graph_retrieve(query: str, max_retries: int = 2) -> tuple[str, dict]:
    """知識圖譜檢索書籍資料，適用於精確實體查詢（某作者寫了哪些書、某人借了哪些書）、
    結構化過濾（某分類目前可借閱的書）、多跳關係推理（合著者還寫了什麼書）、
    聚合統計（哪位作者寫的書最多）等精確事實型問題。
    不適合語意模糊、情境推薦類問題（這類問題請改用 vector_retrieve）。
    內部會先將問題轉成 Cypher，驗證語法與安全性後才查詢 Neo4j，失敗時最多自動重試兩次。

    Args:
        query: 讀者的原始問題。
        max_retries: Cypher 生成或執行失敗時最多重試幾次，預設 2。
    """
    # 記錄目前嘗試的 Cypher 與錯誤訊息，初次呼叫 generate_cypher() 時皆為空字串
    cypher = ""
    error = ""

    # attempt 從 0 開始，最多嘗試 max_retries + 1 次（第一次生成 + 最多 max_retries 次重試）
    for attempt in range(max_retries + 1):
        try:
            # 若是第一次嘗試，previous_cypher／previous_error 皆為空字串；重試時則帶入前次失敗資訊
            cypher = generate_cypher(query, previous_cypher=cypher, previous_error=error)
        except Exception as e:
            # LLM 呼叫本身失敗（例如逾時、限流），視同這一輪生成失敗，記錄錯誤後重試
            error = str(e)
            continue

        # 執行前先驗證 Cypher 是否安全、語法是否正確
        is_valid, validate_error = validate_cypher(cypher)
        if not is_valid:
            # 驗證失敗，記下錯誤訊息，進入下一輪重試（若已達重試上限則跳出迴圈）
            error = validate_error
            continue

        try:
            # 驗證通過，實際查詢 Neo4j
            results = run_cypher(cypher)
        except Exception as e:
            # 少數情況語法驗證通過但執行時仍失敗（例如資料型別不符），同樣記錄錯誤並重試
            error = str(e)
            continue

        # 查詢成功，組成結果 dict，error 為空字串代表成功
        result = {
            "cypher": cypher,
            "results": results,
            "error": "",
            "retries": attempt,
        }
        # content 給 LLM 閱讀；result（artifact）保留原始資料供 agent.py 還原結構化結果
        return format_graph_context(result), result

    # 已達重試上限仍未成功，組成結果 dict，帶入最後一次的 Cypher 與錯誤訊息
    result = {
        "cypher": cypher,
        "results": [],
        "error": error,
        "retries": max_retries,
    }
    return format_graph_context(result), result
