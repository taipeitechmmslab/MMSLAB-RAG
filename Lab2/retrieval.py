"""
Graph RAG - Phase 2：知識圖譜檢索
=================================
retrieval.py 負責接收使用者問題，透過 LLM 將自然語言轉譯為 Cypher 查詢，
再於 Neo4j 知識圖譜執行查詢，回傳結構化結果。

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

此模組提供 retrieve() 函式供 main.py 呼叫。
"""

# ── 載入套件與環境變數 ──────────────────────────────
import os
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from neo4j import GraphDatabase
load_dotenv()


# ── Graph Schema：描述 index.py 建立出來的知識圖譜結構 ──────────
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


# ── 將問題轉成 Cypher 並從 Neo4j 知識圖譜檢索 ────────────
def retrieve(query: str) -> dict:
    # 預設 cypher 為空字串，若在生成前就發生錯誤也能一併回傳
    cypher = ""
    try:
        # 先確認 Neo4j 裡已有 Book 節點，避免還沒索引時就呼叫 LLM
        with GraphDatabase.driver(
            # Neo4j 服務的連線位址
            "bolt://localhost:7687",
            # 從環境變數取得 Neo4j 帳號與密碼
            auth=(os.environ["NEO4J_USERNAME"], os.environ["NEO4J_PASSWORD"]),
        ) as driver:
            # 以 session 執行查詢，離開 with 區塊時自動關閉
            with driver.session() as session:
                # 統計目前知識圖譜中的 Book 節點數
                book_count = session.run(
                    "MATCH (b:Book) RETURN count(b) AS book_count"
                ).single()["book_count"]

        # 沒有任何 Book 節點代表尚未建立索引，丟出例外交由下方 except 處理
        if book_count == 0:
            raise ValueError("Book index is empty")
    except Exception:
        # 尚未索引或連線失敗時，回傳明確提示供 main.py 顯示
        return {
            "cypher": "",
            "results": [],
            "error": "請先執行 index.py 進行書籍索引",
        }

    try:
        # 建立 LLM，準備將使用者問題轉成 Cypher 查詢
        llm = ChatNVIDIA(
            # 從環境變數取得 LLM 模型名稱
            model=os.environ.get("LLM_MODEL"),
            # 從環境變數取得 NVIDIA API 金鑰
            api_key=os.environ.get("NVIDIA_LLM_API_KEY"),
            # 降低 LLM 回答時的隨機性，讓相同問題盡量產生一致的 Cypher
            temperature=0.0,
        )

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

        # Human Prompt 放入使用者問題，要求 LLM 輸出對應的 Cypher
        human_prompt = f"使用者問題：{query}\n\n請輸出對應的 Cypher 查詢："

        # 將使用者問題、Graph Schema 與 Few-shot 範例傳給 LLM
        response = llm.invoke([
            # System Prompt 設定 LLM 的角色與 Cypher 產生規則
            SystemMessage(content=system_prompt),
            # Human Prompt 傳入使用者問題
            HumanMessage(content=human_prompt),
        ])

        # 取得 LLM 回傳的 Cypher 查詢，並去除頭尾空白
        cypher = response.content.strip()

        # 連線 Neo4j 圖資料庫
        driver = GraphDatabase.driver(
            # Neo4j 服務的連線位址
            "bolt://localhost:7687",
            # 從環境變數取得 Neo4j 帳號與密碼
            auth=(os.environ["NEO4J_USERNAME"], os.environ["NEO4J_PASSWORD"]),
        )

        try:
            # 以 session 執行 LLM 產生的 Cypher
            with driver.session() as session:
                # 執行 LLM 產生的 Cypher 查詢
                result = session.run(cypher)
                # data() 會把查詢結果轉成 dict list，方便 main.py 與 generation.py 使用
                results = result.data()
        finally:
            # 無論查詢成功與否，都關閉 Neo4j 連線
            driver.close()
    except Exception as e:
        # 如果 LLM 或 Neo4j 查詢失敗，回傳錯誤訊息供 main.py 顯示
        return {
            "cypher": cypher,
            "results": [],
            "error": str(e),
        }

    # 回傳 Cypher 與 Neo4j 查詢結果，供 main.py 與 generation.py 使用
    return {
        "cypher": cypher,
        "results": results,
        "error": "",
    }
