"""
Hybrid RAG - Phase 2：混合檢索
=================================
retrieval.py 對單一問題同時提供向量檢索與知識圖譜查詢兩條路徑：
  - vector_retrieve(): Milvus 向量相似搜尋
  - graph_retrieve(): LLM 生成 Cypher 並查詢 Neo4j

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

此模組提供 vector_retrieve() 與 graph_retrieve() 函式供 main.py 呼叫。
"""

# ── 載入套件與環境變數 ──────────────────────────────
import os
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_milvus import Milvus
from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings
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


# ── 使用問題向量從 Milvus 向量資料庫檢索書籍 ────────────
def vector_retrieve(query: str, top_k: int = 5) -> list[dict]:
    # 建立 Milvus 向量資料庫的連線物件
    vector_store = Milvus(
        # 問題向量需與建索引時使用相同的 Embedding Model
        embedding_function=NVIDIAEmbeddings(
            # 從環境變數取得 Embedding 模型名稱
            model=os.environ.get("EMBEDDING_MODEL"),
            # 從環境變數取得 NVIDIA NIM API 金鑰
            api_key=os.environ.get("NVIDIA_NIM_API_KEY"),
            # 最多等待 Embedding Model 回應 60 秒
            timeout=60,
        ),
        # 指定要查詢的 Milvus collection 名稱
        collection_name="library_books",
        # Milvus 服務的連線位址
        connection_args={"uri": "http://localhost:19530"},
        # 與建索引時一致開啟動態欄位，搜尋才會把 metadata 一起帶回
        enable_dynamic_field=True,
    )

    # top_k 代表最後要回傳的書籍數量；search_k 代表先從 Milvus 取回的 chunks 數量
    # 因為同一本書可能有多個 chunks 出現在搜尋結果中，所以先取回較多 chunks
    # 再從中整理出最多 top_k 本不同書籍
    search_k = max(10, top_k * 3)
    # 將使用者問題轉成問題向量，在 Milvus 中搜尋最相近的 search_k 個 chunks，同時回傳相似度分數
    results = vector_store.similarity_search_with_score(query, k=search_k)

    # 用 set 記錄已出現的 book_id，避免同一本書重複加入結果
    seen_book_ids = set()
    # 初始化結果 list，存放最終回傳的書籍資訊
    docs = []

    # results 已依相似度由高到低排序，每本書取第一次出現的（即最相關的 chunk）
    for doc, score in results:
        # 從 metadata 取得此 chunk 對應的書籍 ID
        book_id = doc.metadata.get("book_id")
        # 依 book_id 整理搜尋結果，避免同一本書重複出現在推薦結果中
        if not book_id or book_id in seen_book_ids:
            continue

        # 將此書的 book_id 加入 seen 集合，後續相同書籍的 chunk 會被跳過
        seen_book_ids.add(book_id)
        # 將書籍的 metadata、命中的 chunk 內容、相似度分數加入結果 list
        docs.append(
            {
                # metadata 包含書名、作者、借閱狀態等書籍資訊
                "metadata": doc.metadata,
                # 命中的 chunk 原文，可用來檢視為何此書被檢索到
                "matched_page_content": doc.page_content,
                # 相似度分數，四捨五入至小數點後四位
                "score": round(float(score), 4),
            }
        )

        # 已收集到 top_k 本不同書籍時提前結束迴圈
        if len(docs) >= top_k:
            break
    # 回傳最多 top_k 本不同書籍的推薦結果
    return docs


# ── 將問題轉成 Cypher 並從 Neo4j 知識圖譜檢索 ────────────
def graph_retrieve(query: str) -> dict:
    # 預設 cypher 為空字串，若在生成前就發生錯誤也能一併回傳
    cypher = ""
    try:
        # 建立 LLM，準備將使用者問題轉成 Cypher 查詢
        llm = ChatNVIDIA(
            # 從環境變數取得 LLM 模型名稱
            model=os.environ.get("LLM_MODEL"),
            # 從環境變數取得 NVIDIA NIM API 金鑰
            api_key=os.environ.get("NVIDIA_NIM_API_KEY"),
            # 最多等待 LLM 回應 60 秒，逾時後會拋出 Timeout 例外
            timeout=60,
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
        response = llm.invoke(
            [
                # System Prompt 設定 LLM 的角色與 Cypher 產生規則
                SystemMessage(content=system_prompt),
                # Human Prompt 傳入使用者問題
                HumanMessage(content=human_prompt),
            ]
        )
        # 取得 LLM 回傳的 Cypher 查詢，並去除頭尾空白
        cypher = response.content.strip()

        # 連線 Neo4j 圖資料庫
        driver = GraphDatabase.driver(
            # Neo4j 服務的連線位址
            "bolt://localhost:7687",
            # 從環境變數取得 Neo4j 帳號與密碼，未設定時使用預設值
            auth=(
                os.environ.get("NEO4J_USERNAME", "neo4j"),
                os.environ.get("NEO4J_PASSWORD", "graphrag"),
            ),
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
        # 將 LLM 或 Neo4j 的錯誤交給 generation.py，讓 LLM 知道知識圖譜查詢失敗
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
