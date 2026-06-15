"""
Graph RAG - Phase 1：建立知識圖譜索引
=================================
index.py 負責將結構化 JSON 欄位轉成 Neo4j 知識圖譜。

這個階段只處理資料中已經明確存在的結構：
  - 書籍
  - 分類
  - 作者
  - 借閱者

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

執行方式：
  python index.py
"""

# ── 載入套件與環境變數 ──────────────────────────────
import json
import os
from dotenv import load_dotenv
from neo4j import GraphDatabase
load_dotenv()


# ── 讀取圖書館書籍資料 ────────────────────────────
def load_data() -> list[dict]:
    # 以 UTF-8 開啟 JSON 資料檔
    with open("data/library_records.json", "r", encoding="utf-8") as f:
        # 將 JSON 內容解析成 Python list
        records = json.load(f)
    # 輸出讀取筆數
    print(f"已讀取 {len(records)} 筆書籍資料（資料來源：data/library_records.json）")
    # 回傳書籍 list
    return records


# ── 將書籍資料建立成 Neo4j 知識圖譜索引 ──────────
def build_graph_index(records: list[dict]) -> None:
    # 透過 Cypher 查詢語言，將每一筆 JSON 書籍資料轉成圖資料庫中的「節點」與「關係」：
    #   - Book 節點代表一本書，保存書名、價格與 description 等可回答問題的書籍資訊
    #   - Category 節點代表書籍分類，用來回答某分類有哪些書、某本書屬於哪類等問題
    #   - Author 節點代表作者，用來回答某作者寫了哪些書、某本書有哪些作者等問題
    #   - Borrower 節點代表目前借閱者，用來回答書籍是否被借出、誰借了哪本書等問題
    # 節點之間會用關係連起來，讓查詢可以沿著知識圖譜尋找答案：
    #   - (Book)-[:BELONGS_TO]->(Category)：表示這本書屬於某個分類
    #   - (Author)-[:WROTE]->(Book)：表示這位作者寫了這本書
    #   - (Borrower)-[:BORROWED]->(Book)：表示這位借閱者借了這本書

    # 連線到 Neo4j 圖資料庫
    driver = GraphDatabase.driver(
        # Neo4j 服務的連線位址
        "bolt://localhost:7687",
        # 從環境變數取得 Neo4j 帳號與密碼
        auth=(os.environ["NEO4J_USERNAME"], os.environ["NEO4J_PASSWORD"]),
    )

    # 以 session 為單位執行 Cypher，離開 with 區塊時自動關閉 session
    with driver.session() as session:
        # 清空 Neo4j 圖資料庫中的知識圖譜索引，避免保留上一次實驗留下的舊知識圖譜索引
        session.run("MATCH (n) DETACH DELETE n")
        print("已清除舊知識圖譜")

        # 建立 Book / Category / Author / Borrower 節點防重複規則
        # 這些規則告訴 Neo4j：同一本書、同一個分類、同一位作者、同一位借閱者只能建立一次
        for cypher in [
            # Book 以 book_id 作為唯一識別
            "CREATE CONSTRAINT book_id_unique IF NOT EXISTS FOR (b:Book) REQUIRE b.book_id IS UNIQUE",
            # Category 以 name 作為唯一識別
            "CREATE CONSTRAINT category_name_unique IF NOT EXISTS FOR (c:Category) REQUIRE c.name IS UNIQUE",
            # Author 以 name 作為唯一識別
            "CREATE CONSTRAINT author_name_unique IF NOT EXISTS FOR (a:Author) REQUIRE a.name IS UNIQUE",
            # Borrower 以 name 作為唯一識別
            "CREATE CONSTRAINT borrower_name_unique IF NOT EXISTS FOR (p:Borrower) REQUIRE p.name IS UNIQUE",
        ]:
            # 逐條執行防重複規則
            session.run(cypher)
        print("已建立 Book / Category / Author / Borrower 節點防重複規則")

        # 逐筆建立知識圖譜索引：
        #   節點：Book、Category、Author、Borrower
        #   關係：BELONGS_TO、WROTE、BORROWED
        print("正在建立 Neo4j 知識圖譜，請稍候...")
        # 用計數器 index 為每本書標上編號，從 1 開始，方便顯示進度
        for index, record in enumerate(records, 1):
            # 書籍唯一識別碼
            book_id = record["_id"]
            # 書名
            title = record["book"]
            # 書籍分類
            category = record["category"]
            # 售價
            price = float(record["price"])
            # 書籍描述；若不存在則為空字串
            description = record.get("description", "")
            # 作者清單；若不存在則為空 list
            authors = record.get("authors", [])

            # 建立 Book 節點：
            #   - 代表一本書
            #   - 以 book_id 作為唯一識別
            #   - 保存 title、price、description 屬性
            # 建立 Category 節點：
            #   - 代表書籍分類
            #   - 以 name 作為唯一識別
            # 建立 BELONGS_TO 關係：
            #   - (Book)-[:BELONGS_TO]->(Category)
            #   - 表示「這本書屬於哪個分類」
            session.run(
                """
                MERGE (b:Book {book_id: $book_id})
                SET b.title = $title,
                    b.price = $price,
                    b.description = $description
                MERGE (c:Category {name: $category})
                MERGE (b)-[:BELONGS_TO]->(c)
                """,
                # 以參數帶入書籍資料，避免字串拼接造成 Cypher 注入
                book_id=book_id,
                title=title,
                price=price,
                description=description,
                category=category,
            )

            # 建立 Author 節點：
            #   - 代表一位作者
            #   - 以 name 作為唯一識別
            # 建立 WROTE 關係：
            #   - (Author)-[:WROTE]->(Book)
            #   - 表示「這位作者寫了這本書」
            for author in authors:
                # 先比對既有的 Book 節點，再建立作者與 WROTE 關係
                session.run(
                    """
                    MATCH (b:Book {book_id: $book_id})
                    MERGE (a:Author {name: $author})
                    MERGE (a)-[:WROTE]->(b)
                    """,
                    book_id=book_id,
                    author=author,
                )

            # 取得借閱者資訊；若欄位不存在或為 null，預設為空 dict
            borrower = record.get("borrower") or {}
            # 取得借閱者姓名
            borrower_name = borrower.get("name")
            # 如果這本書目前有借閱紀錄：
            # 建立 Borrower 節點：
            #   - 代表一位借閱者
            #   - 以 name 作為唯一識別
            # 建立 BORROWED 關係：
            #   - (Borrower)-[:BORROWED]->(Book)
            #   - 表示「這位借閱者借了這本書」
            #   - 在關係上保存 borrowed_at 屬性，記錄借閱時間
            if borrower_name:
                # 取得借閱時間戳記；若不存在則為空字串，避免 KeyError
                borrowed_at = (borrower.get("timestamp") or {}).get("$date", "")
                # 先比對既有的 Book 節點，再建立借閱者與 BORROWED 關係
                session.run(
                    """
                    MATCH (b:Book {book_id: $book_id})
                    MERGE (p:Borrower {name: $borrower_name})
                    MERGE (p)-[r:BORROWED]->(b)
                    SET r.borrowed_at = $borrowed_at
                    """,
                    book_id=book_id,
                    borrower_name=borrower_name,
                    borrowed_at=borrowed_at,
                )

            # 顯示當前書籍索引進度
            print(f"[{index}/{len(records)}] {record['book']}")

        print("知識圖譜建立完成")

        print("\n知識圖譜統計：")
        # 列印知識圖譜統計資訊，確認建立了多少節點與關係
        for name, cypher in {
            # 統計 Book 節點數
            "Book": "MATCH (b:Book) RETURN count(b) AS count",
            # 統計 Category 節點數
            "Category": "MATCH (c:Category) RETURN count(c) AS count",
            # 統計 Author 節點數
            "Author": "MATCH (a:Author) RETURN count(a) AS count",
            # 統計 Borrower 節點數
            "Borrower": "MATCH (p:Borrower) RETURN count(p) AS count",
            # 統計 BELONGS_TO 關係數
            "BELONGS_TO": "MATCH ()-[r:BELONGS_TO]->() RETURN count(r) AS count",
            # 統計 WROTE 關係數
            "WROTE": "MATCH ()-[r:WROTE]->() RETURN count(r) AS count",
            # 統計 BORROWED 關係數
            "BORROWED": "MATCH ()-[r:BORROWED]->() RETURN count(r) AS count",
        }.items():
            # 執行統計 Cypher 並取出 count 欄位
            count = session.run(cypher).single()["count"]
            # 對齊輸出節點或關係名稱與數量
            print(f"  - {name:<20} {count}")

    # 關閉 Neo4j 連線，釋放資源
    driver.close()


# ── 主程式進入點 ─────────────────────────
def main() -> None:
    # 顯示目前的執行階段
    print("=== Phase 1：建立知識圖譜索引 ===")
    # 從 JSON 檔案讀取書籍原始資料
    records = load_data()
    # 將原始資料建立成 Neo4j 知識圖譜索引
    build_graph_index(records)
    # 顯示完成索引
    print(f"成功建立知識圖譜索引，共處理 {len(records)} 本書，可至 http://localhost:7474 瀏覽")


# 確保此檔案被直接執行時才呼叫 main()，被 import 時不執行
if __name__ == "__main__":
    main()
