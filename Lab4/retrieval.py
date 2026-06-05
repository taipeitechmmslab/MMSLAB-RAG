"""
Vector RAG - Phase 2：語意檢索
=================================
retrieval.py 負責將使用者問題改寫成 HyDE 假想文件後轉成向量，並從 Milvus 找出最相關的書籍資料。

執行流程：
  0. 載入套件與環境變數
  1. 使用 NVIDIA NIM LLM 根據使用者問題生成 HyDE 假想文件
  2. 連線 Milvus library_books collection
  3. 使用 HyDE 假想文件搜尋 search_k 個相關 chunks
  4. 依 book_id 整理搜尋結果，避免同一本書重複出現在推薦結果中
  5. 回傳最多 top_k 本不同書籍的推薦結果

此模組提供 retrieve() 函式供 main.py 呼叫。
"""

# 載入套件與環境變數
import os
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_milvus import Milvus
from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings
load_dotenv()

# 根據使用者問題生成 HyDE 假想文件，作為語意檢索用的查詢
def generate_hypothetical_document(query: str) -> str:
    # 使用 NVIDIA NIM LLM 根據使用者問題生成 HyDE 假想文件
    llm = ChatNVIDIA(
        model=os.environ.get("LLM_MODEL"),
        api_key=os.environ.get("NVIDIA_LLM_API_KEY"),
    )

    system_prompt = (
        "你是圖書館館藏檢索助理。"
        "請根據讀者問題，生成一段可能出現在相關書籍資料中的內容介紹。"
        "不要回答問題，不要列出真實書名，不要編造館藏資料。"
        "只需要生成一段適合向量檢索的繁體中文描述。"
    )
    human_prompt = f"讀者問題：{query}"

    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=human_prompt),
    ])

    return response.content.strip()

# 使用 HyDE 假想文件搜尋 Milvus，回傳最相關的 top_k 本不同書籍
def retrieve(query: str, top_k: int = 3) -> list[dict]:
    # 連線 Milvus library_books collection
    vector_store = Milvus(
        # 初始化 NVIDIA NIM Embedding Model
        embedding_function=NVIDIAEmbeddings(
            model=os.environ.get("EMBEDDING_MODEL"),
            api_key=os.environ.get("NVIDIA_LLM_API_KEY"),
        ),
        collection_name="library_books",
        connection_args={"uri": "http://localhost:19530"},
        # metadata（book_id、書名、作者…）在建索引時是存進 Milvus 的動態欄位
        # 這裡開啟動態欄位，搜尋才會把這些 metadata 一起回傳
        enable_dynamic_field=True,
    )

    # top_k 代表最後要回傳的書籍數量
    # search_k 代表先從 Milvus 取回的 chunks 數量
    # 因為同一本書可能有多個 chunks 出現在搜尋結果中，所以先取回多個 chunks
    # 再從多個 chunks 整理出 top_k 本不同書籍
    search_k = max(10, top_k * 3)

    # 生成假想文件
    hyde_query = generate_hypothetical_document(query)
    print("HyDE 生成的假想文件：")
    print(hyde_query)
    # 使用 HyDE 假想文件搜尋 search_k 個相關 chunks
    results = vector_store.similarity_search_with_score(hyde_query, k=search_k)

    seen_book_ids = set()
    docs = []

    # results 已依相似度排序，每本書第一次出現的就是它最相關的 chunk
    for doc, score in results:
        book_id = doc.metadata.get("book_id")
        # 依 book_id 整理搜尋結果，避免同一本書重複出現在推薦結果中
        if not book_id or book_id in seen_book_ids:
            continue

        seen_book_ids.add(book_id)
        docs.append({
            "metadata": doc.metadata,
            "matched_page_content": doc.page_content,
            "score": round(float(score), 4),
        })

        if len(docs) >= top_k:
            break

    # 回傳最多 top_k 本不同書籍的推薦結果
    return docs
