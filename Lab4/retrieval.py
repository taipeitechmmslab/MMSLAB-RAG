"""
HyDE - Phase 2：語意檢索
=================================
retrieval.py 負責將使用者問題改寫成 HyDE 假想文件後轉成向量，並從 Milvus 找出最相關的書籍資料。

此模組提供 retrieve() 函式供 main.py 呼叫。
"""

# ── 載入套件與環境變數 ──────────────────────────────
import os
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_milvus import Milvus
from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings
load_dotenv()

# ── 根據使用者問題生成假想文件 ────────────────
def generate_hypothetical_document(query: str) -> str:
    # 初始化 NVIDIA NIM LLM
    llm = ChatNVIDIA(
        # 從環境變數取得 LLM 模型名稱
        model=os.environ.get("LLM_MODEL"),
        # 從環境變數取得 NVIDIA API 金鑰
        api_key=os.environ.get("NVIDIA_NIM_API_KEY"),
    )
    # System Prompt 限制 LLM 只生成適合向量檢索的描述，避免直接回答問題
    system_prompt = (
        "你是圖書館館藏檢索助理。"
        "請根據讀者問題，生成一段可能出現在相關書籍資料中的內容介紹。"
        "不要回答問題，不要列出真實書名，不要編造館藏資料。"
        "只需要生成一段適合向量檢索的繁體中文描述。"
    )
    # Human Prompt 傳入使用者問題
    human_prompt = f"讀者問題：{query}"

    # 組合 System Prompt 與 Human Prompt，呼叫 LLM 生成假想文件
    response = llm.invoke([
        # System Prompt 設定 LLM 的角色與規則
        SystemMessage(content=system_prompt),
        # Human Prompt 傳入使用者問題
        HumanMessage(content=human_prompt),
    ])

    # 回傳假想文件內容，並去除空字元
    return response.content.strip()

# ── 使用假想文件從 Milvus 向量資料庫檢索書籍 ────────────
def retrieve(query: str, top_k: int = 5) -> list[dict]:
    # 建立 Milvus 向量資料庫的連線物件
    vector_store = Milvus(
        #問題向量需與建索引時使用相同的 Embedding Model
        embedding_function=NVIDIAEmbeddings(
            # 從環境變數取得 Embedding 模型名稱
            model=os.environ.get("EMBEDDING_MODEL"),
            # 從環境變數取得 NVIDIA API 金鑰
            api_key=os.environ.get("NVIDIA_NIM_API_KEY"),
        ),
        # 指定要查詢的 Milvus collection 名稱
        collection_name="library_books",
        # Milvus 服務的連線位址
        connection_args={"uri": "http://localhost:19530"},
        # 與建索引時一致開啟動態欄位，搜尋才會把 metadata 一起帶回
        enable_dynamic_field=True,
    )

    # 同一本書可能有多個 chunks，先多撈 search_k 個再去重，才能湊出 top_k 本不同書
    # search_k 至少為 10，或 top_k 的 3 倍，以免去重後書籍數量不足
    search_k = max(10, top_k * 3)

    # 將使用者問題轉成假想文件，作為語意檢索的問題向量來源
    hyde_query = generate_hypothetical_document(query)
    # 印出假想文件，方便觀察 HyDE 生成的內容
    print("HyDE 生成的假想文件：")
    print(hyde_query)
    # 用假想文件向量在 Milvus 中搜尋最相近的 search_k 個 chunks，同時回傳相似度分數
    results = vector_store.similarity_search_with_score(hyde_query, k=search_k)

    # 用 set 記錄已出現的 book_id，避免同一本書重複加入結果
    seen_book_ids = set()
    # 初始化結果 list，存放最終回傳的書籍資訊
    docs = []
    # results 已依相似度由高到低排序，每本書取第一次出現的（即最相關的 chunk）
    for doc, score in results:
        # 從 metadata 取得此 chunk 對應的書籍 ID
        book_id = doc.metadata.get("book_id")
        # 若 book_id 不存在或該書已加入結果，則跳過此 chunk
        if not book_id or book_id in seen_book_ids:
            continue

        # 將此書的 book_id 加入 seen 集合，後續相同書籍的 chunk 會被跳過
        seen_book_ids.add(book_id)
        # 將書籍的 metadata、命中的 chunk 內容、相似度分數加入結果 list
        docs.append({
            # metadata 包含書名、作者、借閱狀態等書籍資訊
            "metadata": doc.metadata,
            # 命中的 chunk 原文，可用來檢視為何此書被檢索到
            "matched_page_content": doc.page_content,
            # 相似度分數，四捨五入至小數點後四位
            "score": round(float(score), 4),
        })

        # 已收集到 top_k 本不同書籍時提前結束迴圈
        if len(docs) >= top_k:
            break

    # 回傳最多 top_k 本書籍的檢索結果
    return docs
