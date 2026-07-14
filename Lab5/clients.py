"""
Agentic RAG - 共用外部資源連線
=================================
clients.py 集中管理 LLM、Embedding、Neo4j、Milvus 的連線設定與預設值，
index.py（建索引時）與 tools/ package（查詢時）皆呼叫這裡提供的函式，
避免同一份連線邏輯在多個檔案裡重複維護、甚至寫出不一致的行為。
"""

# ── 載入套件 ──────────────────────────────
import os
from langchain_milvus import Milvus
from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings
from neo4j import Driver, GraphDatabase

# ── 外部服務連線位址與 collection 名稱 ──────────────────────────────
NEO4J_URI = "bolt://localhost:7687"
MILVUS_URI = "http://localhost:19530"
MILVUS_COLLECTION = "library_books"


# ── 建立 NVIDIA NIM LLM ────────────
def get_llm(**kwargs) -> ChatNVIDIA:
    # kwargs 可傳入 temperature 等額外設定，最後交給 ChatNVIDIA 使用
    # 開啟支援模型的推理內容；不支援的模型會忽略此設定
    # max_tokens=None 可避免 ChatNVIDIA 預設 1024 tokens 截斷長答案
    # 長答案生成時間較久，因此把 timeout 從預設 60 秒提高到 120 秒
    return ChatNVIDIA(
        model=os.environ.get("LLM_MODEL"),
        api_key=os.environ.get("NVIDIA_NIM_API_KEY"),
        model_kwargs={"chat_template_kwargs": {"thinking": True}},
        max_tokens=None,
        timeout=120,
        **kwargs,
    )


# ── 建立 NVIDIA NIM Embedding Model ────────────
def get_embeddings() -> NVIDIAEmbeddings:
    # 從環境變數取得 Embedding 模型名稱與 NVIDIA NIM API 金鑰
    return NVIDIAEmbeddings(
        model=os.environ.get("EMBEDDING_MODEL"),
        api_key=os.environ.get("NVIDIA_NIM_API_KEY"),
        # 最多等待 Embedding Model 回應 60 秒
        timeout=60,
    )


# ── 建立 Neo4j 連線 ────────────
def get_neo4j_driver() -> Driver:
    # 連線 Neo4j 圖資料庫，帳號密碼從環境變數取得，未設定時使用預設值
    return GraphDatabase.driver(
        NEO4J_URI,
        auth=(
            os.environ.get("NEO4J_USERNAME", "neo4j"),
            os.environ.get("NEO4J_PASSWORD", "graphrag"),
        ),
    )


# ── 建立 Milvus 向量資料庫連線（查詢用） ────────────
def get_vector_store() -> Milvus:
    # 問題向量需與建索引時使用相同的 Embedding Model
    return Milvus(
        embedding_function=get_embeddings(),
        # 指定要查詢的 Milvus collection 名稱
        collection_name=MILVUS_COLLECTION,
        # Milvus 服務的連線位址
        connection_args={"uri": MILVUS_URI},
        # 與建索引時一致開啟動態欄位，搜尋才會把 metadata 一起帶回
        enable_dynamic_field=True,
    )
