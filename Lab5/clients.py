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
    # 從環境變數取得 LLM 模型名稱與 NVIDIA NIM API 金鑰，kwargs 轉傳給 ChatNVIDIA（例如 temperature）
    # model_kwargs 是 ChatNVIDIA 官方提供的「額外參數」欄位，會直接轉送進 API 請求；
    # chat_template_kwargs={"thinking": True}：部分推理模型（如 Qwen3 系列）預設不輸出思考過程，
    # 需明確開啟才會填 reasoning_content；已實測對不支援此參數的模型（如 gpt-oss）無影響
    # max_tokens=None：ChatNVIDIA 預設值只有 1024，長答案容易被截斷；明確設 None 會讓請求
    # 不帶這個欄位，改用各模型自己在 NVIDIA NIM 上的預設上限（此參數已過時但仍正確生效，
    # 改用新名稱 max_completion_tokens=None 反而因套件本身的邏輯漏洞不會生效）
    # timeout=120：拿掉 max_tokens 上限後生成時間變長，預設 60 秒的連線逾時較容易被撞到
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
