"""
HyDE - Phase 1：建立索引
=================================
index.py 負責將圖書館書籍資料轉成可檢索的向量。

執行方式：
  python index.py
"""

# ── 載入套件與環境變數 ──────────────────────────────
import json
import os
from langchain_core.documents import Document
from langchain_milvus import Milvus
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
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

# ── 將圖書館書籍資料轉成 Document 資料結構──────────
def build_documents(records: list[dict]) -> list[Document]:
    # 初始化空 list，稍後逐一 append Document
    documents = []
    # 逐筆遍歷每本書的資料
    for record in records:
        # 取得借閱者資訊；若欄位不存在或為 null，預設為空 dict
        borrower = record.get("borrower") or {}
        # 取得借閱時間戳記；若不存在則為空 dict，避免 KeyError
        timestamp = borrower.get("timestamp") or {}
        # 建立 metadata dict，儲存不需被搜尋但有用的書籍資訊
        metadata = {
            # 書籍唯一識別碼
            "book_id": record["_id"],
            # 書名
            "book": record["book"],
            # 書籍分類
            "category": record["category"],
            # 將作者清單合併成一個字串，例如 ["張三", "李四"] → "張三, 李四"
            "authors": ", ".join(record["authors"]),
            # 售價
            "price": float(record["price"]),
            # 是否已借出
            "is_borrowed": bool(borrower),
            # 借閱者姓名；未借出時為空字串
            "borrower_name": borrower.get("name", ""),
            # 借閱日期；未借出時為空字串
            "borrowed_at": timestamp.get("$date", ""),
        }
        # 有借閱者資料就顯示「已借出」，否則顯示「可借閱」
        borrowed_text = "已借出" if borrower else "可借閱"
        # 取得借閱者姓名
        borrower_name = borrower.get("name", "")
        # 組合成自然語言格式的字串，供語意檢索使用
        searchable_text = (
            # 書名欄位
            f"書名：{record['book']}\n"
            # 分類欄位
            f"分類：{record['category']}\n"
            # 作者欄位
            f"作者：{', '.join(record['authors'])}\n"
            # 借閱狀態（已借出 / 可借閱）
            f"借閱狀態：{borrowed_text}\n"
            # 借閱者姓名，若無則顯示「無」
            f"借閱者：{borrower_name if borrower_name else '無'}\n"
            # 書籍描述，是語意相似度計算的核心文字
            f"內容介紹：{record['description']}"
        )
        # 將組好的 Document 加入 list
        documents.append(
            Document(
                # page_content 是向量化的對象，會被 Embedding Model 處理
                page_content=searchable_text,
                # metadata 不會被向量化，只在檢索命中後一起回傳
                metadata=metadata,
            )
        )
    # 輸出轉換完成的 Document 數量
    print(f"已轉換成 {len(documents)} 份 Document")
    # 回傳 Document list
    return documents


# ── 將 Documents 切成多個 chunk ──────────
def split_documents(documents: list[Document]) -> list[Document]:
    # 建立文字切割器，遞迴嘗試在段落、句子、字元等位置切割
    splitter = RecursiveCharacterTextSplitter(
        # 每個 chunk 最多 450 個 Token
        chunk_size=450,
        # 相鄰 chunk 重疊 60 個 Token，保留跨 chunk 的上下文
        chunk_overlap=60,
    )
    # 對所有 Document 執行切割，回傳 chunk list
    chunks = splitter.split_documents(documents)

    print(
        # 輸出總 chunk 數
        f"已切成 {len(chunks)} 個 chunk "
        # 同時說明所使用的切割參數
        "（chunk_size=450, overlap=60）"
    )
    # 回傳 chunk list，供向量化步驟使用
    return chunks


# ── 將所有 chunk 向量化並存入 Milvus 向量資料庫────────────
def build_vector_store(chunks: list[Document]) -> None:
    # 提示使用者此步驟需要一些時間
    print("正在向量化 chunks 並寫入 Milvus，請稍候...")
    # 呼叫 LangChain 進行向量化 + 存入資料庫
    Milvus.from_documents(
        # 傳入要存入的 chunk list
        documents=chunks,
        # 指定使用 NVIDIA NIM Embedding Model 進行向量化
        embedding=NVIDIAEmbeddings(
            # 從環境變數取得 Embedding Model 名稱
            model=os.environ.get("EMBEDDING_MODEL"),
            # 從環境變數取得 NVIDIA API 金鑰
            api_key=os.environ.get("NVIDIA_NIM_API_KEY"),
        ),
        # 指定要存入的 Milvus collection 名稱
        collection_name="library_books",
        # Milvus 服務的連線位址
        connection_args={"uri": "http://localhost:19530"},
        # 每次執行前先刪除舊的 collection，確保資料是最新的
        drop_old=True,
        # 允許動態欄位，不需預先定義完整 schema 即可寫入 metadata
        enable_dynamic_field=True,
    )
    # 確認寫入完成
    print(f"已建立 library_books Collection 並存入 {len(chunks)} 個 chunks")


# ── 主程式進入點 ─────────────────────────
def main() -> None:
    # 顯示目前執行的 Phase 階段
    print("=== Phase 1：建立向量索引 ===")
    # 從 JSON 檔案讀取書籍原始資料
    records = load_data()
    # 將原始資料轉成 LangChain Document 格式
    documents = build_documents(records)
    # 將 Document 切成適合向量化的小 chunk
    chunks = split_documents(documents)
    # 向量化 chunks 並存入 Milvus 向量資料庫
    build_vector_store(chunks)
    # 完成提示
    print(f"成功建立索引，共處理 {len(records)} 本書，可至 http://localhost:8000 瀏覽")


# 確保此檔案被直接執行時才呼叫 main()，被 import 時不執行
if __name__ == "__main__":
    # 呼叫主流程
    main()
