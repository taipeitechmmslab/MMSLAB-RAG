"""
HyDE - Phase 1：建立索引
=================================
index.py 負責將圖書館書籍資料轉成可檢索的向量索引。

執行流程：
  0. 載入套件與環境變數
  1. 從 library_records.json 檔案讀取圖書館書籍資料
  2. 將每本書轉換成 Document 資料結構
  3. 使用 RecursiveCharacterTextSplitter 將 Documents 切成 chunks
  4. 使用 NVIDIA NIM Embedding Model 將 chunks 向量化，並透過 Milvus.from_documents 建立 library_books collection 存入向量資料庫

執行方式：
  python index.py
"""

# 載入套件與環境變數
import json
import os
from langchain_core.documents import Document
from langchain_milvus import Milvus
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
load_dotenv()

# 從 library_records.json 檔案讀取圖書館書籍資料
def load_data() -> list[dict]:
    with open("data/library_records.json", "r", encoding="utf-8") as f:
        records = json.load(f)
    print(f"已讀取 {len(records)} 筆書籍資料（資料來源：data/library_records.json）")
    return records

# 將每本書轉換成 Document 資料結構
def build_documents(records: list[dict]) -> list[Document]:
    documents = []
    for record in records:
        # borrower 與 timestamp 可能為 null，先轉成空 dict 以避免後續取值失敗
        borrower = record.get("borrower") or {}
        timestamp = borrower.get("timestamp") or {}
        metadata = {
            "book_id": record["_id"],
            "book": record["book"],
            "category": record["category"],
            "authors": ", ".join(record["authors"]),
            "price": float(record["price"]),
            "is_borrowed": bool(borrower),
            "borrower_name": borrower.get("name", ""),
            "borrowed_at": timestamp.get("$date", ""),
        }
        borrowed_text = "已借出" if borrower else "可借閱"
        borrower_name = borrower.get("name", "")
        searchable_text = (
            f"書名：{record['book']}\n"
            f"分類：{record['category']}\n"
            f"作者：{', '.join(record['authors'])}\n"
            f"借閱狀態：{borrowed_text}\n"
            f"借閱者：{borrower_name if borrower_name else '無'}\n"
            f"內容介紹：{record['description']}"
        )
        # page_content 是語意檢索時會被搜尋的資料
        # metadata 是不參與向量化的附帶資訊，檢索後可取用，供生成回答時引用
        documents.append(
            Document(
                page_content=searchable_text,
                metadata=metadata,
            )
        )
    print(f"已轉換成 {len(documents)} 份 Document")
    return documents

# 使用 RecursiveCharacterTextSplitter 將 Documents 切成 chunks
def split_documents(documents: list[Document]) -> list[Document]:
    # chunk_size 控制文件片段長度
    # chunk_overlap 讓相鄰 chunk 保留部分重疊內容，避免切分時遺失上下文
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=450,
        chunk_overlap=60,
    )
    chunks = splitter.split_documents(documents)

    print(
        f"已切成 {len(chunks)} 個 chunk "
        "（chunk_size=450, overlap=60）"
    )
    return chunks

# 使用 NVIDIA NIM Embedding Model 將 chunks 向量化，並透過 Milvus.from_documents 建立 library_books collection 存入向量資料庫
def build_vector_store(chunks: list[Document]) -> Milvus:
    print("正在向量化 chunks 並寫入 Milvus，請稍候...")
    # 建立與向量資料庫連線，並透過 NVIDIA NIM Embedding Model 將 chunks 向量化後存入向量資料庫
    vector_store = Milvus.from_documents(
        documents=chunks,
        # 初始化 NVIDIA NIM Embedding Model
        embedding=NVIDIAEmbeddings(
            model=os.environ.get("EMBEDDING_MODEL"),
            api_key=os.environ.get("NVIDIA_LLM_API_KEY"),
        ),
        collection_name="library_books",
        connection_args={"uri": "http://localhost:19530"},
        # 每次重建 collection，確保索引內容與目前 JSON 資料一致
        drop_old=True,
        # 允許 metadata 動態欄位寫入，省去預先定義 schema
        enable_dynamic_field=True,
    )
    print(f"已建立 library_books Collection 並存入 {len(chunks)} 個 chunks")
    return vector_store

# main 函數負責讀取資料 → 轉成 Document → 切成 chunks → 向量化並建立索引
def main() -> None:
    print("=== Phase 1：建立向量索引 ===")
    # Step 1：載入套件與環境變數
    records = load_data()
    # Step 2：從 library_records.json 檔案讀取圖書館書籍資料
    documents = build_documents(records)
    # Step 3：使用 RecursiveCharacterTextSplitter 將 Documents 切成 chunks
    chunks = split_documents(documents)
    # Step 4：使用 NVIDIA NIM Embedding Model 將 chunks 向量化，並透過 Milvus.from_documents 建立 library_books collection 存入向量資料庫
    build_vector_store(chunks)
    print(f"成功建立索引，共處理 {len(records)} 本書，可至 http://localhost:8000 瀏覽")


if __name__ == "__main__":
    main()
