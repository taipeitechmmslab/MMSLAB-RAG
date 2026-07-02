# Lab5 架構整理：tools/ package 重構

日期：2026-07-02

## 背景與動機

Lab5 是 Agentic RAG（自由式 ReAct）教學範例，目前只有 `vector_retrieve`、`graph_retrieve` 兩個 LangChain Tool，全部塞在 `retrieval.py` 一份檔案裡。

在重新檢視整個專案（`index.py`、`agent.py`、`generation.py`、`main.py`、`retrieval.py`）後，發現的是實際的重複與不一致問題，而不只是「檔案分類不夠美觀」：

1. **Neo4j 連線邏輯重複且行為不一致**：`retrieval.py` 的 `get_neo4j_driver()` 用 `os.environ.get(key, 預設值)`；`index.py` 的 `build_graph_index()` 另外重寫一份，卻用 `os.environ[key]`（沒有預設值）。同一件事兩處各寫一份，且沒設環境變數時一邊崩潰、一邊靜默套用預設帳密。
2. **Milvus／Embedding 連線設定重複**：`index.py` 的 `build_vector_store()` 與 `retrieval.py` 的 `vector_retrieve()` 各自重寫幾乎相同的 `NVIDIAEmbeddings` 建立邏輯，以及硬編碼的 `connection_args={"uri": "http://localhost:19530"}`、`collection_name="library_books"`。
3. **ChatNVIDIA 建立邏輯重複**：`agent.py` 的 `agent_node()` 與 `retrieval.py` 的 `generate_cypher()` 各自重新建立 `ChatNVIDIA(model=..., api_key=...)`，只差 `temperature` 參數。
4. `retrieval.py` 一份檔案同時扛了向量檢索 Infra、Neo4j Infra、Cypher 生成／驗證邏輯、Few-shot 範例、兩個 Tool 本體，對「示範如何新增一個 Tool」這個教學目的而言不夠直觀——沒有清楚的「一個 Tool 一個檔案」對應關係。

## 目標

- 消除上述 1–3 的重複程式碼與行為不一致，讓連線設定只有一個地方需要維護。
- 把 `retrieval.py` 拆成清楚的 `tools/` package，讓「新增一個 Tool」在教學上等於「新增一個檔案」。
- 不改變任何對外行為／輸出格式（`main.py` 顯示的內容、`run_agentic_rag()` 的回傳格式維持不變），純粹是內部結構整理。

## 非目標

- 不新增任何新的 Tool（例如借書、還書、web search 等），這次只處理既有兩個 Tool 的程式碼組織。
- 不改變 Agent 的決策邏輯（SYSTEM_PROMPT、ReAct 迴圈流程維持不動）。
- 不重構 `generation.py`、`main.py`——這兩個檔案職責已經單一清楚，且維持與 Lab 系列其他課次一致的檔案命名。

## 架構設計

### 目錄結構（變更後）

```
Lab5/
  clients.py            # 新增：集中管理外部資源（LLM／Embedding／Neo4j／Milvus）連線
  tools/
    __init__.py          # 新增：匯出 TOOLS list
    vector_tool.py        # 新增：從 retrieval.py 拆出的 vector_retrieve
    graph_tool.py          # 新增：從 retrieval.py 拆出的 graph_retrieve 與其私有 helper
  agent.py               # 修改：改用 clients.py、從 tools package import
  index.py               # 修改：改用 clients.py，移除重複的連線建立邏輯
  generation.py          # 不動
  main.py                # 不動
  requirements.txt        # 不動
  retrieval.py            # 刪除（內容已搬到 tools/ 與 clients.py）
```

### `clients.py`（新檔案）

集中定義所有外部資源的連線細節與預設值，`index.py`（建索引時）與 `tools/`（查詢時）都呼叫同一份，不再各自維護一份連線程式碼。

```python
NEO4J_URI = "bolt://localhost:7687"
MILVUS_URI = "http://localhost:19530"
MILVUS_COLLECTION = "library_books"

def get_llm(**kwargs) -> ChatNVIDIA:
    """建立 ChatNVIDIA，model/api_key 一律從環境變數取得，kwargs 轉傳給 ChatNVIDIA（例如 temperature）。"""

def get_embeddings() -> NVIDIAEmbeddings:
    """建立 NVIDIAEmbeddings，model/api_key 一律從環境變數取得。"""

def get_neo4j_driver() -> neo4j.Driver:
    """建立 Neo4j 連線，帳密皆用 os.environ.get(key, 預設值)：
    NEO4J_USERNAME 預設 "neo4j"，NEO4J_PASSWORD 預設 "graphrag"。
    index.py 與 tools/graph_tool.py 都呼叫這份，行為統一。"""

def get_vector_store() -> Milvus:
    """查詢用（給 tools/vector_tool.py）：
    Milvus(embedding_function=get_embeddings(), collection_name=MILVUS_COLLECTION,
           connection_args={"uri": MILVUS_URI}, enable_dynamic_field=True)"""
```

### `tools/vector_tool.py`（新檔案，取代 `retrieval.py` 的 `vector_retrieve` 部分）

只放 `vector_retrieve`（`@tool(response_format="content_and_artifact")`），邏輯與現行版本相同，唯一差異是內部改呼叫 `clients.get_vector_store()` 取代自行 `new Milvus(...)`。`format_vector_context` 維持從 `generation.py` import。

### `tools/graph_tool.py`（新檔案，取代 `retrieval.py` 的 `graph_retrieve` 部分）

放入與 `graph_retrieve` 相關的所有內容，維持「Tool 主體＋只服務它的私有 helper 同檔案」的既有模式：
- `GRAPH_SCHEMA`、`FEW_SHOT_EXAMPLES`、`FORBIDDEN_KEYWORDS`（模組層級常數）
- `generate_cypher()`、`validate_cypher()`、`run_cypher()`（私有 helper，邏輯不變，`generate_cypher` 內部改呼叫 `clients.get_llm(temperature=0.0)`，`validate_cypher`／`run_cypher` 改呼叫 `clients.get_neo4j_driver()`）
- `graph_retrieve`（`@tool(response_format="content_and_artifact")`，邏輯不變）

`format_graph_context` 維持從 `generation.py` import。

### `tools/__init__.py`（新檔案）

```python
from .vector_tool import vector_retrieve
from .graph_tool import graph_retrieve

TOOLS = [vector_retrieve, graph_retrieve]
```

### `agent.py`（修改）

- `from retrieval import graph_retrieve, vector_retrieve` → `from tools import TOOLS`（移除原本 `TOOLS = [vector_retrieve, graph_retrieve]` 這行，改直接使用 import 進來的 `TOOLS`）
- `agent_node()` 內的 `ChatNVIDIA(model=..., api_key=...)` → `clients.get_llm()`
- `SYSTEM_PROMPT`、`route_after_agent()`、`build_agent_graph()`、`run_agentic_rag()` 完全不動

### `index.py`（修改）

- `build_graph_index()` 內自建 `GraphDatabase.driver(...)` → 改呼叫 `clients.get_neo4j_driver()`
- `build_vector_store()` 內的 `NVIDIAEmbeddings(...)` 與硬編碼的 `connection_args`／`collection_name` → 改用 `clients.get_embeddings()`、`clients.MILVUS_URI`、`clients.MILVUS_COLLECTION`
- `Milvus.from_documents(...)` 呼叫方式維持不變（建索引是寫入操作，跟查詢用的 `clients.get_vector_store()` 是不同的呼叫方式，不能共用同一個 factory function）

## 行為變更（已與使用者確認）

`index.py` 目前在缺少 `NEO4J_USERNAME`／`NEO4J_PASSWORD` 環境變數時會直接 `KeyError` 崩潰。重構後統一改用 `clients.get_neo4j_driver()` 的寬鬆預設值（`neo4j` / `graphrag`），`index.py` 不會再因為沒設這兩個環境變數而崩潰，而是靜默套用預設帳密——這與現行 `retrieval.py` 的行為一致。此變更已經使用者確認接受。

## 資料流／對外行為

不變。`main.py` 呼叫 `run_agentic_rag()` 的方式、回傳的 `route`／`vector_docs`／`graph_result`／`steps`／`answer` 欄位格式，以及 CLI 顯示內容，皆維持與現行版本一致。這次重構純屬內部程式碼組織調整。

## 測試／驗證計畫

- 重構後執行 `python main.py`，跑過 `main.py` 內建的三個示範問題，確認：
  - 向量檢索問題（示範問題 3）能正常回傳書籍與相似度分數。
  - 圖譜檢索問題（示範問題 1、2）能正常生成 Cypher 並回傳查詢結果。
  - Agent 決策過程（`steps`）顯示內容與重構前一致。
- 確認 `python index.py` 仍可正常建立向量索引與知識圖譜（含統計輸出）。
- 檢查 `tools/`、`clients.py` 的 import 路徑正確，`retrieval.py` 刪除後沒有殘留的 import 參照。

## 範圍確認

此設計聚焦於單一子專案（Lab5）的內部程式碼重構，範圍單純、無需再拆解成多個子專案。
