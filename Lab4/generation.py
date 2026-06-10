"""
HyDE - Phase 3：生成回答
=================================
generation.py 負責將檢索到的書籍資料整理成 prompt，並呼叫 NVIDIA NIM LLM 生成回答。

此模組提供 generate() 函式供 main.py 呼叫。
"""

# ── 載入套件與環境變數 ──────────────────────────────
import os
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_nvidia_ai_endpoints import ChatNVIDIA
load_dotenv()

# ── 根據檢索到的書籍資料呼叫 LLM 生成回答 ────────────
def generate(query: str, retrieved_docs: list[dict]) -> str:
    # 將 retrieved_docs 格式化為可讀的書籍清單，作為 LLM 回答問題的 Context
    if not retrieved_docs:
        # 無檢索結果時給 LLM 一個明確提示，避免 LLM 憑空捏造
        context = "（查無相關書籍資料）"
    else:
        # lines 暫存每本書格式化後的文字，最後再合併成完整 Context
        lines = []
        # 用計數器 i 為每本書標上編號，從 1 開始
        for i, doc in enumerate(retrieved_docs, 1):
            # metadata 保存書名、分類、作者、價格與借閱狀態等書籍資訊
            metadata = doc.get("metadata", {})
            # 取出借閱者姓名，用於組合借閱狀態文字
            borrower_name = metadata.get("borrower_name", "")
            # 根據 is_borrowed 產生對應的借閱狀態文字
            if metadata.get("is_borrowed"):
                # 已借出時若有借閱者姓名則一併顯示
                borrowed_text = f"目前已借出（借閱者：{borrower_name}）" if borrower_name else "目前已借出"
            else:
                borrowed_text = "目前可借閱"
            # matched_page_content 是語意檢索時命中的相關段落，供 LLM 說明推薦理由
            matched_page_content = doc.get("matched_page_content", "")
            # 將單本書的欄位整理成固定格式，作為 LLM 回答時的參考資料
            lines.append(
                f"【書籍 {i}】{metadata.get('book', '')}\n"
                f"  類別：{metadata.get('category', '')}　作者：{metadata.get('authors', '')}　定價：{metadata.get('price', 0.0)} 元\n"
                f"  借閱狀態：{borrowed_text}\n"
                f"  相關段落：{matched_page_content}"
            )
        # 每本書之間以空行分隔，讓 LLM 能清楚區分不同書籍資料
        context = "\n\n".join(lines)

    # 初始化 NVIDIA NIM LLM
    llm = ChatNVIDIA(
        # 從環境變數取得 LLM 模型名稱
        model=os.environ.get("LLM_MODEL"),
        # 從環境變數取得 NVIDIA API 金鑰
        api_key=os.environ.get("NVIDIA_NIM_API_KEY"),
    )

    # System Prompt 放入回答規則，限制 LLM 只能根據提供的書籍資料回答
    system_prompt = (
        "你是一位圖書館智慧助理，專門根據館藏書籍資料回答讀者的問題。\n"
        "請嚴格依據以下提供的書籍資料來回答問題，不可憑空捏造書籍資訊。\n"
        "回答時必須針對每一本書，明確列出以下資訊：\n"
        "  1. 書名\n"
        "  2. 作者\n"
        "  3. 借閱狀態（是否可借閱；若已借出，請說明是被誰借走）\n"
        "  4. 根據「相關段落」說明為何推薦此書\n"
        "每本書獨立條列呈現，格式清晰，讓讀者一目瞭然。\n"
        "若提供的書籍中沒有符合問題的相關資料，請明確回覆「目前資料庫中無相關書籍」。\n"
        "請使用繁體中文回答，語氣親切、條理清晰。"
    )

    # Human Prompt 放入書籍 Context 與使用者問題
    human_prompt = (
        f"以下是從圖書館資料庫中檢索到的相關書籍段落：\n\n"
        f"{context}\n\n"
        f"讀者問題：{query}\n\n"
        f"請根據以上書籍資料回答讀者的問題。"
    )

    # 組合 System Prompt 與 Human Prompt，呼叫 LLM 生成有資料依據的回答
    response = llm.invoke([
        # System Prompt 設定 LLM 的角色與回答規則
        SystemMessage(content=system_prompt),
        # Human Prompt 傳入書籍 Context 與使用者問題
        HumanMessage(content=human_prompt),
    ])

    # 回傳 LLM 生成的回答內容
    return response.content
