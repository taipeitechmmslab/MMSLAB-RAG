"""
Lab0 - NVIDIA NIM 可用模型查詢工具
=================================
main.py 透過 langchain_nvidia_ai_endpoints 套件向 NVIDIA NIM 查詢目前實際可用的
Chat 模型（含是否支援 tool calling）與 Embedding 模型，讓讀者在開始 Lab1 之前，
用程式確認 .env 裡打算填入的 LLM_MODEL／EMBEDDING_MODEL 是否存在、是否符合需求，
不需要憑書上或文件裡的範例名稱盲猜（模型會隨時間下架或改變支援狀態）。

執行方式：
  python main.py
"""

# ── 載入套件與環境變數 ──────────────────────────────
import os
from dotenv import load_dotenv
from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings
load_dotenv()


# ── 查詢並印出目前可用的 Chat 模型，標示是否支援 tool calling ──────────
def list_chat_models(api_key: str) -> None:
    # 查詢 NVIDIA NIM 目前所有可用模型的中繼資料
    models = ChatNVIDIA.get_available_models(api_key=api_key)
    # 只保留 chat 類型（純文字對話／生成），並依模型 ID 排序方便閱讀
    chat_models = sorted(
        (m for m in models if m.model_type == "chat"),
        key=lambda m: m.id,
    )
    # 印出表頭與模型總數
    print(f"Chat 模型（共 {len(chat_models)} 個）：")
    print(f"{'模型 ID':<50} 支援 tool calling")
    # 逐一印出模型 ID 與是否支援 tool calling（agent.py 的 bind_tools() 需要這項支援）
    for m in chat_models:
        supports = "是" if m.supports_tools else "否"
        print(f"{m.id:<50} {supports}")
    print()


# ── 查詢並印出目前可用的 Embedding 模型 ──────────
def list_embedding_models(api_key: str) -> None:
    # 查詢並依模型 ID 排序，Embedding 模型沒有 tool calling 的概念，只列出 ID
    models = sorted(
        NVIDIAEmbeddings.get_available_models(api_key=api_key),
        key=lambda m: m.id,
    )
    print(f"Embedding 模型（共 {len(models)} 個）：")
    for m in models:
        print(f"  - {m.id}")
    print()


# ── 主程式進入點 ─────────────────────────
def main() -> None:
    # 從環境變數取得 NVIDIA NIM API 金鑰，未設定時提早結束並提示
    api_key = os.environ.get("NVIDIA_NIM_API_KEY")
    if not api_key:
        print("錯誤：請先在 .env 中設定 NVIDIA_NIM_API_KEY")
        return

    print("=== 查詢 NVIDIA NIM 目前可用的模型 ===\n")
    try:
        # 依序查詢並印出 Chat 與 Embedding 兩張模型清單
        list_chat_models(api_key)
        list_embedding_models(api_key)
    except Exception as e:
        # API Key 錯誤或網路問題時，印出清楚的錯誤訊息，避免讀者看到原始 Traceback
        print(f"查詢失敗，請確認 NVIDIA_NIM_API_KEY 是否正確：{e}")


# 確保此檔案被直接執行時才呼叫 main()，被 import 時不執行
if __name__ == "__main__":
    main()
