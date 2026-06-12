"""
Vector RAG - 主程式入口
=================================
main.py 負責啟動互動式圖書館問答系統，使用 Phase 1 建好的索引，
並呼叫 Phase 2 與 Phase 3 完成檢索與回答生成：
  Phase 1（index.py）    → 事先建立索引
  Phase 2（retrieval.py）→ 執行向量檢索
  Phase 3（generation.py）→ 生成 AI 回答

執行方式：
  python main.py
"""

# ── 載入套件與環境變數 ──────────────────────────────
from dotenv import load_dotenv
from generation import generate
from retrieval import retrieve
load_dotenv()


# ── 啟動圖書館智慧問答系統 ──────────────────────────────────
def main() -> None:
    # 顯示系統啟動訊息
    print("=" * 55)
    print("      圖書館智慧問答系統")
    print("=" * 55)
    print()
    # 顯示示範問題，幫助使用者快速上手
    print("示範問題：")
    print("  1. 我想要找關於省錢出國旅行的書")
    print("  2. 有什麼書可以幫助我提升職場溝通能力？")
    print("  3. 目前被借出的商業管理類書籍有幾本？")
    print()

    # 進入互動式問答迴圈，持續等待使用者輸入問題
    while True:
        try:
            # 讀取使用者輸入並去除頭尾空白
            query = input("請輸入問題（輸入 'quit' 離開）：\n> ").strip()
        except (KeyboardInterrupt, EOFError):
            # 支援 Ctrl+C 或 EOF 中斷程式
            print("\n感謝使用，再見！")
            break

        # 使用者輸入 quit 時結束互動式問答流程
        if query.lower() == "quit":
            print("感謝使用，再見！")
            break

        # 跳過空白輸入，避免送出無效問題
        if not query:
            continue

        # 呼叫 retrieval.py 進行向量檢索，取回最相關的 5 本書
        print()
        print("正在檢索相關書籍...")
        docs = retrieve(query, top_k=5)

        # 顯示檢索到的書籍清單與相似度分數
        print()
        # 相似度分數越小，代表語意越相近
        print("檢索到的相關書籍，相似度分數越小，代表語意越相近：")
        # 用計數器 i 為每本書標上編號，從 1 開始
        for i, doc in enumerate(docs, start=1):
            # 從 doc 取出 metadata 與相似度分數
            metadata = doc.get("metadata", {})
            score = float(doc["score"])
            # 印出書名、分類與相似度分數
            print(f"  {i}. {metadata.get('book', '')}")
            print(f"     分類：{metadata.get('category', '')}")
            print(f"     相似度分數：{score:.4f}")

        # 呼叫 generation.py 根據檢索結果生成 AI 回答
        print()
        print("AI 回答：")
        answer = generate(query, docs)
        # 印出 AI 回答，並回到問答迴圈等待下一個問題
        print(f"{answer}")
        print()


# 確保此檔案被直接執行時才呼叫 main()，被 import 時不執行
if __name__ == "__main__":
    # 呼叫主流程
    main()
