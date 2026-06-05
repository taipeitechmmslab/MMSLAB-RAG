"""
HyDE - 主程式入口
=================================
main.py 負責啟動互動式圖書館問答系統，使用 Phase 1 建好的索引，
並呼叫 Phase 2 與 Phase 3 完成檢索與回答生成：
  Phase 1（index.py）    → 事先建立索引
  Phase 2（retrieval.py）→ 執行語意檢索
  Phase 3（generation.py）→ 生成 AI 回答

執行方式：
  python main.py
"""

# 載入套件與環境變數
from dotenv import load_dotenv
from generation import generate
from retrieval import retrieve
load_dotenv()


# main 函數負責互動式問答迴圈：讀取問題 → 語意檢索 → 生成回答 → 顯示結果
def main() -> None:
    # 顯示系統啟動訊息與示範問題
    print("=" * 55)
    print("      圖書館智慧問答系統")
    print("=" * 55)
    print()
    print("示範問題：")
    print("  1. 我明明睡很久，白天還是很累")
    print("  2. 我寫的東西在我電腦能跑，上線就一直出狀況")
    print()

    # 進入互動式問答迴圈，等待使用者輸入問題
    while True:
        try:
            query = input("請輸入問題（輸入 'quit' 離開）：\n> ").strip()
        except (KeyboardInterrupt, EOFError):
            # 支援 Ctrl+C 或 EOF 中斷程式
            print("\n感謝使用，再見！")
            break

        # 處理離開指令、程式中斷與空白輸入
        # 使用者輸入 quit 時結束互動式問答流程
        if query.lower() == "quit":
            print("感謝使用，再見！")
            break

        # 跳過空白輸入，避免送出無效問題
        if not query:
            continue

        # 呼叫 retrieval.py 進行語意檢索
        print()
        print("正在檢索相關書籍...")
        docs = retrieve(query, top_k=5)

        # 顯示檢索到的相關書籍與相似度分數
        # 相似度分數越小，代表語意越相近
        print()
        print("檢索到的相關書籍，相似度分數越小，代表語意越相近：")
        for i, doc in enumerate(docs, start=1):
            metadata = doc.get("metadata", {})
            score = float(doc["score"])
            print(f"  {i}. {metadata.get('book', '')}")
            print(f"     分類：{metadata.get('category', '')}")
            print(f"     相似度分數：{score:.4f}")

        # 呼叫 generation.py 根據檢索結果生成回答
        print()
        print("AI 回答：")
        answer = generate(query, docs)
        # 印出 AI 回答，並回到問答迴圈等待下一個問題
        print(f"{answer}")
        print()


if __name__ == "__main__":
    main()