"""
Hybrid RAG - 主程式入口
=================================
main.py 負責啟動互動式圖書館問答系統，使用 Phase 1 建好的混合索引，
並呼叫 Phase 2 與 Phase 3 完成檢索與回答生成：
  Phase 1（index.py）     → 事先建立 Milvus 向量索引與 Neo4j 知識圖譜
  Phase 2（retrieval.py） → 同時執行向量檢索與知識圖譜檢索
  Phase 3（generation.py）→ 生成 AI 整合回答

執行方式：
  python main.py
"""

# ── 載入套件與環境變數 ──────────────────────────────
from dotenv import load_dotenv
from generation import generate
from retrieval import graph_retrieve, vector_retrieve
load_dotenv()


# ── 啟動圖書館智慧問答系統 ──────────────────────────────────
def main() -> None:
    # 顯示系統啟動訊息
    print("=" * 55)
    print("      圖書館智慧問答系統 - Hybrid RAG")
    print("=" * 55)
    print()
    # 顯示示範問題，幫助使用者快速上手
    print("示範問題：")
    print("  1. 我想找省錢出國旅遊的書")
    print("  2. 目前被借出的商業管理類書籍有幾本？")
    print("  3. Mary 和 Jason 共同寫過哪些書？")
    print("  4. 我想第一次自助旅行，最好是有交通卡、城市移動和行程安排建議的書，適合看哪一本？")
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

        print()

        # 呼叫 retrieval.py 的 vector_retrieve 進行向量檢索，取回 Top-K 為 5 的最相關書籍
        print("正在執行向量檢索...")
        vector_docs = vector_retrieve(query, top_k=5)

        # 顯示向量檢索到的書籍清單與相似距離分數
        print("\n向量檢索結果（相似距離分數越小，語意越相近）：")
        # 用計數器 i 為每本書標上編號，從 1 開始
        for i, doc in enumerate(vector_docs, start=1):
            # 從 doc 取出 metadata 與相似距離分數
            metadata = doc.get("metadata", {})
            score = float(doc["score"])
            # 印出書名、分類、作者與相似距離分數
            print(f"  {i}. {metadata.get('book', '')}")
            print(f"     分類：{metadata.get('category', '')}")
            print(f"     作者：{metadata.get('authors', '')}")
            print(f"     相似距離分數：{score:.4f}")

        print()

        # 呼叫 retrieval.py 的 graph_retrieve 進行知識圖譜檢索，由 LLM 生成 Cypher 後查詢 Neo4j
        print("正在生成 Cypher 並查詢知識圖譜...")
        graph_result = graph_retrieve(query)

        # 展示 LLM 生成的 Cypher，逐行加上邊框符號方便閱讀
        print("\nLLM 生成的 Cypher 查詢：")
        for line in graph_result["cypher"].splitlines():
            print(f"  │ {line}")

        # 顯示知識圖譜查詢結果
        print("\n知識圖譜查詢結果：")
        if graph_result["error"]:
            # Cypher 執行失敗時印出錯誤訊息
            print(f"  {graph_result['error']}")
        elif not graph_result["results"]:
            # 查詢成功但沒有任何資料命中
            print("  （查無相關資料）")
        else:
            # 用計數器 i 為每筆結果標上編號，從 1 開始
            for i, row in enumerate(graph_result["results"], 1):
                # 將每筆 row dict 轉成逗號分隔的可讀字串，略過值為 None 的欄位
                parts = [f"{k}={v}" for k, v in row.items() if v is not None]
                print(f"  {i}. " + "，".join(parts))

        print()

        # 呼叫 generation.py 整合向量檢索與知識圖譜結果生成 AI 回答
        print("===== Hybrid RAG 的回答 =====")
        answer = generate(query, vector_docs, graph_result)
        # 印出 AI 回答，並回到問答迴圈等待下一個問題
        print(answer)
        print()


# 確保此檔案被直接執行時才呼叫 main()，被 import 時不執行
if __name__ == "__main__":
    main()
