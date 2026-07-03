"""
Agentic RAG - 主程式入口
=================================
main.py 負責啟動互動式圖書館問答系統，使用 Phase 1 建好的向量索引與知識圖譜，
並呼叫 agent.py 執行完整的 Agentic RAG 流程：
  Phase 1（index.py）  → 事先建立 Milvus 向量索引與 Neo4j 知識圖譜
  Agent（agent.py）    → 依問題性質動態決定檢索策略、驗證 Cypher、視結果品質補檢索、生成回答

執行方式：
  python main.py
"""

# ── 載入套件與環境變數 ──────────────────────────────
from dotenv import load_dotenv
from agent import run_agentic_rag
load_dotenv()


# ── 印出單次工具呼叫的參數、判斷原因與檢索結果 ──────────
# 讓使用者能照順序看懂 Agent 每一步在做什麼、為什麼做
def print_retrieval(retrieval: dict) -> None:
    print(f"── 第 {retrieval['index']} 次檢索：{retrieval['tool']}（參數：{retrieval['args']}）")
    if retrieval["reason"]:
        print(f"   判斷原因：{retrieval['reason']}")

    # 這次呼叫本身拋出未預期的例外（例如網路逾時），沒有 artifact 可顯示
    if retrieval["error"] is not None:
        print(f"   ⚠ 呼叫失敗：{retrieval['error']}")
        print()
        return

    if retrieval["tool"] == "hyde_query":
        # 顯示 HyDE 生成的假想文件，方便觀察後續 vector_retrieve 實際檢索用的 query
        print("   HyDE 生成的假想文件：")
        print(f"     {retrieval['artifact']}")

    elif retrieval["tool"] == "vector_retrieve":
        # 顯示這次向量檢索到的書籍清單與相似度分數
        print("   向量檢索結果（相似度分數越小，語意越相近）：")
        for i, doc in enumerate(retrieval["artifact"], start=1):
            metadata = doc.get("metadata", {})
            score = float(doc["score"])
            print(f"     {i}. {metadata.get('book', '')}")
            print(f"        分類：{metadata.get('category', '')}")
            print(f"        作者：{metadata.get('authors', '')}")
            print(f"        相似度分數：{score:.4f}")

    elif retrieval["tool"] == "graph_retrieve":
        graph_result = retrieval["artifact"]
        # 展示 LLM 生成的 Cypher，逐行加上邊框符號方便閱讀
        print("   LLM 生成的 Cypher 查詢：")
        for line in graph_result["cypher"].splitlines():
            print(f"     │ {line}")
        # retries > 0 代表過程中曾發生 Cypher 生成或執行失敗並重新生成
        if graph_result["retries"] > 0:
            print(f"   （Cypher 生成或執行失敗，已重新生成並重試 {graph_result['retries']} 次）")

        print("   知識圖譜查詢結果：")
        if graph_result["error"]:
            # Cypher 執行失敗時印出錯誤訊息
            print(f"     {graph_result['error']}")
        elif not graph_result["results"]:
            # 查詢成功但沒有任何資料命中
            print("     （查無相關資料）")
        else:
            for i, row in enumerate(graph_result["results"], 1):
                # 將每筆 row dict 轉成逗號分隔的可讀字串，略過值為 None 的欄位
                parts = [f"{k}={v}" for k, v in row.items() if v is not None]
                print(f"     {i}. " + "，".join(parts))
    print()


# ── 啟動圖書館智慧問答系統 ──────────────────────────────────
def main() -> None:
    # 顯示系統啟動訊息
    print("=" * 55)
    print("      圖書館智慧問答系統")
    print("=" * 55)
    print()
    # 顯示示範問題，幫助使用者快速上手
    print("示範問題：")
    print("  1. 我想找一本骨頭很硬、完全沒接觸過瑜珈的初學者在家自我練習的入門書。這本書的借閱者總共借了圖書館的哪幾本書？")
    print("  2. 我想找一本教你看懂財報、了解企業獲利與風險的書。這本書的作者們，還合寫過哪些其他的商業管理類書籍？")
    print("  3. 吳宗翰目前借閱中的語言學習類書籍有哪些？這些書的作者裡，還有誰是「語言學習」類其他書籍的唯一作者？")
    print("  4. 王小明借閱的書裡，屬於語言學習類的是哪一本？如果他想接著挑戰更進階、以商務情境為主的日語教材，你會推薦哪一本？")
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

        # 呼叫 agent.py 的 run_agentic_rag 執行完整的 Agentic RAG 流程
        result = run_agentic_rag(query)

        # 若 LLM 呼叫本身失敗（例如逾時、限流），沒有檢索記錄可顯示，直接印出原始錯誤訊息除錯
        if result.get("error"):
            print(f"Agent 執行失敗：{result['error']}")
            print()

        # 依實際執行順序，逐筆印出每一次工具呼叫的參數、判斷原因與檢索結果
        print("Agent 決策過程：")
        for retrieval in result["retrievals"]:
            print_retrieval(retrieval)

        # 顯示 agent.py 整合檢索結果後生成的 AI 回答
        print("AI 回答：")
        print(result["answer"])
        print()


# 確保此檔案被直接執行時才呼叫 main()，被 import 時不執行
if __name__ == "__main__":
    main()
