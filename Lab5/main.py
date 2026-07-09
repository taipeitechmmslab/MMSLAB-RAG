"""
Agentic RAG - 主程式入口
=================================
這個檔案負責啟動圖書館問答系統。
使用者輸入問題後，main.py 會交給 agent.py 處理，並把每一步結果印出來。

執行方式：
  python main.py
"""

# ── 載入套件與環境變數 ──────────────────────────────
from dotenv import load_dotenv
from agent import run_agentic_rag
load_dotenv()

# 工具名稱對應的中文說明，用於檢索標題行
TOOL_LABELS = {
    "hyde_query": "HyDE 假想文件工具",
    "vector_retrieve": "向量檢索工具",
    "graph_retrieve": "知識圖譜檢索工具",
}

# skill 名稱對應的中文說明，用於動作標題行
SKILL_LABELS = {
    "booklist-markdown-exporter": "儲存推薦書單",
}


# ── 啟動圖書館智慧問答系統 ──────────────────────────────────
def main() -> None:
    # 顯示系統名稱
    print("=" * 55)
    print("      圖書館智慧問答系統")
    print("=" * 55)
    print()
    # 顯示幾個可以直接測試的問題
    print("示範問題：")
    print("  1. 我想為生活帶來一些新的嘗試，去做過去沒有做過的事情。因此，我希望找到一本能引導我發現更多可能性的書，讓自己的生活變得更加豐富")
    print("  2. 可以幫我找跟這本書相同分類的3本書籍給我嗎?")
    print("  3. 請你把剛剛推薦的書籍，整理成一份文件給我，文件名稱叫做推薦書單")
    print()

    # 持續等待使用者輸入問題
    while True:
        try:
            # 去掉輸入前後多餘的空白
            query = input("請輸入問題（輸入 'quit' 離開）：\n> ").strip()
        except (KeyboardInterrupt, EOFError):
            # 使用者按 Ctrl+C 或結束輸入時，正常離開程式
            print("\n感謝使用，再見！")
            break

        # 輸入 quit 時離開程式
        if query.lower() == "quit":
            print("感謝使用，再見！")
            break

        # 空白問題不送給 Agent
        if not query:
            continue

        print()

        # Agent 會邊執行邊回傳事件，主程式收到後就立刻印出
        print("AI Agent 執行中，請稍後...")
        for event in run_agentic_rag(query):
            if event["type"] == "error":
                # LLM 逾時或限流時，顯示錯誤並結束這一輪
                print(f"Agent 執行失敗：{event['error']}")
                print()
                print("抱歉，目前系統暫時無法處理您的問題，請稍後再試。")
                print()
                continue

            if event["type"] == "answer":
                print("AI 回答：")
                if event["answer"] is None:
                    # 沒拿到有效回答時，請使用者重問或換模型
                    print("抱歉，LLM 產生回答時發生意外，請再重新詢問問題或者更換 LLM。")
                else:
                    print(event["answer"])
                print()
                continue

            if event["kind"] == "skill":
                # skill 事件沒有工具參數和檢索結果，只顯示使用原因與 SKILL.md 內容
                print(f"── 第 {event['index']} 次動作：調用{SKILL_LABELS.get(event['skill'], event['skill'])}")
                if event["reason"]:
                    print(f"   判斷原因：{event['reason']}")
                print("   讀取到的 SKILL.md 內容：")
                for line in event["content"].splitlines():
                    print(f"     {line}")
                print()
                continue

            if event["kind"] == "file_write":
                # 寫檔事件沒有檢索結果，只顯示檔案路徑、內容與錯誤訊息
                print(f"── 第 {event['index']} 次動作：寫入檔案「{event['file_path']}」")
                if event["reason"]:
                    print(f"   判斷原因：{event['reason']}")
                if event["error"] is not None:
                    print(f"   ⚠ 寫入失敗：{event['error']}")
                    print()
                    continue
                print("   寫入內容：")
                for line in event["content"].splitlines():
                    print(f"     {line}")
                print()
                continue

            print(f"── 第 {event['index']} 次檢索：調用{TOOL_LABELS.get(event['tool'], '')}{event['tool']}")
            if event["reason"]:
                print(f"   判斷原因：{event['reason']}")

            if event["error"] is not None:
                print(f"   ⚠ 呼叫失敗：{event['error']}")
                print()
                continue

            if event["tool"] == "hyde_query":
                # HyDE 會先產生一段假想文件，再拿去做向量檢索
                print("   HyDE 生成的假想文件：")
                print(f"     {event['artifact']}")

            elif event["tool"] == "vector_retrieve":
                print("   向量檢索結果（相似度分數越小，語意越相近）：")
                for i, doc in enumerate(event["artifact"], start=1):
                    metadata = doc.get("metadata", {})
                    score = float(doc["score"])
                    print(f"     {i}. {metadata.get('book', '')}")
                    print(f"        分類：{metadata.get('category', '')}")
                    print(f"        作者：{metadata.get('authors', '')}")
                    print(f"        相似度分數：{score:.4f}")

            elif event["tool"] == "graph_retrieve":
                graph_result = event["artifact"]
                print("   LLM 生成的 Cypher 查詢：")
                for line in graph_result["cypher"].splitlines():
                    print(f"     │ {line}")
                if graph_result["retries"] > 0:
                    print(f"   （Cypher 生成或執行失敗，已重新生成並重試 {graph_result['retries']} 次）")

                print("   知識圖譜查詢結果：")
                if graph_result["error"]:
                    print(f"     {graph_result['error']}")
                elif not graph_result["results"]:
                    print("     （查無相關資料）")
                else:
                    for i, row in enumerate(graph_result["results"], 1):
                        parts = [f"{k}={v}" for k, v in row.items() if v is not None]
                        print(f"     {i}. " + "，".join(parts))
            print()


# 只有直接執行 main.py 時才啟動程式
if __name__ == "__main__":
    main()
