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


# ── 印出單次 skill 使用的判斷原因與讀到的 SKILL.md 內容 ──────────
# skill 是 Agent 自己讀取 SKILL.md 後套用的指示，沒有 args／artifact，改印出讀到的原始內容
def print_skill_use(skill_use: dict) -> None:
    print(f"── 第 {skill_use['index']} 次動作：使用 skill「{skill_use['skill']}」")
    if skill_use["reason"]:
        print(f"   判斷原因：{skill_use['reason']}")
    print("   讀取到的 SKILL.md 內容：")
    for line in skill_use["content"].splitlines():
        print(f"     {line}")
    print()


# ── 印出單次寫檔動作的路徑、內容與判斷原因 ──────────
# write_file 是 booklist-markdown-exporter 這個 skill 實際落地成檔案的那一步，沒有 args／artifact，
# 改印出這次寫入的路徑與內容
def print_file_write(file_write: dict) -> None:
    print(f"── 第 {file_write['index']} 次動作：寫入檔案「{file_write['file_path']}」")
    if file_write["reason"]:
        print(f"   判斷原因：{file_write['reason']}")
    if file_write["error"] is not None:
        print(f"   ⚠ 寫入失敗：{file_write['error']}")
        print()
        return
    print("   寫入內容：")
    for line in file_write["content"].splitlines():
        print(f"     {line}")
    print()


# ── 印出單次工具呼叫的參數、判斷原因與檢索結果 ──────────
# 讓使用者能照順序看懂 Agent 每一步在做什麼、為什麼做
def print_retrieval(retrieval: dict) -> None:
    # kind 為 skill 時，交給 print_skill_use 處理，不是工具呼叫沒有 args／artifact 可顯示
    if retrieval["kind"] == "skill":
        print_skill_use(retrieval)
        return

    # kind 為 file_write 時，交給 print_file_write 處理，不是工具呼叫沒有 args／artifact 可顯示
    if retrieval["kind"] == "file_write":
        print_file_write(retrieval)
        return

    # 印出這次檢索的編號、呼叫的工具與參數
    print(f"── 第 {retrieval['index']} 次檢索：{retrieval['tool']}（參數：{retrieval['args']}）")
    # reason 有值才代表推理模型記錄了判斷原因，一併印出
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
            # 取出這本書的 metadata 與相似度分數
            metadata = doc.get("metadata", {})
            score = float(doc["score"])
            # 依序印出書名、分類、作者、相似度分數
            print(f"     {i}. {metadata.get('book', '')}")
            print(f"        分類：{metadata.get('category', '')}")
            print(f"        作者：{metadata.get('authors', '')}")
            print(f"        相似度分數：{score:.4f}")

    elif retrieval["tool"] == "graph_retrieve":
        # graph_result 是 graph_retrieve 回傳的 artifact，包含 cypher／results／error／retries
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
    print("  1. 我想為生活帶來一些新的嘗試，去做過去沒有做過的事情。因此，我希望找到一本能引導我發現更多可能性的書，讓自己的生活變得更加豐富")
    print("  2. 可以幫我找跟這本書相同分類的3本書籍給我嗎?")
    print("  3. 請你把剛剛推薦的書籍，整理成一份文件給我，文件名稱叫做推薦書單")
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

        # 呼叫 agent.py 的 run_agentic_rag，逐步消費 generator：
        # 每一次工具呼叫／skill 套用一完成就立即印出，不用等整個 Agentic RAG 流程跑完
        print("AI Agent 執行中，請稍後...")
        for event in run_agentic_rag(query):
            if event["type"] == "retrieval":
                print_retrieval(event)
            elif event["type"] == "error":
                # LLM 呼叫本身失敗時（例如逾時、限流），印出原始錯誤訊息除錯，不再有後續事件
                print(f"Agent 執行失敗：{event['error']}")
                print()
                print("抱歉，目前系統暫時無法處理您的問題，請稍後再試。")
                print()
            elif event["type"] == "answer":
                print("AI 回答：")
                if event["answer"] is None:
                    # answer 為 None 代表這輪沒有拿到有效內容（空回覆，或工具呼叫語法混入回答文字）
                    print("抱歉，LLM 產生回答時發生意外，請再重新詢問問題或者更換 LLM。")
                else:
                    print(event["answer"])
                print()


# 確保此檔案被直接執行時才呼叫 main()，被 import 時不執行
if __name__ == "__main__":
    main()
