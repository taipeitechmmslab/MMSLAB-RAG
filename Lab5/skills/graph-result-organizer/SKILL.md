---
name: graph-result-organizer
description: 把 graph_retrieve 撈回的原始 Cypher 查詢結果，轉譯成自然語言摘要。查完 graph_retrieve 且要整理知識圖譜結果時使用。
---

# graph-result-organizer

## 適用時機
呼叫 `graph_retrieve` 取得知識圖譜查詢結果（Cypher 與結構化資料列）之後，
在整理成給讀者看的自然語言摘要之前使用。

## Instructions
1. 把結構化的查詢結果（book、author、category、borrower、price 等欄位）轉譯成通順的中文敘述。
2. 只能根據查詢結果實際包含的欄位整理，不可捏造書名、作者或借閱資訊。
3. 若查詢結果顯示 Cypher 執行失敗，直接說明「知識圖譜查詢失敗」與失敗原因，不要嘗試整理不存在的資料。
4. 若查詢成功但沒有任何資料列，直接說明「查無相關資料」。
