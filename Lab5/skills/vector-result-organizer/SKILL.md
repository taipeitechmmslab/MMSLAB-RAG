---
name: vector-result-organizer
description: 整理 vector_retrieve 撈回的原始書籍資料，依讀者問題判斷優先順序並寫推薦理由。查完 vector_retrieve 且要整理向量檢索結果時使用。
---

# vector-result-organizer

## 適用時機
呼叫 `vector_retrieve` 取得原始書籍資料（含 metadata、相似度分數、命中段落）之後，
在整理成給讀者看的推薦清單之前使用。

## Instructions
1. 依讀者問題與相似度分數，判斷每本書的優先順序（相似度分數越小代表語意越相近）。
2. 針對每本書寫一句精簡的推薦理由，說明為什麼符合讀者的需求。
3. 只能根據 `vector_retrieve` 實際回傳的資料整理，不可捏造書名、作者、分類、價格或借閱狀態。
4. 輸出格式：依優先順序列出書名、類別、作者、定價、借閱狀態、推薦理由。
