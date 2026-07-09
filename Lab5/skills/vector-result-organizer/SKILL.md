---
name: vector-result-organizer
description: 依相似度分數判斷書籍排序，並撰寫語意/情境角度的推薦理由。查完 vector_retrieve 且該結果會用於最終回答時使用。
---

# vector-result-organizer

## 適用時機
呼叫 `vector_retrieve` 取得原始書籍資料（含 metadata、相似度分數 score、命中段落 matched_page_content）之後，
在整理成給讀者看的推薦清單之前使用。
書名、作者、分類、價格、借閱狀態、推薦理由這六個共同欄位的名稱與呈現順序已定義在 system prompt，
本文件只補充向量證據特有的排序依據與推薦理由寫法，不重複宣告欄位。

## Instructions
1. 排序依據：依讀者問題與相似度分數（score）判斷每本書的優先順序，分數越小代表語意越相近，越應排在前面。
2. 推薦理由怎麼寫：從 `matched_page_content`（命中的段落原文）萃取這本書為何符合讀者「語意或情境」需求的重點，
   用一句精簡文字說明，例如「適合完全沒接觸過的初學者」「內容涵蓋在家自我練習的技巧」。
   理由要聚焦內容或情境上的相關性，事實面的符合條件（分類、借閱狀態是否吻合）不是這個 skill 的重點，
   那是 graph-result-organizer 的職責。
3. 若這次 `vector_retrieve` 沒有查到任何書籍，這代表向量證據這個來源沒有資料，不代表最終答案要回覆查無資料；
   是否改查 graph_retrieve 或直接判定查無資料，依 system prompt 的【資料不足時的處理】規則決定。
