# 授權、合規與原創性聲明

本文件說明 MacauSplat 與上游專案的關係、必須遵守的授權條件，以及（供比賽／發表使用）
本作品的**原創貢獻邊界**。

---

## 一、與上游的關係

| 項目 | 內容 |
|---|---|
| 上游專案 | [OOOSplat](https://github.com/ooolabdev/ooosplat)，作者 ooolabdev |
| 上游授權 | Apache License 2.0 |
| 本專案名稱 | **MacauSplat**（未使用上游名稱、標誌、圖示或任何品牌元素） |
| 取得方式 | Fork 後在其框架上開發；git 歷史保留上游提交，完整可追溯 |
| 修改內容 | 逐檔列於 [`NOTICE`](../NOTICE) 的 “Statement of changes” 一節 |

**我們沒有做的事**：沒有聲稱發明 3D 高斯潑濺、COLMAP、Brush 或影像式重建流程；
沒有移除或修改上游的授權與商標文件；沒有使用上游的品牌元素。

## 二、Apache-2.0 的義務（本專案如何履行）

| 條款 | 要求 | 本專案的履行方式 |
|---|---|---|
| §4(a) | 附上授權全文 | 保留 [`LICENSE`](../LICENSE) |
| §4(b) | 標明曾修改的檔案 | [`NOTICE`](../NOTICE) 逐檔列出新增與修改 |
| §4(c) | 保留來源的著作權、專利、商標與免責聲明 | 保留上游 `NOTICE` 的著作權行與 `TRADEMARK_POLICY.md` |
| §4(d) | 若上游有 NOTICE 檔，衍生作品須一併附上 | 本專案的 `NOTICE` 內含上游聲明 |
| §6 | 商標權不隨授權授予 | 產品名改為 MacauSplat；`identifier` 欄位保留原值僅為相容既有設定，非品牌使用 |

第三方程式（FFmpeg／LGPL-2.1、COLMAP／BSD-3-Clause、Brush／Apache-2.0、PlayCanvas、Mediabunny）
的授權與版本見 [`licenses/THIRD_PARTY_NOTICES.txt`](../licenses/THIRD_PARTY_NOTICES.txt) 與
[`engines/manifest.json`](../engines/manifest.json)。**這些元件的授權不因與本專案一同散布而改變。**

> 注意：`final.ply` 等**生成結果**不會僅因使用本工具而自動適用 Apache-2.0；
> 模型的權利歸屬取決於輸入素材與相關法律，詳見上游 [`GENERATED_OUTPUTS.md`](../GENERATED_OUTPUTS.md)。

## 三、原創貢獻邊界（比賽／發表時請照這樣寫）

> 三維重建使用開源工具 OOOSplat（Apache-2.0）及其內建的 COLMAP 與 Brush；
> 本作品的貢獻在於**重建之後的自動標註階段**與其應用，具體包含：
>
> 1. **COLMAP 稀疏重建的解析與相機投影**（自行實作，含 11 種相機模型與畸變處理）；
> 2. **由相機軌跡推定上軸**，解決 COLMAP 世界座標無重力方向的問題；
> 3. **由稀疏點雲擬合建築三維包圍盒**，並投影為逐影格的二維標註；
> 4. **以 COLMAP track 推導的遮擋代理指標**，用既有資料判斷影格品質；
> 5. **可稽核的資料集輸出**（YOLO 標籤、逐影格明細、免依賴的 HTML 複核表）。

這種寫法的好處是**誠實且可驗證**：評審能清楚區分「用了什麼工具」與「自己做了什麼」。
使用開源工具本身不會扣分；說不清自己做了什麼才會。

## 四、AI 工具使用聲明（建議一併揭露）

本專案在開發過程中使用了 AI 輔助工具（含大型語言模型）協助：程式骨架、單元測試擬稿、
文件整理與除錯。**所有設計決定、實測數據與最終程式碼均經作者審查與執行驗證**：
每個模組都有單元測試，且以真實專案資料端到端驗證（見 `docs/DESIGN.md` 第六節）。

若比賽要求填寫 AI 使用申報表，建議如實列出上述範圍。ISEF 已發布生成式 AI 使用指引
（[International Rules 頁面](https://www.societyforscience.org/isef/international-rules/) 所列的
Generative AI Use Table），申報「用於輔助寫程式與文件、核心設計與驗證由作者完成」即可。

## 五、資料與素材的授權

本專案**不包含**任何建築影像。使用時請注意：

- 若以自行拍攝的影片作為輸入，素材權利屬於拍攝者；
- 若使用 Wikimedia Commons 等外部影像，需遵守其個別授權並保留出處
  （本團隊另一支管線 `macaumemory/tools/yolo-dataset` 會自動產生 `ATTRIBUTION.csv`）；
- 含有可辨識人臉的影像建議先做模糊處理（上游與本團隊管線皆有隱私處理元件）。

## 六、聯絡與致謝

- 上游專案：<https://github.com/ooolabdev/ooosplat>（Apache-2.0，作者 ooolabdev）
- 內建引擎：COLMAP（BSD-3-Clause）、Brush（Apache-2.0）、FFmpeg（LGPL-2.1-or-later）

謹此致謝上游作者與上述開源專案的貢獻者。
