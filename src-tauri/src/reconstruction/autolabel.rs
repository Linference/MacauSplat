//! 自動標註：把三維包圍盒投影成每一張影格的二維框。
//!
//! 這是本專案相對於既有重建工具（OOOSplat／Brush／COLMAP）新增的環節。
//!
//! ## 為什麼這樣做
//! 手動框一張圖要數十秒，一棟建築環繞一圈有數百張影格；用開放詞彙偵測器自動框得到的是弱標註，
//! 同一個建築面在不同影格上框得鬆緊不一。但只要在三維空間定義一次包圍盒，就能投影到每一張影格，
//! 得到**幾何一致、可稽核、零漏框**的二維標註。
//!
//! ## 座標系的難題（本模組最關鍵的設計）
//! COLMAP 的世界座標由第一台相機定義，**沒有重力方向**，所以「建築的高度」不能假設沿 Y 軸。
//! 本模組的解法是**由相機軌跡推斷上軸**：拍攝建築時操作者通常繞著建築走，相機中心大致落在
//! 一個水平面上，因此對相機中心做 PCA、取變異最小的方向即為上軸。
//! 若軌跡退化（例如原地旋轉、只有兩三個位置、或相機中心在三維上散開），
//! 則退回世界 Y 軸並在 `Box3D::up_source` 標明來源（見 `UpAxisSource`）。
//!
//! ## 標註品質的自我檢查
//! 每一張影格都會計算：
//!   - 可見角點數（8 個角有幾個在相機前方且落在畫面內）
//!   - 投影面積佔畫面比例（太大或太小都不適合當訓練樣本）
//!   - **該影格實際觀測到的三維點數**（用 COLMAP 的 track 資料：有多少屬於這個包圍盒的點，
//!     其 track 包含這張影格）——這是遮擋與視角的客觀代理指標
//!
//! 另外，若相機模型不是本模組完整實作的針孔＋徑向畸變（例如魚眼），
//! 被採用的影格仍會帶上 `FrameAnnotation::warning`：投影只是近似，這件事不能默默帶過。
//!
//! ## 寫檔的兩個約定
//!   - **標籤目錄會被整個換掉**（`write_yolo_labels` 先寫 `labels.tmp/` 再整目錄更名）：
//!     YOLO 的慣例是「沒有標籤檔 = 背景」，因此舊的 `.txt` 若留著，重跑時被略過的影格
//!     就會變成憑空多出的正樣本。被清掉的數量會回報出來（`LabelWriteReport::removed`）。
//!   - **所有插值都經過轉義**（HTML 與 CSV 各一個小工具）：COLMAP 記錄的檔名來自使用者磁碟，
//!     Windows 上檔名可以合法包含 `,`、`"`、`&`，直接插進樣板會讓整份複核表或 CSV 表格壞掉。

use std::path::Path;

use serde::{Deserialize, Serialize};

use super::colmap::{camera_to_pixel, Camera, Image, SparseReconstruction};

/// 上軸（`Box3D::axes[2]`）的來源。
///
/// 為什麼要三種而不是一個布林：舊設計只有 `up_axis_estimated: bool`，
/// 而 `true` 其實代表「退回世界 Y 軸」——欄位名講的是「推定值」，
/// 與它真正的意思（**沒有推定成功**）剛好相反，介面文案因此也照著錯的意思寫。
/// 這裡把三種情況分開，誰都不能再靠猜：
///   - `CameraTrajectory`：真的由相機軌跡推出來了；
///   - `WorldY`：軌跡退化，只能退回世界 Y 軸（等同「高度可能不準」）；
///   - `Manual`：操作者自己指定（`infer_up = false` 或人工給的軸對齊盒）。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum UpAxisSource {
    /// 由相機軌跡推定（PCA 最小主軸）。
    CameraTrajectory,
    /// 軌跡退化，退回世界 +Y 軸；**這是可能不準的那一種**。
    #[default]
    WorldY,
    /// 由操作者明確指定。
    Manual,
}

impl UpAxisSource {
    /// 是否屬於「靠推定、可能不準」的來源（目前只有退回世界 Y 軸的情況）。
    ///
    /// 為什麼預設值是 `WorldY`：反序列化舊的 `box.json`（沒有這個欄位）時，
    /// 唯一的資訊是「不知道上軸怎麼來的」；此時選擇「可能不準」是保守的一方，
    /// 寧可多顯示一次警示，也不要讓使用者以為軸向一定可靠。
    pub fn is_estimated(self) -> bool {
        matches!(self, Self::WorldY)
    }

    /// 給人看的中文說明（QA 表與命令列輸出共用）。
    pub fn label(self) -> &'static str {
        match self {
            Self::CameraTrajectory => "由相機軌跡推定",
            Self::WorldY => "推定失敗，退回世界 Y 軸（高度可能不準）",
            Self::Manual => "操作者指定",
        }
    }
}

/// 三維有向包圍盒。
///
/// `center` 為世界座標中心；`size` 為三個軸向的長度；`axes` 為三個單位向量（右手系），
/// 分別對應 size 的三個分量方向。角點 = center ± axes[i] * size[i] / 2。
#[derive(Debug, Clone, Serialize)]
pub struct Box3D {
    pub center: [f64; 3],
    pub size: [f64; 3],
    /// 三個正交單位軸：`axes[0]` 為長、`axes[1]` 為寬、`axes[2]` 為上（垂直方向）。
    pub axes: [[f64; 3]; 3],
    /// 上軸的來源。
    #[serde(default)]
    pub up_source: UpAxisSource,
    /// 擬合時使用的點數。
    #[serde(default)]
    pub fitted_from_points: usize,
}

impl Box3D {
    /// 軸對齊盒（快速手動指定用）。
    ///
    /// 這裡的軸向是世界座標軸，也就是「操作者指定」而不是任何推定結果，
    /// 因此來源是 `Manual`——舊版寫死的 `up_axis_estimated: true` 是錯的語意。
    pub fn axis_aligned(center: [f64; 3], size: [f64; 3]) -> Self {
        Self {
            center,
            size,
            axes: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            up_source: UpAxisSource::Manual,
            fitted_from_points: 0,
        }
    }

    /// 8 個角點的世界座標。
    pub fn corners(&self) -> [[f64; 3]; 8] {
        let mut out = [[0.0; 3]; 8];
        let mut i = 0;
        for sx in [-0.5, 0.5] {
            for sy in [-0.5, 0.5] {
                for sz in [-0.5, 0.5] {
                    let p = std::array::from_fn(|k| {
                        self.center[k]
                            + self.axes[0][k] * self.size[0] * sx
                            + self.axes[1][k] * self.size[1] * sy
                            + self.axes[2][k] * self.size[2] * sz
                    });
                    out[i] = p;
                    i += 1;
                }
            }
        }
        out
    }

    /// 點是否落在盒內。
    pub fn contains(&self, p: [f64; 3]) -> bool {
        for axis in 0..3 {
            let d: f64 = p
                .iter()
                .zip(self.center.iter())
                .zip(self.axes[axis].iter())
                .map(|((pk, ck), ak)| (pk - ck) * ak)
                .sum();
            if d.abs() > self.size[axis] / 2.0 {
                return false;
            }
        }
        true
    }
}

/// 舊版 `box.json` 沒有 `upSource`，只有 `up_axis_estimated: bool`。
///
/// 這裡用一個影子結構同時接受兩種格式，讓舊的 `box.json` 仍可被 `annotate --box-path` 讀取：
/// `true`（舊語意：退回世界 Y 軸）→ `WorldY`；`false`（舊語意：由軌跡推定）→ `CameraTrajectory`。
impl<'de> Deserialize<'de> for Box3D {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct Shadow {
            center: [f64; 3],
            size: [f64; 3],
            axes: [[f64; 3]; 3],
            #[serde(default)]
            up_source: Option<UpAxisSource>,
            #[serde(default)]
            up_axis_estimated: Option<bool>,
            #[serde(default)]
            fitted_from_points: usize,
        }
        let shadow = Shadow::deserialize(deserializer)?;
        let up_source = shadow.up_source.unwrap_or(match shadow.up_axis_estimated {
            Some(true) => UpAxisSource::WorldY,
            Some(false) => UpAxisSource::CameraTrajectory,
            None => UpAxisSource::WorldY,
        });
        Ok(Box3D {
            center: shadow.center,
            size: shadow.size,
            axes: shadow.axes,
            up_source,
            fitted_from_points: shadow.fitted_from_points,
        })
    }
}

/// 擬合參數。
#[derive(Debug, Clone)]
pub struct FitOptions {
    /// 百分位修剪（0.02 表示各軸取 2%~98% 範圍，用來排除離群的稀疏點）。
    pub trim: f64,
    /// 是否由相機軌跡推定上軸（false 則用世界 Y 軸）。
    pub infer_up: bool,
    /// 各軸最小尺寸（公尺），避免退化成一條線。
    pub min_extent: f64,
}

impl Default for FitOptions {
    fn default() -> Self {
        Self {
            trim: 0.02,
            infer_up: true,
            min_extent: 0.2,
        }
    }
}

/// 上軸推定的結果。
#[derive(Debug, Clone, PartialEq)]
pub struct UpAxisEstimate {
    /// 上軸單位向量。
    pub up: [f64; 3],
    /// 來源（只可能是 `CameraTrajectory` 或 `WorldY`：`Manual` 由呼叫端決定）。
    pub source: UpAxisSource,
    /// 軌跡有多「平」：最小主軸變異量 ÷ 次小主軸變異量。
    ///
    /// 0 表示相機中心完全落在一個平面上（理想）；越大代表軌跡在三維上散開，
    /// 最小主軸就越不可能是重力方向。超過 `MAX_PLANARITY_RATIO` 即判定退化。
    pub planarity_ratio: f64,
    /// 判定退化而退回世界 Y 軸的原因（`source == WorldY` 時必定有值）。
    pub degenerate_reason: Option<String>,
}

impl UpAxisEstimate {
    /// 退化（無法由軌跡推定）時退回世界 +Y 軸。
    fn fallback(reason: impl Into<String>, planarity_ratio: f64) -> Self {
        Self {
            up: [0.0, 1.0, 0.0],
            source: UpAxisSource::WorldY,
            planarity_ratio,
            degenerate_reason: Some(reason.into()),
        }
    }
}

/// 判定上軸是否可靠的上限：最小主軸變異量 ÷ 次小主軸變異量。
///
/// 為什麼是比較「最小」與「次小」而不是「最小 ÷ 總變異」（舊版的做法）：
/// 後者會被**水平方向的變異量**主導——相機繞得越廣（基線越長），分母越大，
/// 判定就越寬鬆，於是「軌跡其實是一團三維散開的點」也照樣過關。
/// 真正該問的是「這團相機中心是否幾乎共面」，也就是最小主軸相對次小主軸是否夠小。
///
/// ## 門檻是量出來的（24 台相機的合成軌跡，實測記錄）
/// | 軌跡 | 最小/次小 | 真正的最小主軸偏離垂直 | 舊判定的比值 | 舊判定 |
/// |---|---|---|---|---|
/// | 環繞（R=8 m、高度 ±0.5 m） | 0.003 | 0.6° | 0.001 | 放行 |
/// | 環繞（R=2 m、高度 ±0.5 m） | 0.050 | 2.5° | 0.022 | 放行 |
/// | 環繞（R=1.5 m、高度 ±0.5 m） | 0.088 | 3.4° | 0.039 | 放行 |
/// | **沿街直走（10 m、抖動 ±0.4 m）** | **0.169** | **49.5°** | **0.003** | **放行** |
/// | 廣場繞行（10×6 m、起伏 ±1.2 m） | 0.197 | 2.9° | 0.048 | 放行 |
/// | 三維散開（每軸 ±1 m） | 0.577 | 67.0° | 0.192 | 退回 |
/// | 近距離環繞（R=1 m、起伏 ±1 m） | 1.000 | 90.0° | 0.333 | 退回 |
///
/// 兩件事從這張表看得出來：
///   1. 舊判定的比值在**所有**危險的軌跡上都遠低於門檻（0.003！），因為它量的是
///      「回傳向量自己的變異」——歪掉的向量剛好把變異量算得很小，等於自己證明自己沒錯；
///   2. 比例與實際誤差不是單調關係（0.169 → 49.5°，0.197 → 2.9°），
///      因此不可能訂出一個「保證誤差在 N 度內」的門檻；0.10 的取捨是
///      「只要軌跡在三維上明顯散開，就不再宣稱推定成功」——退回世界 Y 軸雖然不一定等於重力，
///      但它會被明確標示出來（`UpAxisSource::WorldY`），使用者可以人工指定或重拍，
///      比默默給出偏 20° 以上的上軸要好。
const MAX_PLANARITY_RATIO: f64 = 0.10;

/// 由相機中心推定上軸（PCA：變異最小的主方向）。
///
/// 演算法：對相機中心建 3×3 共變異數 → 用 Jacobi 旋轉求完整特徵分解 →
/// 最小特徵值的方向即為上軸，但必須先確認軌跡真的接近一個平面（見 `MAX_PLANARITY_RATIO`）；
/// 否則退回世界 +Y 軸並把原因寫進 `degenerate_reason`。
///
/// **為什麼換掉舊版的冪迭代**：
///   1. 冪迭代是在 `(trace·I − cov)` 上求最大特徵向量，而那兩個主導特徵值在
///      「cov 的兩個較小特徵值接近」時幾乎相等——沿著一條街直走就是這種情況
///      （實測特徵值 `[218.5, 2.03, 0.34]`）。此時冪迭代收斂到兩者的**任意混合**：
///      舊版在該軌跡上回傳偏離垂直 27.8° 的向量，真正的最小主軸則是偏離 49.5°。
///      Jacobi 是直接法，對稱矩陣必定收斂，回傳真正的特徵向量。
///   2. 舊版的退化判定用**回傳向量自己**算變異量（`v^T·cov·v ÷ trace`）。這是自我證明：
///      向量越歪，它落在最小特徵值上的比重越大，算出來的比值就越小——同一條直走軌跡的
///      比值是 0.003，遠低於 0.17 的門檻，於是「偏掉 20° 以上」照樣被標成推定成功。
///      新判定改為比較兩個**特徵值**（最小 ÷ 次小），與回傳向量無關，也不受環繞半徑影響。
///
/// 誠實的界線：COLMAP 的世界座標沒有重力方向，所以「推定出來的上軸對不對」在本模組內
/// 無法驗證，只能判斷軌跡夠不夠「平」。因此 `degenerate_reason` 與 `UpAxisSource`
/// 必須一路傳到 QA 表與介面上（見 `Box3D::up_source`）。
pub fn infer_up_axis(images: &[Image]) -> UpAxisEstimate {
    if images.len() < 4 {
        return UpAxisEstimate::fallback(
            format!(
                "已註冊影格僅 {} 張（少於 4 張），無法由軌跡推定上軸",
                images.len()
            ),
            0.0,
        );
    }
    let centers: Vec<[f64; 3]> = images.iter().map(|i| i.camera_center()).collect();
    let n = centers.len() as f64;
    let mut mean = [0.0; 3];
    for c in &centers {
        for k in 0..3 {
            mean[k] += c[k] / n;
        }
    }
    // 3x3 共變異數（未除以 n；特徵向量的方向與縮放無關，但特徵值的比例要保持一致）
    let mut cov = [[0.0f64; 3]; 3];
    for c in &centers {
        let d = [c[0] - mean[0], c[1] - mean[1], c[2] - mean[2]];
        for i in 0..3 {
            for j in 0..3 {
                cov[i][j] += d[i] * d[j];
            }
        }
    }
    let (eigenvalues, eigenvectors) = symmetric_eigen_decomposition(cov);
    let (largest, middle, smallest) = (eigenvalues[0], eigenvalues[1], eigenvalues[2]);
    // 全部相機在同一個位置：沒有任何方向資訊
    if largest <= 1e-12 {
        return UpAxisEstimate::fallback("相機中心幾乎完全重合，軌跡沒有任何方向資訊", 0.0);
    }
    // 相機中心幾乎共線（例如沿著一條走廊直走）：平面本身沒有定義，最小主軸無意義
    if middle <= 1e-12 {
        return UpAxisEstimate::fallback("相機中心幾乎共線，無法決定「水平面」的方向", 0.0);
    }
    let planarity_ratio = (smallest / middle).max(0.0);
    if planarity_ratio > MAX_PLANARITY_RATIO {
        return UpAxisEstimate::fallback(
            format!(
                "相機軌跡在三維上散開（最小/次小主軸變異比 {planarity_ratio:.3} > {MAX_PLANARITY_RATIO:.2}），上軸不可靠"
            ),
            planarity_ratio,
        );
    }
    let mut up = eigenvectors[2];
    // 讓上軸與世界 +Y 同向（避免上下顛倒，讓高度為正）
    if up[1] < 0.0 {
        up = [-up[0], -up[1], -up[2]];
    }
    UpAxisEstimate {
        up,
        source: UpAxisSource::CameraTrajectory,
        planarity_ratio,
        degenerate_reason: None,
    }
}

/// 3×3 對稱矩陣的特徵分解（Jacobi 旋轉法）。
///
/// 回傳 `(eigenvalues, eigenvectors)`：特徵值由大到小排序，`eigenvectors[i]` 為對應的單位特徵向量。
/// 對稱矩陣的 Jacobi 迭代保證收斂（每一步都把最大的非對角元素消為 0，非對角平方和單調下降），
/// 因此沒有冪迭代「特徵值接近就停在中間」的問題；64 次旋轉對 3×3 而言已遠遠足夠
/// （每次旋轉至少讓非對角平方和乘上一個 < 1 的因子，且這裡的 `1e-12` 收斂門檻遠高於浮點雜訊）。
fn symmetric_eigen_decomposition(matrix: [[f64; 3]; 3]) -> ([f64; 3], [[f64; 3]; 3]) {
    let mut a = matrix;
    // v 累積所有旋轉；其**行**為特徵向量（v[i][j] = 第 i 個分量、第 j 個特徵向量）
    let mut v = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    for _ in 0..64 {
        // 找出絕對值最大的非對角元素
        let mut p = 0usize;
        let mut q = 1usize;
        let mut max = a[0][1].abs();
        for (i, j) in [(0usize, 2usize), (1usize, 2usize)] {
            if a[i][j].abs() > max {
                max = a[i][j].abs();
                p = i;
                q = j;
            }
        }
        if max < 1e-12 {
            break;
        }
        // 旋轉角：選 t 使得 a[p][q] 在旋轉後為 0（用數值穩定的形式，避免兩個相近數字相減）
        let theta = (a[q][q] - a[p][p]) / (2.0 * a[p][q]);
        let t = if theta >= 0.0 {
            1.0 / (theta + (theta * theta + 1.0).sqrt())
        } else {
            -1.0 / (-theta + (theta * theta + 1.0).sqrt())
        };
        let c = 1.0 / (t * t + 1.0).sqrt();
        let s = t * c;
        let (app, aqq, apq) = (a[p][p], a[q][q], a[p][q]);
        a[p][p] = c * c * app - 2.0 * s * c * apq + s * s * aqq;
        a[q][q] = s * s * app + 2.0 * s * c * apq + c * c * aqq;
        a[p][q] = 0.0;
        a[q][p] = 0.0;
        for r in 0..3 {
            if r != p && r != q {
                let (arp, arq) = (a[r][p], a[r][q]);
                a[r][p] = c * arp - s * arq;
                a[r][q] = s * arp + c * arq;
                a[p][r] = a[r][p];
                a[q][r] = a[r][q];
            }
            let (vrp, vrq) = (v[r][p], v[r][q]);
            v[r][p] = c * vrp - s * vrq;
            v[r][q] = s * vrp + c * vrq;
        }
    }
    let mut order = [0usize, 1, 2];
    order.sort_by(|&i, &j| {
        a[j][j]
            .partial_cmp(&a[i][i])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let eigenvalues = [
        a[order[0]][order[0]],
        a[order[1]][order[1]],
        a[order[2]][order[2]],
    ];
    let eigenvectors = [
        [v[0][order[0]], v[1][order[0]], v[2][order[0]]],
        [v[0][order[1]], v[1][order[1]], v[2][order[1]]],
        [v[0][order[2]], v[1][order[2]], v[2][order[2]]],
    ];
    (eigenvalues, eigenvectors)
}

/// 擬合包圍盒所需的最少點數。
const MIN_FIT_POINTS: usize = 8;

/// 包圍盒擬合失敗的原因。
///
/// 為什麼不是 `Option`：舊版只回 `None`，呼叫端於是把所有失敗都寫成「點雲不足」——
/// 但實際會走到 `None` 的情況至少有三種（沒有點、非有限值佔多數、有效點太少），
/// 而真正量到的退化（8 個完全重合的點）反而**成功**回傳了一個 0.2 m 的立方體。
/// 錯誤訊息必須說得出真正的原因，否則使用者無從修正。
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum FitError {
    #[error("點雲沒有任何點")]
    NoPoints,
    #[error("點雲 {non_finite} / {total} 個點含 NaN/Inf（超過半數），擬合結果不可信")]
    MostlyNonFinite { total: usize, non_finite: usize },
    #[error("有效點數僅 {valid} 個（需 ≥ {required} 個），無法擬合包圍盒")]
    TooFewPoints { valid: usize, required: usize },
    #[error(
        "點雲三軸跨度不足：最大跨度僅 {max_extent:.4} m（需 ≥ {min_extent:.4} m），無法構成包圍盒"
    )]
    DegenerateExtent { max_extent: f64, min_extent: f64 },
}

/// 由稀疏點雲擬合包圍盒。
///
/// 步驟：以 `up` 為垂直軸 → 把點投影到垂直平面上做 PCA 取得主方向 → 在三個方向上取修剪後的範圍。
pub fn fit_box_from_points(
    reconstruction: &SparseReconstruction,
    options: &FitOptions,
) -> Result<Box3D, FitError> {
    let points: Vec<[f64; 3]> = reconstruction.points.iter().map(|p| p.xyz).collect();
    fit_box_from_xyz(&points, &reconstruction.images, options)
}

/// 由座標陣列擬合（便於測試與外部重用）。
pub fn fit_box_from_xyz(
    points: &[[f64; 3]],
    images: &[Image],
    options: &FitOptions,
) -> Result<Box3D, FitError> {
    if points.is_empty() {
        return Err(FitError::NoPoints);
    }
    // 先剔除 NaN／Inf。
    //
    // 為什麼非在這裡過濾不可：Rust 的 `f64::min` / `f64::max` 會**忽略** NaN（回傳另一個運算元），
    // 與 Python 相反；`partial_cmp` 對 NaN 也回 None（原本用 `unwrap_or(Equal)` 當作相等）。
    // 於是單一個 NaN 座標就能讓包圍盒中心變成 NaN，而 size 看起來仍正常——
    // 接著 `area_ratio > max` 這類比較因 NaN 恆為 false 而攔不住，最後寫出
    // `0 NaN 0.5 NaN 0.5` 這種 YOLO 標籤，靜默污染整份訓練資料。
    let total = points.len();
    let points: Vec<[f64; 3]> = points
        .iter()
        .copied()
        .filter(|p| p.iter().all(|v| v.is_finite()))
        .collect();
    // 非有限值佔多數時直接拒絕：此時「成功」的擬合只反映了少數的點，
    // 卻會讓使用者以為整份點雲都參與了擬合。
    let non_finite = total - points.len();
    if non_finite * 2 > total {
        return Err(FitError::MostlyNonFinite { total, non_finite });
    }
    if points.len() < MIN_FIT_POINTS {
        return Err(FitError::TooFewPoints {
            valid: points.len(),
            required: MIN_FIT_POINTS,
        });
    }
    let estimate = if options.infer_up {
        let estimate = infer_up_axis(images);
        if let Some(reason) = &estimate.degenerate_reason {
            // 退化不是錯誤（標註照樣能做），但必須留下紀錄：使用者會想知道
            // 「為什麼我的建築高度看起來不對」。
            tracing::debug!("上軸推定退化，改用世界 Y 軸：{reason}");
        }
        estimate
    } else {
        // 操作者明確選擇不使用軌跡推定 → 來源是「指定」而不是「推定失敗」。
        UpAxisEstimate {
            up: [0.0, 1.0, 0.0],
            source: UpAxisSource::Manual,
            planarity_ratio: 0.0,
            degenerate_reason: None,
        }
    };
    let up = estimate.up;
    // 建立與 up 正交的基底 (e1, e2)
    let helper = if up[0].abs() < 0.9 {
        [1.0, 0.0, 0.0]
    } else {
        [0.0, 0.0, 1.0]
    };
    let mut e1 = cross(helper, up);
    normalize(&mut e1);
    let mut e2 = cross(up, e1);
    normalize(&mut e2);

    // 投影到 (e1, e2, up) 座標
    let local: Vec<[f64; 3]> = points
        .iter()
        .map(|p| [dot(*p, e1), dot(*p, e2), dot(*p, up)])
        .collect();

    // 垂直平面上的 PCA 取得長軸方向
    let mut mean2 = [0.0f64; 2];
    for p in &local {
        mean2[0] += p[0] / local.len() as f64;
        mean2[1] += p[1] / local.len() as f64;
    }
    let (mut cxx, mut cyy, mut cxy) = (0.0f64, 0.0f64, 0.0f64);
    for p in &local {
        let dx = p[0] - mean2[0];
        let dy = p[1] - mean2[1];
        cxx += dx * dx;
        cyy += dy * dy;
        cxy += dx * dy;
    }
    // 主方向角（2D）
    let theta = 0.5 * (2.0 * cxy).atan2(cxx - cyy);
    let (ct, st) = (theta.cos(), theta.sin());
    // 旋轉後的座標：u 沿長軸、v 沿短軸
    let rotated: Vec<[f64; 3]> = local
        .iter()
        .map(|p| {
            let du = p[0] - mean2[0];
            let dv = p[1] - mean2[1];
            [du * ct + dv * st, -du * st + dv * ct, p[2]]
        })
        .collect();

    // ※ `options.trim` 的範圍驗證**只存在一處**：命令列的 `value_parser`（0.0..=0.2）。
    //    這裡刻意不再 `clamp`——舊版的 `trim.clamp(0.0, 0.2)` 會讓 `--trim 0.9` 靜默等同 `--trim 0.2`，
    //    使用者以為自己調了參數卻毫無效果，是最糟的一種「容錯」。
    //    索引計算本身是飽和的（見下方 `range`），因此就算真的傳入離譜的值也只會得到退化結果，不會 panic。
    let trim = options.trim;
    let range = |axis: usize| -> (f64, f64) {
        let mut vals: Vec<f64> = rotated.iter().map(|p| p[axis]).collect();
        vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let last = vals.len() - 1;
        let lo = ((vals.len() as f64 * trim) as usize).min(last);
        let hi = ((vals.len() as f64 * (1.0 - trim)) as usize).min(last);
        (vals[lo], vals[hi].max(vals[lo]))
    };
    let (u0, u1) = range(0);
    let (v0, v1) = range(1);
    let (w0, w1) = range(2);
    let raw_extent = [u1 - u0, v1 - v0, w1 - w0];
    // 退化檢查：全部點重合（或幾乎重合）時，三個方向的跨度都是 0，
    // 靠 `min_extent` 撐出來的「0.2 m 立方體」是假的——它會讓呼叫端以為擬合成功。
    let max_extent = raw_extent.iter().copied().fold(0.0f64, f64::max);
    if max_extent < options.min_extent {
        return Err(FitError::DegenerateExtent {
            max_extent,
            min_extent: options.min_extent,
        });
    }
    let size = [
        raw_extent[0].max(options.min_extent),
        raw_extent[1].max(options.min_extent),
        raw_extent[2].max(options.min_extent),
    ];
    let center_local = [(u0 + u1) / 2.0, (v0 + v1) / 2.0, (w0 + w1) / 2.0];

    // 把中心轉回世界座標：先用 (ct, st) 轉回 (e1,e2) 平面，再回到世界
    let c1 = mean2[0] + center_local[0] * ct - center_local[1] * st;
    let c2 = mean2[1] + center_local[0] * st + center_local[1] * ct;
    let mut center = [0.0; 3];
    for k in 0..3 {
        center[k] = e1[k] * c1 + e2[k] * c2 + up[k] * center_local[2];
    }
    // 三個軸向：長軸 (e1 轉 theta)、短軸、上軸
    let mut ax0 = [0.0; 3];
    let mut ax1 = [0.0; 3];
    for k in 0..3 {
        ax0[k] = e1[k] * ct + e2[k] * st;
        ax1[k] = -e1[k] * st + e2[k] * ct;
    }
    Ok(Box3D {
        center,
        size,
        axes: [ax0, ax1, up],
        up_source: estimate.source,
        fitted_from_points: points.len(),
    })
}

fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn normalize(v: &mut [f64; 3]) {
    let n = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    if n > 1e-12 {
        v[0] /= n;
        v[1] /= n;
        v[2] /= n;
    }
}

/// 單張影格的標註結果。
#[derive(Debug, Clone, Serialize)]
pub struct FrameAnnotation {
    pub frame: String,
    /// 正規化後的 YOLO 框 `(cx, cy, w, h)`；`None` 表示這張影格不採用。
    pub yolo: Option<[f64; 4]>,
    /// 像素座標框 `(x0, y0, x1, y1)`。
    pub pixel: Option<[f64; 4]>,
    /// 8 個角點中，位於相機前方且投影落在畫面內（留 1% 邊界）的數量。
    pub visible_corners: usize,
    /// 投影框面積佔畫面比例。
    pub area_ratio: f64,
    /// 這張影格實際觀測到、且落在包圍盒內的三維點數（來自 COLMAP track）。
    pub observed_points: usize,
    /// 是否採用。
    pub accepted: bool,
    /// 不採用的原因。
    pub reason: Option<String>,
    /// 非致命的警示：這張影格仍然採用，但結果有已知的精度限制。
    ///
    /// 為什麼不塞進 `reason`：`reason` 的語意是「這張不採用」，
    /// 而警示的影格是**要寫進資料集**的。混在一起會讓「拒絕原因分佈」出現大量假原因，
    /// 也讓「採用但要注意」這件事在統計裡消失。
    /// 目前唯一的來源是相機模型（魚眼等）造成的近似投影。
    pub warning: Option<String>,
}

/// 標註參數。
#[derive(Debug, Clone)]
pub struct AnnotateOptions {
    /// 至少要有幾個可見角點才採用。
    pub min_visible_corners: usize,
    /// 投影框面積佔畫面比例的下限與上限。
    ///
    /// 上限設 0.85 而非 0.95：對「偵測器訓練資料」而言，物件佔滿畫面並不是好樣本
    /// （相機貼太近、看不到建築全貌），這種影格應該排除。
    pub min_area_ratio: f64,
    pub max_area_ratio: f64,
    /// 影格內至少要有幾個包圍盒內的點被觀測到（遮擋代理指標）。
    pub min_observed_points: usize,
    /// 嚴格模式：要求 8 個角點全部可見才採用。
    ///
    /// 為什麼需要這個選項：當某些角點落在**相機後方**時，它們無法投影，
    /// 此時用「可見角點的投影範圍」得到的框會**低估**真實輪廓（框可能比物件小）。
    /// 需要「框一定完整包住物件」的場合請開啟此選項；預設關閉以容許部分可見的情況。
    pub require_all_corners: bool,
}

impl Default for AnnotateOptions {
    fn default() -> Self {
        Self {
            min_visible_corners: 4,
            min_area_ratio: 0.01,
            max_area_ratio: 0.85,
            min_observed_points: 3,
            require_all_corners: false,
        }
    }
}

/// 相機模型的投影近似警示（沒有問題時回 `None`）。
///
/// 為什麼要有這個：`camera_to_pixel` 實作的是針孔＋Brown 徑向／切向畸變，
/// 對魚眼（以及 Fov、FullOpenCv 的高階項）只能給近似值。被採用的影格照樣會寫出標籤，
/// 但使用者必須知道「這些框的像素位置不是精確的」——默默給出偏掉的框是不能接受的。
fn projection_warning(camera: &Camera) -> Option<String> {
    if !camera.model.projection_is_approximate() {
        return None;
    }
    Some(if camera.model.is_fisheye() {
        format!(
            "相機模型 {:?} 為魚眼；本模組以針孔＋徑向畸變近似投影，框位可能偏移",
            camera.model
        )
    } else {
        format!(
            "相機模型 {:?} 的部分畸變項未實作；投影僅為近似，框位可能偏移",
            camera.model
        )
    })
}

/// 把包圍盒投影到單張影格。
pub fn annotate_frame(
    image: &Image,
    camera: &Camera,
    box3d: &Box3D,
    options: &AnnotateOptions,
    observed_points: usize,
) -> FrameAnnotation {
    // 不論採用與否，只要這個相機模型的投影只是近似，就要在結果上留下警示
    let warning = projection_warning(camera);
    let (w, h) = (camera.width as f64, camera.height as f64);
    let mut min_u = f64::INFINITY;
    let mut min_v = f64::INFINITY;
    let mut max_u = f64::NEG_INFINITY;
    let mut max_v = f64::NEG_INFINITY;
    let mut visible = 0usize;
    for corner in box3d.corners() {
        if let Some((u, v, _z)) = camera_to_pixel(image.to_camera(corner), camera) {
            if u >= -0.01 * w && u <= 1.01 * w && v >= -0.01 * h && v <= 1.01 * h {
                visible += 1;
            }
            min_u = min_u.min(u);
            min_v = min_v.min(v);
            max_u = max_u.max(u);
            max_v = max_v.max(v);
        }
    }
    if !min_u.is_finite() {
        return FrameAnnotation {
            frame: image.name.clone(),
            yolo: None,
            pixel: None,
            visible_corners: 0,
            area_ratio: 0.0,
            observed_points,
            accepted: false,
            reason: Some("整個包圍盒都在相機後方".into()),
            warning,
        };
    }
    // 夾在畫面內，並限制成合法範圍
    let x0 = min_u.clamp(0.0, w);
    let y0 = min_v.clamp(0.0, h);
    let x1 = max_u.clamp(0.0, w);
    let y1 = max_v.clamp(0.0, h);
    let bw = (x1 - x0).max(0.0);
    let bh = (y1 - y0).max(0.0);
    let area_ratio = (bw * bh) / (w * h);

    let mut reason = None;
    // 非有限值守門：幾何算完後再檢查一次。前面已過濾點雲中的 NaN，這裡擋的是
    // 相機參數或包圍盒本身退化的情況——與其寫出壞標籤，不如明確拒絕這張影格。
    let finite_projection = x0.is_finite()
        && y0.is_finite()
        && x1.is_finite()
        && y1.is_finite()
        && bw.is_finite()
        && bh.is_finite()
        && area_ratio.is_finite();
    if !finite_projection {
        reason = Some("投影結果非有限值（點雲或包圍盒含 NaN/Inf）".into());
    } else if options.require_all_corners && visible < 8 {
        reason = Some(format!(
            "嚴格模式要求 8 個角點全可見，此影格僅 {visible} 個（框可能低估輪廓）"
        ));
    } else if visible < options.min_visible_corners {
        reason = Some(format!(
            "可見角點僅 {visible} 個（需 ≥{}）",
            options.min_visible_corners
        ));
    } else if area_ratio < options.min_area_ratio {
        reason = Some(format!("投影框過小（{:.1}% 畫面）", area_ratio * 100.0));
    } else if area_ratio > options.max_area_ratio {
        reason = Some(format!(
            "投影框過大（{:.1}% 畫面，可能相機貼太近）",
            area_ratio * 100.0
        ));
    } else if observed_points < options.min_observed_points {
        reason = Some(format!(
            "此影格觀測到的盒內點僅 {observed_points} 個（可能被遮擋）"
        ));
    }
    let accepted = reason.is_none();
    FrameAnnotation {
        frame: image.name.clone(),
        yolo: accepted.then(|| [((x0 + x1) / 2.0) / w, ((y0 + y1) / 2.0) / h, bw / w, bh / h]),
        pixel: accepted.then_some([x0, y0, x1, y1]),
        visible_corners: visible,
        area_ratio,
        observed_points,
        accepted,
        reason,
        warning,
    }
}

/// 對整個重建產生標註。
pub fn annotate_reconstruction(
    reconstruction: &SparseReconstruction,
    box3d: &Box3D,
    options: &AnnotateOptions,
) -> Vec<FrameAnnotation> {
    // 先統計每張影格觀測到的盒內點數（用 track 反查，避免逐影格掃描全部點）
    let mut observed: std::collections::HashMap<u32, usize> = std::collections::HashMap::new();
    for p in &reconstruction.points {
        if !box3d.contains(p.xyz) {
            continue;
        }
        for (image_id, _) in &p.track {
            *observed.entry(*image_id).or_insert(0) += 1;
        }
    }
    let mut out = Vec::with_capacity(reconstruction.images.len());
    for image in &reconstruction.images {
        let Some(camera) = reconstruction.cameras.get(&image.camera_id) else {
            out.push(FrameAnnotation {
                frame: image.name.clone(),
                yolo: None,
                pixel: None,
                visible_corners: 0,
                area_ratio: 0.0,
                observed_points: 0,
                accepted: false,
                reason: Some(format!("找不到 camera_id={} 的內參", image.camera_id)),
                warning: None,
            });
            continue;
        };
        let n = observed.get(&image.id).copied().unwrap_or(0);
        out.push(annotate_frame(image, camera, box3d, options, n));
    }
    out.sort_by(|a, b| a.frame.cmp(&b.frame));
    out
}

/// 標註統計摘要。
#[derive(Debug, Clone, Serialize)]
pub struct AnnotationSummary {
    pub frames_total: usize,
    pub frames_accepted: usize,
    pub acceptance_ratio: f64,
    pub median_visible_corners: f64,
    pub median_area_ratio: f64,
    pub median_observed_points: f64,
    /// 各拒絕原因的分佈。
    pub rejection_reasons: Vec<(String, usize)>,
    /// 被採用、但帶有警示的影格分佈（例如「相機為魚眼，投影僅近似」）。
    pub warnings: Vec<(String, usize)>,
    /// 這次寫標籤時，舊標籤目錄中被清掉的 `.txt` 檔數。
    ///
    /// `summarize` 只看得見記憶體裡的標註，因此它恆為 0；真正的值由
    /// `write_yolo_labels` 的 `LabelWriteReport::removed` 填入（見呼叫端）。
    /// 為什麼要讓它可見：重跑標註（例如換了 `--require-all-corners`）時，
    /// 上一輪被採用的影格這一輪可能被略過；那些舊 `.txt` 若留著，就等於憑空多出的正樣本。
    pub labels_removed: usize,
}

/// 把訊息中的數字換成 `#`，讓「可見角點僅 1 個」與「可見角點僅 0 個」歸為同一類。
fn normalize_message(text: &str) -> String {
    let head = text.split('（').next().unwrap_or(text).trim();
    head.chars()
        .map(|ch| if ch.is_ascii_digit() { '#' } else { ch })
        .collect()
}

/// 把訊息清單聚合成「訊息 → 次數」，並依次數由多到少排序。
fn aggregate_messages<'a>(messages: impl Iterator<Item = &'a str>) -> Vec<(String, usize)> {
    let mut counts: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
    for message in messages {
        *counts.entry(normalize_message(message)).or_insert(0) += 1;
    }
    let mut out: Vec<(String, usize)> = counts.into_iter().collect();
    out.sort_by_key(|(_, count)| std::cmp::Reverse(*count));
    out
}

/// 產生摘要統計（拒絕原因與警示都會按訊息聚合）。
pub fn summarize(annotations: &[FrameAnnotation]) -> AnnotationSummary {
    let total = annotations.len();
    let accepted = annotations.iter().filter(|a| a.accepted).count();
    let med = |mut v: Vec<f64>| -> f64 {
        if v.is_empty() {
            return 0.0;
        }
        v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        v[v.len() / 2]
    };
    let rejection_reasons = aggregate_messages(
        annotations
            .iter()
            .filter(|a| !a.accepted)
            .map(|a| a.reason.as_deref().unwrap_or("未知")),
    );
    let warnings = aggregate_messages(annotations.iter().filter_map(|a| a.warning.as_deref()));
    AnnotationSummary {
        frames_total: total,
        frames_accepted: accepted,
        acceptance_ratio: if total == 0 {
            0.0
        } else {
            accepted as f64 / total as f64
        },
        median_visible_corners: med(annotations
            .iter()
            .map(|a| a.visible_corners as f64)
            .collect()),
        median_area_ratio: med(annotations.iter().map(|a| a.area_ratio).collect()),
        median_observed_points: med(annotations
            .iter()
            .map(|a| a.observed_points as f64)
            .collect()),
        rejection_reasons,
        warnings,
        labels_removed: 0,
    }
}

/// 寫出標籤的結果。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LabelWriteReport {
    /// 這次寫出的標籤檔數。
    pub written: usize,
    /// 上次留下的舊標籤中被清掉的檔數。
    pub removed: usize,
}

/// 標籤目錄的暫存鄰居（`labels` → `labels.tmp`）。
fn staging_directory(dir: &Path) -> std::path::PathBuf {
    let name = dir
        .file_name()
        .map(|n| n.to_string_lossy().to_string())
        .unwrap_or_else(|| "labels".to_string());
    dir.with_file_name(format!("{name}.tmp"))
}

/// 舊標籤目錄裡的 `.txt` 檔數（目錄不存在時為 0）。
fn count_existing_labels(dir: &Path) -> std::io::Result<usize> {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return Ok(0);
    };
    let mut count = 0;
    for entry in entries {
        let entry = entry?;
        let path = entry.path();
        if path.is_file()
            && path
                .extension()
                .is_some_and(|ext| ext.eq_ignore_ascii_case("txt"))
        {
            count += 1;
        }
    }
    Ok(count)
}

/// 以 YOLO 格式寫出標籤（`<class_id> <cx> <cy> <w> <h>`，皆為 0~1 正規化）。
///
/// **整個標籤目錄會被換掉**，不是逐檔覆蓋：
/// YOLO 的慣例是「沒有標籤檔 = 背景」，所以上一輪寫出、這一輪不再被採用的影格，
/// 其 `.txt` 必須消失。舊版只做 `create_dir_all` + 逐檔 `fs::write`，
/// 於是「換參數重跑」（例如加上 `--require-all-corners`）會留下上一輪的標籤，
/// 讓被略過的影格變成憑空多出的正樣本——而且完全沒有提示。
///
/// 作法是先寫進 `<dir>.tmp/`，全部成功後才刪掉舊目錄並更名過來：
/// 寫入中途失敗（磁碟滿、權限）時，舊標籤仍是完整可用的那一份，不會留下半套資料。
/// 副作用要明講：放在這個目錄裡的**任何**檔案都會被移除，人工補充的說明檔請放別處。
pub fn write_yolo_labels(
    dir: impl AsRef<Path>,
    annotations: &[FrameAnnotation],
    class_id: usize,
) -> std::io::Result<LabelWriteReport> {
    let dir = dir.as_ref();
    if let Some(parent) = dir.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let staging = staging_directory(dir);
    // 上次中斷可能留下暫存目錄：先清掉，避免舊檔混進這次的結果
    if staging.exists() {
        std::fs::remove_dir_all(&staging)?;
    }
    std::fs::create_dir_all(&staging)?;

    let mut written = 0;
    for a in annotations.iter().filter(|a| a.accepted) {
        let Some(y) = a.yolo else { continue };
        // 最後一道防線：寧可少一個標籤，也不要把 NaN 寫進訓練資料
        if !y.iter().all(|v| v.is_finite()) {
            continue;
        }
        let stem = Path::new(&a.frame)
            .file_stem()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_else(|| a.frame.clone());
        let line = format!(
            "{class_id} {:.6} {:.6} {:.6} {:.6}\n",
            y[0], y[1], y[2], y[3]
        );
        std::fs::write(staging.join(format!("{stem}.txt")), line)?;
        written += 1;
    }

    let removed = count_existing_labels(dir)?;
    if dir.exists() {
        std::fs::remove_dir_all(dir)?;
    }
    std::fs::rename(&staging, dir)?;
    if removed > 0 {
        tracing::debug!("重跑標註：已清掉 {removed} 個舊標籤檔（{dir:?}）");
    }
    Ok(LabelWriteReport { written, removed })
}

/// CSV 欄位轉義（RFC 4180）：欄位含 `,`、`"`、換行時用雙引號包住，內部的 `"` 變成 `""`。
///
/// 為什麼需要：COLMAP 記錄的是磁碟上的檔名，而 Windows 的檔名合法包含 `,` 與 `"`；
/// 直接串接會讓那一列多出欄位（表格整個歪掉），或讓引號吃掉落後的內容。
fn escape_csv(field: &str) -> String {
    if field.contains([',', '"', '\n', '\r']) {
        format!("\"{}\"", field.replace('"', "\"\""))
    } else {
        field.to_string()
    }
}

/// HTML 文字轉義：`&`、`<`、`>`、`"`、`'`。
///
/// 為什麼需要：檔名直接插進 `src="..."`、`alt="..."` 時，一個 `"` 就能讓屬性提前結束，
/// 之後的內容會被當成標記解析（複核表上就再也看不到那張影格了）。
/// `&` 必須第一個換，否則後面產生的實體會被二次轉義。
fn escape_html(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    for ch in text.chars() {
        match ch {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '"' => out.push_str("&quot;"),
            '\'' => out.push_str("&#39;"),
            _ => out.push(ch),
        }
    }
    out
}

/// 寫出逐影格的標註明細（CSV，方便用試算表檢查）。
///
/// 欄位一律經過 `escape_csv`：檔名與原因都可能含逗號或引號。
/// （舊版把原因裡的 `,` 直接換成全形 `，`——那是為了掩蓋沒有轉義的問題而**改動了資料**，
/// 現在原因原文照寫，交給 CSV 規則處理。）
pub fn write_annotations_csv(
    path: impl AsRef<Path>,
    annotations: &[FrameAnnotation],
) -> std::io::Result<()> {
    let mut out = String::from(
        "frame,accepted,cx,cy,w,h,x0,y0,x1,y1,visible_corners,area_ratio,observed_points,reason,warning\n",
    );
    for a in annotations {
        let y = a.yolo.unwrap_or([f64::NAN; 4]);
        let p = a.pixel.unwrap_or([f64::NAN; 4]);
        let fmt = |v: f64| {
            if v.is_nan() {
                String::new()
            } else {
                format!("{v:.4}")
            }
        };
        let fields = [
            escape_csv(&a.frame),
            escape_csv(&a.accepted.to_string()),
            escape_csv(&fmt(y[0])),
            escape_csv(&fmt(y[1])),
            escape_csv(&fmt(y[2])),
            escape_csv(&fmt(y[3])),
            escape_csv(&fmt(p[0])),
            escape_csv(&fmt(p[1])),
            escape_csv(&fmt(p[2])),
            escape_csv(&fmt(p[3])),
            escape_csv(&a.visible_corners.to_string()),
            escape_csv(&format!("{:.4}", a.area_ratio)),
            escape_csv(&a.observed_points.to_string()),
            escape_csv(a.reason.as_deref().unwrap_or_default()),
            escape_csv(a.warning.as_deref().unwrap_or_default()),
        ];
        out.push_str(&fields.join(","));
        out.push('\n');
    }
    std::fs::write(path, out)
}

/// 產生人工複核用的 HTML 聯絡表（不需任何影像程式庫：圖片用 `<img>`，框用絕對定位的 `<div>`）。
///
/// `frames_dir` 為**相對於 `out_path` 所在目錄**的影格路徑前綴。
/// 呼叫端必須傳入實際找到的影格目錄（`inspect::frames_href` 會算好），
/// 不可以再寫死 `../work/frames`：影格目錄有三個候選位置（`work/frames`／`frames`／`source/images`），
/// 寫死會讓「影格其實在 source/images」的專案產生一整頁破圖。
///
/// 所有插值都經過 `escape_html`：檔名來自使用者的磁碟，可以合法包含 `"`、`&`、`<`。
pub fn write_qa_html(
    out_path: impl AsRef<Path>,
    frames_dir: &str,
    annotations: &[FrameAnnotation],
    summary: &AnnotationSummary,
    box3d: &Box3D,
    max_frames: usize,
) -> std::io::Result<()> {
    let out_path = out_path.as_ref();
    let frames_dir = escape_html(frames_dir);
    let mut html = String::new();
    html.push_str(
        "<!doctype html><html lang=\"zh-Hant\"><head><meta charset=\"utf-8\">\
         <title>MacauSplat 標註複核</title><style>\
         body{margin:0;background:#0f1620;color:#e8eef5;font-family:'Microsoft JhengHei',system-ui,sans-serif}\
         header{padding:14px 20px;background:#10243a;border-bottom:3px solid #c8a24a}\
         h1{margin:0;font-size:19px} .sub{color:#9fb3c8;font-size:13px;margin-top:6px}\
         .grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(260px,1fr));gap:10px;padding:14px}\
         .cell{background:#16202c;border-radius:8px;overflow:hidden;border:1px solid #26374a}\
         .wrap{position:relative;width:100%;background:#000}\
         .wrap img{width:100%;display:block}\
         .box{position:absolute;border:2px solid #c8a24a;box-sizing:border-box}\
         .meta{padding:6px 8px;font-size:12px;line-height:1.5}\
         .ok{color:#4ade80} .no{color:#f87171} .dim{color:#8fa3b8} .warn{color:#e0b055}\
         </style></head><body><header>\
         <h1>MacauSplat 標註複核表</h1><div class=\"sub\">",
    );
    html.push_str(&format!(
        "共 {} 張影格，採用 {} 張（{:.1}%）｜中位可見角點 {:.0}｜中位畫面佔比 {:.2}%｜中位觀測點數 {:.0}\
         ｜包圍盒 {:.2}×{:.2}×{:.2} m（上軸：{}）",
        summary.frames_total,
        summary.frames_accepted,
        summary.acceptance_ratio * 100.0,
        summary.median_visible_corners,
        summary.median_area_ratio * 100.0,
        summary.median_observed_points,
        box3d.size[0],
        box3d.size[1],
        box3d.size[2],
        escape_html(box3d.up_source.label()),
    ));
    // 重跑時清掉的舊標籤數：只有真的清掉東西才顯示，避免每天看到一個 0
    if summary.labels_removed > 0 {
        html.push_str(&format!(
            "<br><span class=\"warn\">本次重跑清掉了 {} 個上一輪的舊標籤檔（不再採用的影格不會留下舊框）</span>",
            summary.labels_removed
        ));
    }
    for (message, count) in &summary.warnings {
        html.push_str(&format!(
            "<br><span class=\"warn\">{count} 張影格有警示：{}</span>",
            escape_html(message)
        ));
    }
    html.push_str("</div></header><div class=\"grid\">");

    // 先顯示被採用的（人工複核的重點），再顯示被拒絕的樣本
    let mut ordered: Vec<&FrameAnnotation> = annotations.iter().filter(|a| a.accepted).collect();
    ordered.extend(annotations.iter().filter(|a| !a.accepted));
    for a in ordered.into_iter().take(max_frames) {
        let name = escape_html(
            &Path::new(&a.frame)
                .file_name()
                .map(|s| s.to_string_lossy().to_string())
                .unwrap_or_else(|| a.frame.clone()),
        );
        html.push_str("<div class=\"cell\"><div class=\"wrap\">");
        html.push_str(&format!(
            "<img loading=\"lazy\" src=\"{frames_dir}/{name}\" alt=\"{name}\">"
        ));
        // 框的位置：YOLO 的 (cx, cy, w, h) 本來就是相對畫面比例，直接當 CSS 百分比用即可，
        // 不需要任何影像程式庫，也不需要在瀏覽器端重算。
        if let Some(y) = a.yolo {
            html.push_str(&format!(
                "<div class=\"box\" style=\"left:{:.3}%;top:{:.3}%;width:{:.3}%;height:{:.3}%\"></div>",
                (y[0] - y[2] / 2.0) * 100.0,
                (y[1] - y[3] / 2.0) * 100.0,
                y[2] * 100.0,
                y[3] * 100.0
            ));
        }
        html.push_str("</div><div class=\"meta\">");
        html.push_str(&format!(
            "<b>{}</b> <span class=\"{}\">{}</span><br>",
            name,
            if a.accepted { "ok" } else { "no" },
            if a.accepted { "採用" } else { "略過" }
        ));
        html.push_str(&format!(
            "<span class=\"dim\">角點 {}/8｜畫面 {:.1}%｜盒內觀測點 {}</span>",
            a.visible_corners,
            a.area_ratio * 100.0,
            a.observed_points
        ));
        if let Some(r) = &a.reason {
            html.push_str(&format!("<br><span class=\"no\">{}</span>", escape_html(r)));
        }
        if let Some(w) = &a.warning {
            html.push_str(&format!(
                "<br><span class=\"warn\">{}</span>",
                escape_html(w)
            ));
        }
        html.push_str("</div></div>");
    }
    html.push_str("</div>");
    html.push_str(&format!(
        "<p style=\"padding:0 20px 24px;color:#8fa3b8;font-size:12px\">僅顯示前 {max_frames} 張；完整明細見 annotations.csv</p></body></html>"
    ));
    std::fs::write(out_path, html)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::reconstruction::colmap::{CameraModel, Point3D};

    fn camera() -> Camera {
        Camera {
            id: 1,
            model: CameraModel::Pinhole,
            width: 1000,
            height: 1000,
            params: vec![1000.0, 1000.0, 500.0, 500.0],
        }
    }

    /// 相機在 (0,0,0) 朝向 +Z 時，位於正前方 10 公尺、邊長 2 公尺的立方體
    /// 應投影成畫面中央、邊長約 200 像素、面積比約 4%。
    #[test]
    fn projects_box_in_front_of_camera() {
        let image = Image {
            id: 1,
            qvec: [1.0, 0.0, 0.0, 0.0],
            tvec: [0.0, 0.0, 0.0],
            camera_id: 1,
            name: "f.jpg".into(),
        };
        let b = Box3D::axis_aligned([0.0, 0.0, 10.0], [2.0, 2.0, 2.0]);
        let a = annotate_frame(&image, &camera(), &b, &AnnotateOptions::default(), 10);
        assert_eq!(a.visible_corners, 8, "8 個角都應在畫面內");
        assert!(a.accepted, "應被採用：{a:?}");
        // 注意：立方體近面在 z=9、遠面在 z=11，透視下近面投影更大，
        // 因此框寬是 2 * (1000/9) ≈ 222 px 而非 2 * (1000/10) = 200 px。
        let px = a.pixel.unwrap();
        let expect = 500.0 - 1000.0 / 9.0;
        assert!(
            (px[0] - expect).abs() < 1.0 && (px[2] - (1000.0 - expect)).abs() < 1.0,
            "框位置錯誤（應為近面投影 {expect:.1}）：{px:?}"
        );
        assert!(
            (a.area_ratio - 0.0493).abs() < 0.005,
            "面積比應約 4.9%，實際 {}",
            a.area_ratio
        );
        let y = a.yolo.unwrap();
        assert!((y[0] - 0.5).abs() < 0.01, "YOLO 中心應在畫面中央：{y:?}");
    }

    /// 相機後方的盒應被拒絕，且不可 panic。
    #[test]
    fn rejects_box_behind_camera() {
        let image = Image {
            id: 1,
            qvec: [1.0, 0.0, 0.0, 0.0],
            tvec: [0.0, 0.0, 0.0],
            camera_id: 1,
            name: "f.jpg".into(),
        };
        let b = Box3D::axis_aligned([0.0, 0.0, -10.0], [2.0, 2.0, 2.0]);
        let a = annotate_frame(&image, &camera(), &b, &AnnotateOptions::default(), 10);
        assert!(!a.accepted);
        assert_eq!(a.visible_corners, 0);
        assert!(a.reason.is_some());
    }

    /// 太遠（框太小）與太近（框太大）都應被拒絕。
    #[test]
    fn rejects_too_small_or_too_large_boxes() {
        let image = Image {
            id: 1,
            qvec: [1.0, 0.0, 0.0, 0.0],
            tvec: [0.0, 0.0, 0.0],
            camera_id: 1,
            name: "f.jpg".into(),
        };
        let far = Box3D::axis_aligned([0.0, 0.0, 5000.0], [2.0, 2.0, 2.0]);
        let a = annotate_frame(&image, &camera(), &far, &AnnotateOptions::default(), 10);
        assert!(
            !a.accepted && a.reason.as_deref().unwrap().contains("過小"),
            "{a:?}"
        );

        // 「過大」的測法：盒子必須完整落在視錐內（8 個角都可見），但投影後佔滿畫面。
        // f=1000、畫面 1000px ⇒ 半視角 26.6°；邊長 8 的盒子放在 z=12.2 時，
        // 近面 (z=8.2) 的半寬投影為 4*1000/8.2 ≈ 488 px，框寬約 976 px（≈95% 畫面）→ 應判為「過大」。
        let near = Box3D::axis_aligned([0.0, 0.0, 12.2], [8.0, 8.0, 8.0]);
        let a = annotate_frame(&image, &camera(), &near, &AnnotateOptions::default(), 10);
        assert_eq!(a.visible_corners, 8, "8 個角都應在畫面內：{a:?}");
        assert!(
            !a.accepted && a.reason.as_deref().unwrap().contains("過大"),
            "{a:?}"
        );

        // 反之，若盒子大到角點全部跑出畫面／跑到相機後方，會先被「可見角點」攔下——
        // 這是合理行為（連完整輪廓都看不到的影格不該當訓練樣本）。
        let absurd = Box3D::axis_aligned([0.0, 0.0, 1.0], [40.0, 40.0, 40.0]);
        let a = annotate_frame(&image, &camera(), &absurd, &AnnotateOptions::default(), 10);
        assert!(
            !a.accepted && a.reason.as_deref().unwrap().contains("可見角點"),
            "{a:?}"
        );
    }

    /// 觀測點數太少（遮擋）應被拒絕。
    #[test]
    fn rejects_occluded_frames() {
        let image = Image {
            id: 1,
            qvec: [1.0, 0.0, 0.0, 0.0],
            tvec: [0.0, 0.0, 0.0],
            camera_id: 1,
            name: "f.jpg".into(),
        };
        let b = Box3D::axis_aligned([0.0, 0.0, 10.0], [2.0, 2.0, 2.0]);
        let a = annotate_frame(&image, &camera(), &b, &AnnotateOptions::default(), 0);
        assert!(
            !a.accepted && a.reason.as_deref().unwrap().contains("遮擋"),
            "{a:?}"
        );
    }

    /// 上軸推定：相機繞同一水平圓環走一圈 → 上軸應接近世界 Y 軸，來源為「由相機軌跡推定」。
    #[test]
    fn infers_up_axis_from_circular_camera_path() {
        let mut images = Vec::new();
        for i in 0..12 {
            let a = i as f64 * std::f64::consts::TAU / 12.0;
            // 相機位置 (cos, 0, sin)*5 → 相機中心 = -R^T t；此處直接給 t，R=I 時 center = -t
            images.push(Image {
                id: i as u32 + 1,
                qvec: [1.0, 0.0, 0.0, 0.0],
                tvec: [-5.0 * a.cos(), 0.0, -5.0 * a.sin()],
                camera_id: 1,
                name: format!("f{i:02}.jpg"),
            });
        }
        let estimate = infer_up_axis(&images);
        assert_eq!(
            estimate.source,
            UpAxisSource::CameraTrajectory,
            "圓環軌跡應可推出上軸：{estimate:?}"
        );
        assert!(
            estimate.degenerate_reason.is_none(),
            "不該有退化原因：{estimate:?}"
        );
        assert!(
            estimate.up[1].abs() > 0.98,
            "上軸應接近世界 Y 軸，實際 {:?}",
            estimate.up
        );
    }

    /// 真實的繞行軌跡不可能完美（半徑抖動、角度不均、高度小幅散布），
    /// 而上軸仍必須落在垂直方向的 3° 以內。
    ///
    /// 這是**精度**測試（不是舊 bug 的回歸測試）：舊版冪迭代在這個資料上也會收斂到正確方向
    /// （實測差異 0.0°），因為水平面兩個主軸的變異量比例還夠分開。
    /// 真正讓舊版靜默出錯的是「最小與次小主軸變異量接近」的軌跡
    /// （見 `straight_line_walk_never_silently_claims_an_up_axis`）。
    /// 兩者都值得測：一個守住精度，一個守住誠實。
    #[test]
    fn up_axis_survives_a_nearly_isotropic_orbit() {
        let mut images = Vec::new();
        for i in 0..24 {
            let jitter = ((i * 7 % 13) as f64 - 6.0) / 6.0; // -1..1
            let radius = 8.0 * (1.0 + 0.005 * jitter);
            let a = i as f64 * std::f64::consts::TAU / 24.0 + 0.01 * (i as f64).sin();
            let height = 0.05 * (((i * 5) % 7) as f64 - 3.0) / 3.0;
            images.push(Image {
                id: i as u32 + 1,
                qvec: [1.0, 0.0, 0.0, 0.0],
                tvec: [-radius * a.cos(), -height, -radius * a.sin()],
                camera_id: 1,
                name: format!("f{i:02}.jpg"),
            });
        }
        let estimate = infer_up_axis(&images);
        assert_eq!(estimate.source, UpAxisSource::CameraTrajectory);
        // cos(3°) ≈ 0.9986；舊版在這個資料上只到 ~0.93（偏離 22°）
        assert!(
            estimate.up[1] > 0.9986,
            "上軸偏離垂直過多（應在 3° 內）：{:?}，planarity={}",
            estimate.up,
            estimate.planarity_ratio
        );
        assert!(
            estimate.planarity_ratio < 0.01,
            "這種軌跡應被判定為接近平面，實際 {}",
            estimate.planarity_ratio
        );
    }

    /// 螺旋上升（邊繞邊爬升）仍是「接近平面」的軌跡：上軸應該正確，而不是被判定退化。
    #[test]
    fn gentle_spiral_still_yields_the_vertical_axis() {
        let mut images = Vec::new();
        for i in 0..24 {
            let t = i as f64 / 23.0;
            let a = t * 1.5 * std::f64::consts::TAU;
            let height = -0.5 + t; // 高度散布 ±0.5 m
            images.push(Image {
                id: i as u32 + 1,
                qvec: [1.0, 0.0, 0.0, 0.0],
                tvec: [-8.0 * a.cos(), -height, -8.0 * a.sin()],
                camera_id: 1,
                name: format!("f{i:02}.jpg"),
            });
        }
        let estimate = infer_up_axis(&images);
        assert_eq!(
            estimate.source,
            UpAxisSource::CameraTrajectory,
            "溫和的螺旋仍可推定上軸：{estimate:?}"
        );
        assert!(
            estimate.up[1] > 0.99,
            "螺旋軌跡的上軸仍應接近垂直，實際 {:?}",
            estimate.up
        );
    }

    /// 軌跡在三維上散開（相機高度變化與環繞半徑同量級）時，上軸不可靠 →
    /// 必須退回世界 Y 軸並說出原因，而不是給一個看起來很確定的錯方向。
    #[test]
    fn dispersed_trajectory_falls_back_to_world_y_with_a_reason() {
        let mut images = Vec::new();
        for i in 0..27 {
            let (x, y, z) = (
                (i % 3) as f64 - 1.0,
                ((i / 3) % 3) as f64 - 1.0,
                (i / 9) as f64 - 1.0,
            );
            images.push(Image {
                id: i as u32 + 1,
                qvec: [1.0, 0.0, 0.0, 0.0],
                tvec: [-x, -y, -z],
                camera_id: 1,
                name: format!("f{i}.jpg"),
            });
        }
        let estimate = infer_up_axis(&images);
        assert_eq!(estimate.source, UpAxisSource::WorldY);
        assert_eq!(estimate.up, [0.0, 1.0, 0.0]);
        let reason = estimate.degenerate_reason.expect("必須說明退化原因");
        assert!(
            reason.contains("三維上散開"),
            "原因應指出軌跡不是平面：{reason}"
        );
    }

    /// 相機中心共線（沿一條走廊直走）時沒有定義良好的「水平面」→ 退回世界 Y 軸。
    #[test]
    fn collinear_trajectory_falls_back_to_world_y() {
        let images: Vec<Image> = (0..8)
            .map(|i| Image {
                id: i + 1,
                qvec: [1.0, 0.0, 0.0, 0.0],
                tvec: [-(i as f64), 0.0, 0.0],
                camera_id: 1,
                name: format!("f{i}.jpg"),
            })
            .collect();
        let estimate = infer_up_axis(&images);
        assert_eq!(estimate.source, UpAxisSource::WorldY);
        assert!(
            estimate
                .degenerate_reason
                .as_deref()
                .is_some_and(|r| r.contains("共線")),
            "{estimate:?}"
        );
    }

    /// 只有兩三張影格時不做推定。
    #[test]
    fn too_few_frames_never_infer_an_up_axis() {
        let images: Vec<Image> = (0..3)
            .map(|i| Image {
                id: i + 1,
                qvec: [1.0, 0.0, 0.0, 0.0],
                tvec: [-(i as f64), 0.0, 0.0],
                camera_id: 1,
                name: format!("f{i}.jpg"),
            })
            .collect();
        let estimate = infer_up_axis(&images);
        assert_eq!(estimate.source, UpAxisSource::WorldY);
        assert!(estimate.degenerate_reason.is_some());
    }

    /// 修剪參數不再被靜默夾住：`trim` 的驗證只在命令列的 `value_parser`（0.0..=0.2）一處。
    ///
    /// 舊版 `fit_box_from_xyz` 會 `trim.clamp(0.0, 0.2)`，所以 `trim = 0.9` 與 `0.2` 得到
    /// 完全一樣的結果——使用者以為自己調了參數，實際上什麼都沒變。
    /// 現在超範圍的值會給出退化的結果（這裡是「跨度不足」），而不是被悄悄改成合法值；
    /// 而且索引計算是飽和的，不會 panic。
    #[test]
    fn out_of_range_trim_is_not_silently_clamped() {
        let points: Vec<[f64; 3]> = (0..200)
            .map(|i| {
                let t = i as f64 / 199.0;
                [t * 10.0 - 5.0, t * 4.0 - 2.0, ((i % 9) as f64) * 0.5 - 2.0]
            })
            .collect();
        let images: Vec<Image> = (0..8)
            .map(|i| {
                let a = i as f64 * std::f64::consts::TAU / 8.0;
                Image {
                    id: i + 1,
                    qvec: [1.0, 0.0, 0.0, 0.0],
                    tvec: [-8.0 * a.cos(), -1.0, -8.0 * a.sin()],
                    camera_id: 1,
                    name: format!("f{i}.jpg"),
                }
            })
            .collect();

        let clamped = fit_box_from_xyz(
            &points,
            &images,
            &FitOptions {
                trim: 0.2,
                ..FitOptions::default()
            },
        )
        .expect("0.2 是合法值");
        assert!(clamped.size.iter().all(|s| *s > 0.0));

        // 0.9 以前會被夾成 0.2（結果相同）→ 現在必須是不同的結果（這裡是退化）
        let out_of_range = fit_box_from_xyz(
            &points,
            &images,
            &FitOptions {
                trim: 0.9,
                ..FitOptions::default()
            },
        );
        assert!(
            out_of_range.is_err(),
            "超範圍的 trim 不該被夾成 0.2 後成功：{out_of_range:?}"
        );
        // 邊界值本身合法
        assert!(fit_box_from_xyz(
            &points,
            &images,
            &FitOptions {
                trim: 0.0,
                ..FitOptions::default()
            }
        )
        .is_ok());
    }

    /// **關鍵回歸測試**：沿著一條街直走（真實拍攝最常見的軌跡之一）時，
    /// 舊版會回傳一個「偏離垂直 27.8°」的上軸，而且 `estimated = false`（自稱推定成功）。
    ///
    /// 實測（24 台相機，x 移動 10 m、y/z 各抖動 ±0.4 m）：
    ///   - 共變異數特徵值 = [218.5, 2.03, 0.34]；
    ///   - 真正的**最小主軸偏離垂直 49.5°**（這條軌跡根本沒有提供「上」的資訊）；
    ///   - 舊版冪迭代回傳的向量偏離垂直 27.8°，而它的自我檢查（`v^T·cov·v / trace = 0.003`）
    ///     遠低於 0.17 的門檻 → **靜默通過**。
    ///
    /// 新的判定比較「最小 ÷ 次小主軸變異量」（0.34/2.03 = 0.17 > 0.10）→ 明確退回世界 Y 軸並說明原因。
    #[test]
    fn straight_line_walk_never_silently_claims_an_up_axis() {
        let images: Vec<Image> = (0..24)
            .map(|i| {
                let t = i as f64 / 23.0;
                let x = 10.0 * t;
                let y = 0.4 * (t * 7.0).sin();
                let z = 0.4 * (t * 5.0).cos();
                Image {
                    id: i as u32 + 1,
                    qvec: [1.0, 0.0, 0.0, 0.0],
                    tvec: [-x, -y, -z],
                    camera_id: 1,
                    name: format!("f{i:02}.jpg"),
                }
            })
            .collect();
        let estimate = infer_up_axis(&images);
        assert_eq!(
            estimate.source,
            UpAxisSource::WorldY,
            "這種軌跡不該宣稱推定成功：{estimate:?}"
        );
        assert!(
            estimate.planarity_ratio > MAX_PLANARITY_RATIO,
            "應由最小/次小主軸的比例判定退化：{}",
            estimate.planarity_ratio
        );
        assert!(
            estimate
                .degenerate_reason
                .as_deref()
                .is_some_and(|r| r.contains("三維上散開")),
            "必須說明為什麼：{:?}",
            estimate.degenerate_reason
        );
    }

    /// 取捨的界線（實測記錄）：三維散開的軌跡（廣場繞行 10×6 m、上下起伏 ±1.2 m）
    /// 其真正的最小主軸剛好接近垂直（偏 2.9°），但比例 0.197 已超過門檻，因此本模組選擇
    /// **退回世界 Y 軸並警示**，而不是賭一把。
    ///
    /// 為什麼這個取捨是對的：同一組比例下也出現過偏離 49.5° 的軌跡（見上一個測試），
    /// 光憑特徵值無法分辨這兩種情況；而世界座標軸雖然不一定等於重力方向，
    /// 至少會被明確標成「推定失敗、可能不準」，使用者可以人工指定（`Manual`）或重新拍攝。
    /// 反過來說：默默給出 22° 偏掉的上軸（問題回報的實際情況），才是不能接受的。
    #[test]
    fn dispersed_orbit_prefers_an_honest_fallback_over_a_lucky_guess() {
        let images: Vec<Image> = (0..24)
            .map(|i| {
                let t = i as f64 / 23.0;
                Image {
                    id: i as u32 + 1,
                    qvec: [1.0, 0.0, 0.0, 0.0],
                    tvec: [
                        -(10.0 * t - 5.0),
                        -(1.2 * (t * 5.0 * std::f64::consts::TAU).sin()),
                        -(3.0 * (t * 2.0 * std::f64::consts::TAU).sin()),
                    ],
                    camera_id: 1,
                    name: format!("f{i:02}.jpg"),
                }
            })
            .collect();
        let estimate = infer_up_axis(&images);
        assert!(estimate.planarity_ratio > MAX_PLANARITY_RATIO);
        assert_eq!(estimate.source, UpAxisSource::WorldY);
        assert_eq!(estimate.up, [0.0, 1.0, 0.0]);
    }

    /// Jacobi 特徵分解本身：特徵值／特徵向量必須真的滿足 `A·v = λ·v`，
    /// 特徵向量必須兩兩正交且單位長，特徵值必須由大到小。
    ///
    /// 這是上軸推定唯一依賴的數學工具，因此用「直接驗算」而不是比對寫死的答案來測
    /// （後者只要排序或符號不同就會誤判）。
    #[test]
    fn jacobi_eigen_decomposition_is_exact_for_symmetric_matrices() {
        let matrices: [[[f64; 3]; 3]; 4] = [
            [[2.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 1.0]],
            [[4.0, 1.0, 2.0], [1.0, 5.0, 3.0], [2.0, 3.0, 6.0]],
            // 兩個特徵值幾乎相等（冪迭代會爛掉的形狀）
            [[1.0, 0.0, 0.0], [0.0, 1.0 + 1e-9, 0.0], [0.0, 0.0, 1e-4]],
            // 共變異數的實際形狀：水平面兩個方向接近相等、第三個方向極小
            [[768.0, 0.4, 0.02], [0.4, 770.0, -0.01], [0.02, -0.01, 2.0]],
        ];
        for matrix in matrices {
            let (values, vectors) = symmetric_eigen_decomposition(matrix);
            assert!(
                values[0] >= values[1] && values[1] >= values[2],
                "特徵值必須由大到小：{values:?}"
            );
            // 跡與行列式是不變量，可以用來確認特徵值沒有算錯
            let trace: f64 = (0..3).map(|i| matrix[i][i]).sum();
            assert!(
                (values.iter().sum::<f64>() - trace).abs() < 1e-6 * trace.abs().max(1.0),
                "特徵值總和必須等於跡：{values:?} vs {trace}"
            );
            for (k, v) in vectors.iter().enumerate() {
                let norm = dot(*v, *v).sqrt();
                assert!((norm - 1.0).abs() < 1e-9, "特徵向量必須是單位向量：{v:?}");
                // A·v 必須等於 λ·v
                for i in 0..3 {
                    let av = matrix[i][0] * v[0] + matrix[i][1] * v[1] + matrix[i][2] * v[2];
                    let lambda_v = values[k] * v[i];
                    assert!(
                        (av - lambda_v).abs() < 1e-6 * values[0].abs().max(1.0),
                        "A·v ≠ λ·v（第 {k} 個）：{av} vs {lambda_v}"
                    );
                }
            }
            for (i, j) in [(0usize, 1usize), (0, 2), (1, 2)] {
                assert!(
                    dot(vectors[i], vectors[j]).abs() < 1e-9,
                    "特徵向量必須兩兩正交：{} vs {}",
                    i,
                    j
                );
            }
        }
    }

    /// 由點雲擬合：一團沿 X 軸長 10、Y 軸高 3、Z 軸寬 4 的點 → 尺寸應接近 (10, 3, 4)（次序可能因 PCA 而換）。
    #[test]
    fn fits_box_from_point_cloud() {
        let mut points = Vec::new();
        for i in 0..20 {
            for j in 0..10 {
                for k in 0..10 {
                    let x = -5.0 + 10.0 * i as f64 / 19.0;
                    let y = -1.5 + 3.0 * j as f64 / 9.0;
                    let z = -2.0 + 4.0 * k as f64 / 9.0;
                    points.push([x, y, z]);
                }
            }
        }
        let images: Vec<Image> = (0..8)
            .map(|i| {
                let a = i as f64 * std::f64::consts::TAU / 8.0;
                Image {
                    id: i + 1,
                    qvec: [1.0, 0.0, 0.0, 0.0],
                    tvec: [-8.0 * a.cos(), -1.0, -8.0 * a.sin()],
                    camera_id: 1,
                    name: format!("f{i}.jpg"),
                }
            })
            .collect();
        let b = fit_box_from_xyz(&points, &images, &FitOptions::default()).expect("應可擬合");
        let mut size = b.size;
        size.sort_by(|a, c| c.partial_cmp(a).unwrap());
        // 最大邊 ≈10、中間 ≈4、最小 ≈3
        assert!((size[0] - 10.0).abs() < 0.6, "最長邊錯誤：{size:?}");
        assert!((size[1] - 4.0).abs() < 0.6, "次長邊錯誤：{size:?}");
        assert!((size[2] - 3.0).abs() < 0.6, "最短邊錯誤：{size:?}");
        // 中心應接近原點
        for c in b.center {
            assert!(c.abs() < 0.4, "中心偏移過大：{:?}", b.center);
        }
    }

    /// 點數太少時不應硬擬合，而且要說出真正的原因（不是籠統的「點雲不足」）。
    #[test]
    fn refuses_to_fit_with_too_few_points() {
        let err = fit_box_from_xyz(&[[0.0; 3]; 3], &[], &FitOptions::default())
            .expect_err("3 個點不足以擬合");
        assert_eq!(
            err,
            FitError::TooFewPoints {
                valid: 3,
                required: 8
            },
            "{err}"
        );
        // 完全沒有點是另一種原因
        assert_eq!(
            fit_box_from_xyz(&[], &[], &FitOptions::default()).unwrap_err(),
            FitError::NoPoints
        );
    }

    /// **關鍵回歸測試**：8 個完全重合的點不該「成功」擬合出 0.2 m 的立方體。
    ///
    /// 舊版唯一的退化判定是「點數 < 8」，所以 8 個（或一萬個）座標完全相同的點
    /// 會一路通過，最後由 `min_extent` 撐出一個 0.2 m 立方體並回報成功。
    /// 使用者拿到的是「有盒子」的假訊號，投影出來的框卻完全沒有意義。
    #[test]
    fn coincident_points_are_reported_as_degenerate() {
        let points = vec![[3.0, 4.0, 5.0]; 8];
        let err = fit_box_from_xyz(&points, &[], &FitOptions::default()).expect_err("應判為退化");
        match err {
            FitError::DegenerateExtent {
                max_extent,
                min_extent,
            } => {
                assert!(max_extent < 1e-9, "跨度應為 0，實際 {max_extent}");
                assert!((min_extent - 0.2).abs() < 1e-12);
            }
            other => panic!("原因應為跨度不足，實際 {other}"),
        }
        // 幾乎共線（跨度 1 mm）同樣不該給出盒子
        let mut thin = vec![[0.0; 3]; 12];
        for (i, p) in thin.iter_mut().enumerate() {
            p[0] = i as f64 * 0.001;
        }
        assert!(matches!(
            fit_box_from_xyz(&thin, &[], &FitOptions::default()),
            Err(FitError::DegenerateExtent { .. })
        ));
    }

    /// 非有限值佔多數時要明確拒絕：此時的「成功」只反映了少數點，卻會被當成整份點雲的結果。
    #[test]
    fn mostly_non_finite_points_are_rejected_with_counts() {
        let mut points: Vec<[f64; 3]> = (0..40).map(|i| [i as f64, 0.0, 0.0]).collect();
        points.extend(std::iter::repeat_n([f64::NAN, 0.0, 0.0], 60));
        let err = fit_box_from_xyz(&points, &[], &FitOptions::default()).unwrap_err();
        assert_eq!(
            err,
            FitError::MostlyNonFinite {
                total: 100,
                non_finite: 60
            },
            "{err}"
        );
        // 少數壞點仍可擬合（NaN 一律被剔除）
        let mut mostly_fine: Vec<[f64; 3]> = (0..100).map(|i| [i as f64 * 0.1, 1.0, 2.0]).collect();
        mostly_fine.push([f64::INFINITY, 0.0, 0.0]);
        let b = fit_box_from_xyz(&mostly_fine, &[], &FitOptions::default()).expect("應可擬合");
        assert!(b.center.iter().all(|v| v.is_finite()));
        assert_eq!(b.fitted_from_points, 100, "非有限值不該計入擬合點數");
    }

    /// 含離群點的點雲：修剪後尺寸不應被少數遠端點拉大。
    #[test]
    fn trimming_rejects_outliers() {
        let mut points: Vec<[f64; 3]> = (0..500)
            .map(|i| {
                let t = i as f64 / 499.0;
                [t * 4.0 - 2.0, t * 2.0 - 1.0, ((i % 7) as f64) * 0.2 - 0.6]
            })
            .collect();
        points.push([500.0, 500.0, 500.0]); // 離群點
        let images: Vec<Image> = (0..8)
            .map(|i| Image {
                id: i + 1,
                qvec: [1.0, 0.0, 0.0, 0.0],
                tvec: [-(i as f64), -1.0, -3.0],
                camera_id: 1,
                name: format!("f{i}.jpg"),
            })
            .collect();
        let b = fit_box_from_xyz(&points, &images, &FitOptions::default()).expect("應可擬合");
        let diagonal = (b.size[0].powi(2) + b.size[1].powi(2) + b.size[2].powi(2)).sqrt();
        assert!(diagonal < 12.0, "離群點未被修剪掉：尺寸 {:?}", b.size);
    }

    /// 摘要統計：拒絕原因應被聚合，警示則獨立統計（不可混進拒絕原因）。
    #[test]
    fn summary_aggregates_rejection_reasons() {
        let mk = |name: &str, ok: bool, reason: &str| FrameAnnotation {
            frame: name.into(),
            yolo: ok.then_some([0.5, 0.5, 0.2, 0.2]),
            pixel: None,
            visible_corners: if ok { 8 } else { 1 },
            area_ratio: 0.2,
            observed_points: if ok { 10 } else { 0 },
            accepted: ok,
            reason: (!ok).then(|| reason.to_string()),
            warning: None,
        };
        let list = vec![
            mk("a", true, ""),
            mk("b", false, "可見角點僅 1 個（需 ≥4）"),
            mk("c", false, "可見角點僅 0 個（需 ≥4）"),
            mk("d", false, "投影框過小（0.1% 畫面）"),
        ];
        let s = summarize(&list);
        assert_eq!((s.frames_total, s.frames_accepted), (4, 1));
        assert!((s.acceptance_ratio - 0.25).abs() < 1e-9);
        let corners = s
            .rejection_reasons
            .iter()
            .find(|(k, _)| k.contains("可見角點"))
            .map(|(_, v)| *v);
        assert_eq!(
            corners,
            Some(2),
            "同類原因應聚合：{:?}",
            s.rejection_reasons
        );
        assert_eq!(s.labels_removed, 0, "摘要本身不知道寫檔結果");
        assert!(s.warnings.is_empty());
    }

    /// 警示（會被採用的影格）必須與拒絕原因分開聚合，否則統計會出現假原因。
    #[test]
    fn warnings_are_counted_separately_from_rejections() {
        let warned = FrameAnnotation {
            frame: "fisheye.jpg".into(),
            yolo: Some([0.5, 0.5, 0.2, 0.2]),
            pixel: None,
            visible_corners: 8,
            area_ratio: 0.2,
            observed_points: 10,
            accepted: true,
            reason: None,
            warning: Some(
                "相機模型 OpenCvFisheye 為魚眼；本模組以針孔＋徑向畸變近似投影，框位可能偏移"
                    .into(),
            ),
        };
        let clean = FrameAnnotation {
            warning: None,
            ..warned.clone()
        };
        let s = summarize(&[warned, clean]);
        assert_eq!(s.frames_accepted, 2);
        assert!(s.rejection_reasons.is_empty(), "警示不是拒絕原因");
        assert_eq!(s.warnings.len(), 1);
        assert_eq!(s.warnings[0].1, 1);
        assert!(s.warnings[0].0.contains("魚眼"));
    }

    /// 寫出 YOLO 標籤：只有被採用的影格會產生檔案。
    #[test]
    fn writes_yolo_labels_only_for_accepted() {
        let dir = std::env::temp_dir().join(format!("macausplat-labels-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let a = FrameAnnotation {
            frame: "frame_0001.jpg".into(),
            yolo: Some([0.5, 0.4, 0.3, 0.6]),
            pixel: None,
            visible_corners: 8,
            area_ratio: 0.18,
            observed_points: 12,
            accepted: true,
            reason: None,
            warning: None,
        };
        let b = FrameAnnotation {
            frame: "frame_0002.jpg".into(),
            yolo: None,
            ..a.clone()
        };
        let report = write_yolo_labels(&dir, &[a, b], 5).expect("寫入應成功");
        assert_eq!(report.written, 1);
        assert_eq!(report.removed, 0, "第一次寫入沒有舊標籤");
        let content = std::fs::read_to_string(dir.join("frame_0001.txt")).unwrap();
        assert!(
            content.starts_with("5 0.500000 0.400000 0.300000 0.600000"),
            "內容錯誤：{content}"
        );
        assert!(!dir.join("frame_0002.txt").exists());
        assert!(!staging_directory(&dir).exists(), "暫存目錄應已被更名掉");
        let _ = std::fs::remove_dir_all(dir);
    }

    /// **關鍵回歸測試**：換參數重跑時，上一輪的舊標籤必須消失。
    ///
    /// 失效鏈：YOLO 的慣例是「沒有標籤檔 = 背景」。舊版只做 `create_dir_all` + 逐檔覆寫，
    /// 所以第一輪被採用、第二輪被 `--require-all-corners` 略過的影格，
    /// 它的舊 `.txt` 會留在目錄裡——對訓練來說那就是一筆**錯誤的正樣本**，而且毫無提示。
    #[test]
    fn rerunning_labels_removes_stale_ones_and_reports_the_difference() {
        let dir = std::env::temp_dir().join(format!("macausplat-rerun-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let mk = |name: &str, accepted: bool| FrameAnnotation {
            frame: format!("{name}.jpg"),
            yolo: accepted.then_some([0.5, 0.5, 0.2, 0.2]),
            pixel: None,
            visible_corners: 8,
            area_ratio: 0.2,
            observed_points: 10,
            accepted,
            reason: (!accepted).then(|| "嚴格模式要求 8 個角點全可見".to_string()),
            warning: None,
        };

        // 第一輪：兩張都採用
        let first = write_yolo_labels(&dir, &[mk("frame_0001", true), mk("frame_0002", true)], 0)
            .expect("寫入應成功");
        assert_eq!((first.written, first.removed), (2, 0));
        assert!(dir.join("frame_0002.txt").is_file());

        // 第二輪：嚴格的參數讓 frame_0002 不再被採用；舊標籤必須被清掉並回報
        let second = write_yolo_labels(&dir, &[mk("frame_0001", true), mk("frame_0002", false)], 0)
            .expect("寫入應成功");
        assert_eq!(second.written, 1);
        assert_eq!(second.removed, 2, "上一輪的兩個標籤都該被清掉");
        assert!(
            dir.join("frame_0001.txt").is_file(),
            "這一輪採用的影格仍要有標籤"
        );
        assert!(
            !dir.join("frame_0002.txt").exists(),
            "不再採用的影格不得留下舊標籤（否則等於憑空多出正樣本）"
        );
        // 這次重跑清掉的數量必須能被摘要帶出去（介面／QA 表看得到）
        let mut summary = summarize(&[mk("frame_0001", true), mk("frame_0002", false)]);
        summary.labels_removed = second.removed;
        assert_eq!(summary.labels_removed, 2);

        // 第三次：沒有任何影格被採用時，目錄要變成空的（而不是留著上一輪的標籤）
        let third = write_yolo_labels(&dir, &[mk("frame_0001", false)], 0).expect("寫入應成功");
        assert_eq!((third.written, third.removed), (0, 1));
        assert_eq!(
            std::fs::read_dir(&dir).unwrap().count(),
            0,
            "沒有任何影格被採用時，標籤目錄必須是空的"
        );
        let _ = std::fs::remove_dir_all(dir);
    }

    /// 寫標籤的暫存目錄名（`labels` → `labels.tmp`）。
    #[test]
    fn staging_directory_is_a_sibling_of_the_label_directory() {
        let staging = staging_directory(Path::new("out/labels"));
        assert_eq!(staging, Path::new("out/labels.tmp"));
    }

    /// CSV 轉義：檔名與原因含逗號或引號時，欄位數不能變、原文不能失真。
    #[test]
    fn csv_escapes_commas_and_quotes() {
        let dir = std::env::temp_dir().join(format!("macausplat-csv-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("annotations.csv");
        let annotations = vec![
            FrameAnnotation {
                // Windows 允許檔名包含逗號與雙引號
                frame: "frame,01\"a\".jpg".into(),
                yolo: Some([0.5, 0.5, 0.2, 0.2]),
                pixel: None,
                visible_corners: 8,
                area_ratio: 0.2,
                observed_points: 10,
                accepted: true,
                reason: None,
                // 半形逗號正是 CSV 的分隔符：警示欄也必須被引號包起來
                warning: Some("相機模型 OpenCvFisheye 為魚眼, 投影僅近似".into()),
            },
            FrameAnnotation {
                frame: "plain.jpg".into(),
                yolo: None,
                pixel: None,
                visible_corners: 1,
                area_ratio: 0.0,
                observed_points: 0,
                accepted: false,
                // 半形逗號正是 CSV 的分隔符：這種原因文字必須被引號包起來
                reason: Some("投影框過小, 且被遮擋".into()),
                warning: None,
            },
        ];
        write_annotations_csv(&path, &annotations).expect("寫入應成功");
        let csv = std::fs::read_to_string(&path).unwrap();
        let lines: Vec<&str> = csv.lines().collect();
        assert_eq!(lines.len(), 3, "標頭 + 兩列：{csv}");
        assert_eq!(lines[0].split(',').count(), 15, "欄位數：{}", lines[0]);
        // 逗號與引號都必須被包在引號裡，內部的 " 變成 ""
        assert!(
            lines[1].starts_with("\"frame,01\"\"a\"\".jpg\","),
            "檔名轉義錯誤：{}",
            lines[1]
        );
        // 原因原文照寫（不再被改成全形逗號），只是整欄被引號包住
        assert!(
            lines[2].contains("\"投影框過小, 且被遮擋\""),
            "原因應原文保留：{}",
            lines[2]
        );
        assert!(
            lines[1].contains("\"相機模型 OpenCvFisheye 為魚眼, 投影僅近似\""),
            "警示欄也必須轉義：{}",
            lines[1]
        );
        // 每一列的欄位數都要一致，否則試算表會整個歪掉
        for (i, line) in lines.iter().enumerate().skip(1) {
            assert_eq!(
                count_csv_fields(line),
                15,
                "第 {i} 列的欄位數不一致：{line}"
            );
        }
        let _ = std::fs::remove_dir_all(dir);
    }

    /// 依 RFC 4180 規則數一列 CSV 的欄位數（引號內的逗號不算分隔）。
    fn count_csv_fields(line: &str) -> usize {
        let mut fields = 1;
        let mut in_quotes = false;
        let mut chars = line.chars().peekable();
        while let Some(ch) = chars.next() {
            match ch {
                '"' if in_quotes && chars.peek() == Some(&'"') => {
                    chars.next();
                }
                '"' => in_quotes = !in_quotes,
                ',' if !in_quotes => fields += 1,
                _ => {}
            }
        }
        fields
    }

    /// HTML 轉義：檔名插進 `src` / `alt` 時不能提前結束屬性，也不能注入標記。
    #[test]
    fn qa_html_escapes_filenames_and_uses_the_given_frames_directory() {
        let dir = std::env::temp_dir().join(format!("macausplat-qa-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("qa.html");
        let annotations = vec![FrameAnnotation {
            frame: "a\"onerror=\"alert(1)<b>&'.jpg".into(),
            yolo: Some([0.5, 0.5, 0.2, 0.2]),
            pixel: None,
            visible_corners: 8,
            area_ratio: 0.2,
            observed_points: 10,
            accepted: true,
            reason: None,
            warning: Some("相機模型 Fov 的畸變項未實作".into()),
        }];
        let summary = summarize(&annotations);
        let box3d = Box3D::axis_aligned([0.0; 3], [1.0, 1.0, 1.0]);
        write_qa_html(
            &path,
            "../source/images",
            &annotations,
            &summary,
            &box3d,
            10,
        )
        .expect("寫入應成功");
        let html = std::fs::read_to_string(&path).unwrap();
        assert!(
            !html.contains("onerror=\"alert(1)"),
            "未轉義的引號會讓屬性提前結束：{html}"
        );
        assert!(html.contains("&quot;"), "雙引號必須被轉義");
        assert!(html.contains("&amp;"), "& 必須被轉義");
        assert!(html.contains("&lt;b&gt;"), "< 與 > 必須被轉義");
        assert!(html.contains("&#39;"), "單引號必須被轉義");
        // 呼叫端給的影格目錄必須照用（不再寫死 ../work/frames）
        assert!(
            html.contains("../source/images/a&quot;onerror="),
            "應使用呼叫端傳入的影格目錄：{html}"
        );
        assert!(!html.contains("../work/frames"));
        // 上軸來源的文字要跟著列舉走
        assert!(html.contains("操作者指定"), "應說明上軸來源：{html}");
        // 警示要出現在頁面上（被採用的影格也要讓人看到「投影只是近似」）
        assert!(html.contains("張影格有警示"), "警示應匯總在頁首：{html}");

        // 重跑清掉舊標籤時，差異必須顯示出來（否則使用者不知道資料集變了）
        let mut rerun = summary.clone();
        rerun.labels_removed = 7;
        write_qa_html(&path, "../source/images", &annotations, &rerun, &box3d, 10)
            .expect("寫入應成功");
        let html = std::fs::read_to_string(&path).unwrap();
        assert!(
            html.contains("清掉了 7 個上一輪的舊標籤檔"),
            "重跑清掉的標籤數必須可見：{html}"
        );
        let _ = std::fs::remove_dir_all(dir);
    }

    /// 上軸來源的三種語意與標籤文字。
    #[test]
    fn up_axis_source_explains_where_the_axis_came_from() {
        assert!(UpAxisSource::WorldY.is_estimated());
        assert!(!UpAxisSource::CameraTrajectory.is_estimated());
        assert!(!UpAxisSource::Manual.is_estimated());
        assert!(UpAxisSource::CameraTrajectory.label().contains("相機軌跡"));
        assert!(UpAxisSource::WorldY.label().contains("世界 Y 軸"));
        assert!(UpAxisSource::Manual.label().contains("操作者"));

        // 手動指定的軸對齊盒：來源是 Manual，而不是舊版寫死的「推定值」
        let manual = Box3D::axis_aligned([0.0; 3], [1.0; 3]);
        assert_eq!(manual.up_source, UpAxisSource::Manual);
        assert!(!manual.up_source.is_estimated());

        // `infer_up = false` 也是操作者的明確選擇 → Manual（不是「推定失敗」）
        let points: Vec<[f64; 3]> = (0..30)
            .map(|i| [i as f64 * 0.3, (i % 5) as f64 * 0.4, (i % 3) as f64 * 0.5])
            .collect();
        let manual = fit_box_from_xyz(
            &points,
            &[],
            &FitOptions {
                infer_up: false,
                ..FitOptions::default()
            },
        )
        .expect("應可擬合");
        assert_eq!(manual.up_source, UpAxisSource::Manual);
        assert_eq!(manual.axes[2], [0.0, 1.0, 0.0]);
    }

    /// 舊版 `box.json`（只有 `up_axis_estimated` 布林）仍必須能讀，且語意要對得上。
    #[test]
    fn legacy_box_json_still_loads_with_the_right_semantics() {
        let legacy = |estimated: bool| {
            format!(
                r#"{{"center":[1.0,2.0,3.0],"size":[4.0,5.0,6.0],
                    "axes":[[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,1.0]],
                    "up_axis_estimated":{estimated},"fitted_from_points":42}}"#
            )
        };
        let fallback: Box3D = serde_json::from_str(&legacy(true)).expect("舊格式應可讀");
        assert_eq!(fallback.up_source, UpAxisSource::WorldY);
        assert!(fallback.up_source.is_estimated());
        assert_eq!(fallback.fitted_from_points, 42);

        let inferred: Box3D = serde_json::from_str(&legacy(false)).expect("舊格式應可讀");
        assert_eq!(inferred.up_source, UpAxisSource::CameraTrajectory);

        // 完全沒有上軸欄位的檔案：保守地當成「可能不準」
        let bare = r#"{"center":[0.0,0.0,0.0],"size":[1.0,1.0,1.0],
            "axes":[[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,1.0]]}"#;
        let bare: Box3D = serde_json::from_str(bare).expect("缺少欄位也應可讀");
        assert_eq!(bare.up_source, UpAxisSource::WorldY);

        // 新格式往返
        let round_trip: Box3D =
            serde_json::from_str(&serde_json::to_string(&inferred).unwrap()).unwrap();
        assert_eq!(round_trip.up_source, UpAxisSource::CameraTrajectory);
        assert_eq!(round_trip.center, inferred.center);
        assert_eq!(round_trip.axes, inferred.axes);
    }

    /// 魚眼相機：投影只是近似，被採用的影格必須帶上警示（而不是默默寫出偏掉的框）。
    #[test]
    fn fisheye_cameras_are_flagged_as_approximate() {
        let image = Image {
            id: 1,
            qvec: [1.0, 0.0, 0.0, 0.0],
            tvec: [0.0, 0.0, 0.0],
            camera_id: 1,
            name: "fisheye.jpg".into(),
        };
        let b = Box3D::axis_aligned([0.0, 0.0, 10.0], [2.0, 2.0, 2.0]);
        let fisheye = Camera {
            id: 1,
            model: CameraModel::OpenCvFisheye,
            width: 1000,
            height: 1000,
            params: vec![1000.0, 1000.0, 500.0, 500.0, 0.0, 0.0, 0.0, 0.0],
        };
        let a = annotate_frame(&image, &fisheye, &b, &AnnotateOptions::default(), 10);
        assert!(a.accepted, "近似不代表不能用：{a:?}");
        let warning = a.warning.as_deref().expect("魚眼必須有警示");
        assert!(warning.contains("魚眼"), "警示要講清楚原因：{warning}");
        assert!(warning.contains("近似"), "{warning}");

        // 針孔相機不該有雜訊般的警示
        let clean = annotate_frame(&image, &camera(), &b, &AnnotateOptions::default(), 10);
        assert!(clean.warning.is_none(), "{clean:?}");
    }

    /// contains()：盒內／盒外判斷。
    #[test]
    fn contains_works_for_oriented_box() {
        let b = Box3D::axis_aligned([0.0, 0.0, 0.0], [2.0, 2.0, 2.0]);
        assert!(b.contains([0.9, 0.0, 0.0]));
        assert!(!b.contains([1.1, 0.0, 0.0]));
        assert!(b.contains([0.0, 0.0, 0.0]));
    }

    /// 用真實重建結構跑一次端到端（合成 3 台相機環繞一個盒）。
    #[test]
    fn annotate_reconstruction_end_to_end() {
        let mut cameras = std::collections::HashMap::new();
        cameras.insert(1, camera());
        let mut images = Vec::new();
        for i in 0..8 {
            let a = i as f64 * std::f64::consts::TAU / 8.0;
            // 相機位於半徑 8 的圓上、朝向原點。取繞 Y 軸旋轉 θ 的姿態：
            //   R_y(θ) = [[cosθ,0,sinθ],[0,1,0],[-sinθ,0,cosθ]]
            //   相機的 +Z 軸在世界座標為 R^T·(0,0,1) = (-sinθ, 0, cosθ)
            //   要讓它指向 -C/|C| = (-cos a, 0, -sin a)：
            //     -sinθ = -cos a → sinθ = cos a；cosθ = -sin a
            //   故 θ = atan2(cos a, -sin a)
            let (ca, sa) = (a.cos(), a.sin());
            let center = [8.0 * ca, 0.0, 8.0 * sa];
            let theta = f64::atan2(ca, -sa);
            let (st, ct) = (theta.sin(), theta.cos());
            let r = [[ct, 0.0, st], [0.0, 1.0, 0.0], [-st, 0.0, ct]];
            // t = -R * C
            let t = [
                -(r[0][0] * center[0] + r[0][1] * center[1] + r[0][2] * center[2]),
                -(r[1][0] * center[0] + r[1][1] * center[1] + r[1][2] * center[2]),
                -(r[2][0] * center[0] + r[2][1] * center[1] + r[2][2] * center[2]),
            ];
            // R_y(θ) 對應的四元數 (w, x, y, z) = (cos(θ/2), 0, sin(θ/2), 0)
            let half = theta / 2.0;
            let qvec = [half.cos(), 0.0, half.sin(), 0.0];
            // 先自我檢查：由 qvec/tvec 反推的相機中心必須等於預期位置
            let probe = Image {
                id: i as u32 + 1,
                qvec,
                tvec: t,
                camera_id: 1,
                name: String::new(),
            };
            let recovered = probe.camera_center();
            for k in 0..3 {
                assert!(
                    (recovered[k] - center[k]).abs() < 1e-6,
                    "測試夾具的相機姿態有誤：預期 {center:?}，反推 {recovered:?}"
                );
            }
            images.push(Image {
                id: i as u32 + 1,
                qvec,
                tvec: t,
                camera_id: 1,
                name: format!("frame_{i:04}.jpg"),
            });
        }
        // 盒內點：一立方體，每個點在每張影格的 track 都記錄 → observed_points = 8
        let points: Vec<Point3D> = (-1..=1)
            .flat_map(|x| (-1..=1).map(move |y| (x, y)))
            .flat_map(|(x, y)| (-1..=1).map(move |z| (x, y, z)))
            .map(|(x, y, z)| Point3D {
                id: 1,
                xyz: [x as f64, y as f64, z as f64],
                rgb: [1, 1, 1],
                track: (1..=8).map(|i| (i, 0)).collect(),
            })
            .collect();
        let recon = SparseReconstruction {
            cameras,
            images,
            points,
        };
        let b = fit_box_from_points(&recon, &FitOptions::default()).expect("應可擬合");
        let ann = annotate_reconstruction(&recon, &b, &AnnotateOptions::default());
        let s = summarize(&ann);
        assert_eq!(s.frames_total, 8);
        assert!(
            s.frames_accepted >= 6,
            "環繞一圈應有至少 6 張被採用，實際 {}/{}，拒絕原因 {:?}",
            s.frames_accepted,
            s.frames_total,
            s.rejection_reasons
        );
    }

    /// **關鍵回歸測試**：點雲中的 NaN 不得污染包圍盒，也不得寫出 NaN 標籤。
    ///
    /// 實測過的失效鏈：Rust 的 `f64::min/max` 會忽略 NaN，所以
    /// 「檢查 min_u.is_finite()」這類守門全失效；單一個 NaN 就讓包圍盒中心變 NaN，
    /// 而 size 看起來正常，最後寫出 `0 NaN 0.5 NaN 0.5` 這種標籤且沒有任何錯誤訊息。
    #[test]
    fn nan_points_never_reach_the_labels() {
        let mut points: Vec<[f64; 3]> = (0..600)
            .map(|i| {
                let t = i as f64 / 599.0;
                [t * 10.0 - 5.0, t * 4.0 - 2.0, ((i % 11) as f64) * 0.3 - 1.5]
            })
            .collect();
        points.push([f64::NAN, 0.0, 0.0]);
        points.push([0.0, f64::INFINITY, 0.0]);

        let images: Vec<Image> = (0..8)
            .map(|i| {
                let a = i as f64 * std::f64::consts::TAU / 8.0;
                Image {
                    id: i + 1,
                    qvec: [1.0, 0.0, 0.0, 0.0],
                    tvec: [-12.0 * a.cos(), -1.0, -12.0 * a.sin()],
                    camera_id: 1,
                    name: format!("f{i}.jpg"),
                }
            })
            .collect();

        let b = fit_box_from_xyz(&points, &images, &FitOptions::default()).expect("仍應可擬合");
        assert!(
            b.center.iter().all(|v| v.is_finite()),
            "包圍盒中心不得為 NaN/Inf：{:?}",
            b.center
        );
        assert!(
            b.size.iter().all(|v| v.is_finite() && *v > 0.0),
            "尺寸無效：{:?}",
            b.size
        );
        for axis in &b.axes {
            assert!(
                axis.iter().all(|v| v.is_finite()),
                "軸向含非有限值：{axis:?}"
            );
        }

        // 用這個盒產生標註，任何被採用的框都必須是有限值
        let image = Image {
            id: 1,
            qvec: [1.0, 0.0, 0.0, 0.0],
            tvec: [0.0, 0.0, 0.0],
            camera_id: 1,
            name: "f.jpg".into(),
        };
        let a = annotate_frame(&image, &camera(), &b, &AnnotateOptions::default(), 100);
        if let Some(y) = a.yolo {
            assert!(
                y.iter().all(|v| v.is_finite()),
                "YOLO 標籤含非有限值：{y:?}"
            );
        }

        // 即使有人手工造出 NaN 標籤，write_yolo_labels 也必須擋下
        let dir = std::env::temp_dir().join(format!("macausplat-nan-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let bad = FrameAnnotation {
            frame: "bad.jpg".into(),
            yolo: Some([f64::NAN, 0.5, 0.5, 0.5]),
            pixel: None,
            visible_corners: 8,
            area_ratio: 0.2,
            observed_points: 10,
            accepted: true,
            reason: None,
            warning: None,
        };
        let good = FrameAnnotation {
            frame: "good.jpg".into(),
            yolo: Some([0.5, 0.5, 0.2, 0.2]),
            ..bad.clone()
        };
        let report = write_yolo_labels(&dir, &[bad, good], 0).expect("寫入應成功");
        assert_eq!(report.written, 1, "NaN 標籤必須被略過");
        assert!(!dir.join("bad.txt").exists(), "不得產生含 NaN 的標籤檔");
        let content = std::fs::read_to_string(dir.join("good.txt")).unwrap();
        assert!(!content.contains("NaN"), "標籤內容不得含 NaN：{content}");
        let _ = std::fs::remove_dir_all(dir);
    }
}
