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
//! 一個水平面上，因此對相機中心做 PCA、取變異最小的方向即為上軸。若軌跡退化（例如原地旋轉、
//! 只有兩三個位置），則退回世界 Y 軸並在報告中標明「上軸為推定值」。
//!
//! ## 標註品質的自我檢查
//! 每一張影格都會計算：
//!   - 可見角點數（8 個角有幾個在相機前方且落在畫面內）
//!   - 投影面積佔畫面比例（太大或太小都不適合當訓練樣本）
//!   - **該影格實際觀測到的三維點數**（用 COLMAP 的 track 資料：有多少屬於這個包圍盒的點，
//!     其 track 包含這張影格）——這是遮擋與視角的客觀代理指標

use std::path::Path;

use serde::{Deserialize, Serialize};

use super::colmap::{camera_to_pixel, Camera, Image, SparseReconstruction};

/// 三維有向包圍盒。
///
/// `center` 為世界座標中心；`size` 為三個軸向的長度；`axes` 為三個單位向量（右手系），
/// 分別對應 size 的三個分量方向。角點 = center ± axes[i] * size[i] / 2。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Box3D {
    pub center: [f64; 3],
    pub size: [f64; 3],
    /// 三個正交單位軸：`axes[0]` 為長、`axes[1]` 為寬、`axes[2]` 為上（垂直方向）。
    pub axes: [[f64; 3]; 3],
    /// 上軸是否為推定值（true 表示軌跡退化、退回世界座標軸）。
    #[serde(default)]
    pub up_axis_estimated: bool,
    /// 擬合時使用的點數。
    #[serde(default)]
    pub fitted_from_points: usize,
}

impl Box3D {
    /// 軸對齊盒（快速手動指定用）。
    pub fn axis_aligned(center: [f64; 3], size: [f64; 3]) -> Self {
        Self {
            center,
            size,
            axes: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            up_axis_estimated: true,
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
                    let mut p = self.center;
                    for k in 0..3 {
                        p[k] += self.axes[0][k] * self.size[0] * sx
                            + self.axes[1][k] * self.size[1] * sy
                            + self.axes[2][k] * self.size[2] * sz;
                    }
                    out[i] = p;
                    i += 1;
                }
            }
        }
        out
    }

    /// 對角線長度（用來判斷與相機的距離是否合理）。
    pub fn diagonal(&self) -> f64 {
        (self.size[0].powi(2) + self.size[1].powi(2) + self.size[2].powi(2)).sqrt()
    }

    /// 點是否落在盒內。
    pub fn contains(&self, p: [f64; 3]) -> bool {
        for axis in 0..3 {
            let mut d = 0.0;
            for k in 0..3 {
                d += (p[k] - self.center[k]) * self.axes[axis][k];
            }
            if d.abs() > self.size[axis] / 2.0 {
                return false;
            }
        }
        true
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
        Self { trim: 0.02, infer_up: true, min_extent: 0.2 }
    }
}

/// 由相機中心推定上軸（PCA：變異最小的主方向）。
///
/// 回傳 `(up, estimated)`；`estimated = true` 表示軌跡退化、改用世界 Y 軸。
pub fn infer_up_axis(images: &[Image]) -> ([f64; 3], bool) {
    if images.len() < 4 {
        return ([0.0, 1.0, 0.0], true);
    }
    let centers: Vec<[f64; 3]> = images.iter().map(|i| i.camera_center()).collect();
    let n = centers.len() as f64;
    let mut mean = [0.0; 3];
    for c in &centers {
        for k in 0..3 {
            mean[k] += c[k] / n;
        }
    }
    // 3x3 共變異數
    let mut cov = [[0.0f64; 3]; 3];
    for c in &centers {
        let d = [c[0] - mean[0], c[1] - mean[1], c[2] - mean[2]];
        for i in 0..3 {
            for j in 0..3 {
                cov[i][j] += d[i] * d[j];
            }
        }
    }
    // 以冪迭代求最小特徵值對應的特徵向量（等價於對 (trace*I - cov) 求最大特徵向量）
    let trace = cov[0][0] + cov[1][1] + cov[2][2];
    let mut m = [[0.0f64; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            m[i][j] = if i == j { trace } else { 0.0 } - cov[i][j];
        }
    }
    let mut v = [0.3, 1.0, 0.2];
    for _ in 0..200 {
        let mut next = [0.0; 3];
        for i in 0..3 {
            for j in 0..3 {
                next[i] += m[i][j] * v[j];
            }
        }
        let norm = (next[0].powi(2) + next[1].powi(2) + next[2].powi(2)).sqrt();
        if norm < 1e-12 {
            return ([0.0, 1.0, 0.0], true);
        }
        v = [next[0] / norm, next[1] / norm, next[2] / norm];
    }
    // 檢查軌跡是否真的接近一個平面：若三個方向變異量接近，代表軌跡是三維分散的，上軸不可靠
    let mut eig = [0.0f64; 3];
    for i in 0..3 {
        let mut mv = [0.0; 3];
        for (a, row) in cov.iter().enumerate() {
            for (b, val) in row.iter().enumerate() {
                mv[a] += val * v[b];
            }
        }
        eig[i] = v[0] * mv[0] + v[1] * mv[1] + v[2] * mv[2];
        if i == 0 {
            break;
        }
    }
    let smallest = eig[0].max(0.0);
    let total = trace.max(1e-12);
    // 最小方向的變異若超過總變異的 1/6，表示軌跡不夠「平」，上軸不可靠
    if smallest / total > 0.17 {
        return ([0.0, 1.0, 0.0], true);
    }
    // 讓上軸與世界 +Y 同向（避免上下顛倒，讓高度為正）
    if v[1] < 0.0 {
        v = [-v[0], -v[1], -v[2]];
    }
    (v, false)
}

/// 由稀疏點雲擬合包圍盒。
///
/// 步驟：以 `up` 為垂直軸 → 把點投影到垂直平面上做 PCA 取得主方向 → 在三個方向上取修剪後的範圍。
pub fn fit_box_from_points(
    reconstruction: &SparseReconstruction,
    options: &FitOptions,
) -> Option<Box3D> {
    let points: Vec<[f64; 3]> = reconstruction.points.iter().map(|p| p.xyz).collect();
    fit_box_from_xyz(&points, &reconstruction.images, options)
}

/// 由座標陣列擬合（便於測試與外部重用）。
pub fn fit_box_from_xyz(points: &[[f64; 3]], images: &[Image], options: &FitOptions) -> Option<Box3D> {
    if points.len() < 8 {
        return None;
    }
    let (up, estimated) = if options.infer_up {
        infer_up_axis(images)
    } else {
        ([0.0, 1.0, 0.0], true)
    };
    // 建立與 up 正交的基底 (e1, e2)
    let helper = if up[0].abs() < 0.9 { [1.0, 0.0, 0.0] } else { [0.0, 0.0, 1.0] };
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

    let trim = options.trim.clamp(0.0, 0.2);
    let range = |axis: usize| -> (f64, f64) {
        let mut vals: Vec<f64> = rotated.iter().map(|p| p[axis]).collect();
        vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let lo = (vals.len() as f64 * trim) as usize;
        let hi = ((vals.len() as f64 * (1.0 - trim)) as usize).min(vals.len() - 1);
        (vals[lo], vals[hi])
    };
    let (u0, u1) = range(0);
    let (v0, v1) = range(1);
    let (w0, w1) = range(2);
    let size = [
        (u1 - u0).max(options.min_extent),
        (v1 - v0).max(options.min_extent),
        (w1 - w0).max(options.min_extent),
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
    Some(Box3D {
        center,
        size,
        axes: [ax0, ax1, up],
        up_axis_estimated: estimated,
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

/// 把包圍盒投影到單張影格。
pub fn annotate_frame(
    image: &Image,
    camera: &Camera,
    box3d: &Box3D,
    options: &AnnotateOptions,
    observed_points: usize,
) -> FrameAnnotation {
    let (w, h) = (camera.width as f64, camera.height as f64);
    let mut min_u = f64::INFINITY;
    let mut min_v = f64::INFINITY;
    let mut max_u = f64::NEG_INFINITY;
    let mut max_v = f64::NEG_INFINITY;
    let mut visible = 0usize;
    for corner in box3d.corners() {
        match camera_to_pixel(image.to_camera(corner), camera) {
            Some((u, v, _z)) => {
                if u >= -0.01 * w && u <= 1.01 * w && v >= -0.01 * h && v <= 1.01 * h {
                    visible += 1;
                }
                min_u = min_u.min(u);
                min_v = min_v.min(v);
                max_u = max_u.max(u);
                max_v = max_v.max(v);
            }
            None => {}
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
    if options.require_all_corners && visible < 8 {
        reason = Some(format!(
            "嚴格模式要求 8 個角點全可見，此影格僅 {visible} 個（框可能低估輪廓）"
        ));
    } else if visible < options.min_visible_corners {
        reason = Some(format!("可見角點僅 {visible} 個（需 ≥{}）", options.min_visible_corners));
    } else if area_ratio < options.min_area_ratio {
        reason = Some(format!("投影框過小（{:.1}% 畫面）", area_ratio * 100.0));
    } else if area_ratio > options.max_area_ratio {
        reason = Some(format!("投影框過大（{:.1}% 畫面，可能相機貼太近）", area_ratio * 100.0));
    } else if observed_points < options.min_observed_points {
        reason = Some(format!(
            "此影格觀測到的盒內點僅 {observed_points} 個（可能被遮擋）"
        ));
    }
    let accepted = reason.is_none();
    FrameAnnotation {
        frame: image.name.clone(),
        yolo: accepted.then(|| {
            [
                ((x0 + x1) / 2.0) / w,
                ((y0 + y1) / 2.0) / h,
                bw / w,
                bh / h,
            ]
        }),
        pixel: accepted.then_some([x0, y0, x1, y1]),
        visible_corners: visible,
        area_ratio,
        observed_points,
        accepted,
        reason,
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
}

/// 產生摘要統計（拒絕原因會按訊息聚合）。
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
    let mut reasons: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
    for a in annotations.iter().filter(|a| !a.accepted) {
        let key = a
            .reason
            .as_deref()
            .map(|r| {
                // 把數字換成 #，讓「可見角點僅 1 個」與「可見角點僅 0 個」歸為同一類原因
                let head = r.split('（').next().unwrap_or(r).trim();
                head.chars()
                    .map(|ch| if ch.is_ascii_digit() { '#' } else { ch })
                    .collect::<String>()
            })
            .unwrap_or_else(|| "未知".into());
        *reasons.entry(key).or_insert(0) += 1;
    }
    let mut rejection_reasons: Vec<(String, usize)> = reasons.into_iter().collect();
    rejection_reasons.sort_by(|a, b| b.1.cmp(&a.1));
    AnnotationSummary {
        frames_total: total,
        frames_accepted: accepted,
        acceptance_ratio: if total == 0 { 0.0 } else { accepted as f64 / total as f64 },
        median_visible_corners: med(annotations.iter().map(|a| a.visible_corners as f64).collect()),
        median_area_ratio: med(annotations.iter().map(|a| a.area_ratio).collect()),
        median_observed_points: med(annotations.iter().map(|a| a.observed_points as f64).collect()),
        rejection_reasons,
    }
}

/// 以 YOLO 格式寫出標籤（`<class_id> <cx> <cy> <w> <h>`，皆為 0~1 正規化）。
pub fn write_yolo_labels(
    dir: impl AsRef<Path>,
    annotations: &[FrameAnnotation],
    class_id: usize,
) -> std::io::Result<usize> {
    let dir = dir.as_ref();
    std::fs::create_dir_all(dir)?;
    let mut written = 0;
    for a in annotations.iter().filter(|a| a.accepted) {
        let Some(y) = a.yolo else { continue };
        let stem = Path::new(&a.frame)
            .file_stem()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_else(|| a.frame.clone());
        let line = format!("{class_id} {:.6} {:.6} {:.6} {:.6}\n", y[0], y[1], y[2], y[3]);
        std::fs::write(dir.join(format!("{stem}.txt")), line)?;
        written += 1;
    }
    Ok(written)
}

/// 寫出逐影格的標註明細（CSV，方便用試算表檢查）。
pub fn write_annotations_csv(path: impl AsRef<Path>, annotations: &[FrameAnnotation]) -> std::io::Result<()> {
    let mut out = String::from("frame,accepted,cx,cy,w,h,x0,y0,x1,y1,visible_corners,area_ratio,observed_points,reason\n");
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
        out.push_str(&format!(
            "{},{},{},{},{},{},{},{},{},{},{},{:.4},{},{}\n",
            a.frame,
            a.accepted,
            fmt(y[0]),
            fmt(y[1]),
            fmt(y[2]),
            fmt(y[3]),
            fmt(p[0]),
            fmt(p[1]),
            fmt(p[2]),
            fmt(p[3]),
            a.visible_corners,
            a.area_ratio,
            a.observed_points,
            a.reason.clone().unwrap_or_default().replace(',', "，"),
        ));
    }
    std::fs::write(path, out)
}

/// 產生人工複核用的 HTML 聯絡表（不需任何影像程式庫：圖片用 `<img>`，框用絕對定位的 `<div>`）。
///
/// `frames_dir` 為相對於 `out_path` 所在目錄的影格路徑前綴（例如 `../work/frames`）。
pub fn write_qa_html(
    out_path: impl AsRef<Path>,
    frames_dir: &str,
    annotations: &[FrameAnnotation],
    summary: &AnnotationSummary,
    box3d: &Box3D,
    max_frames: usize,
) -> std::io::Result<()> {
    let out_path = out_path.as_ref();
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
         .ok{color:#4ade80} .no{color:#f87171} .dim{color:#8fa3b8}\
         </style></head><body><header>\
         <h1>MacauSplat 標註複核表</h1><div class=\"sub\">",
    );
    html.push_str(&format!(
        "共 {} 張影格，採用 {} 張（{:.1}%）｜中位可見角點 {:.0}｜中位畫面佔比 {:.2}%｜中位觀測點數 {:.0}\
         ｜包圍盒 {:.2}×{:.2}×{:.2} m（上軸{}）",
        summary.frames_total,
        summary.frames_accepted,
        summary.acceptance_ratio * 100.0,
        summary.median_visible_corners,
        summary.median_area_ratio * 100.0,
        summary.median_observed_points,
        box3d.size[0],
        box3d.size[1],
        box3d.size[2],
        if box3d.up_axis_estimated { "為推定值" } else { "由相機軌跡推定" },
    ));
    html.push_str("</div></header><div class=\"grid\">");

    // 先顯示被採用的（人工複核的重點），再顯示被拒絕的樣本
    let mut ordered: Vec<&FrameAnnotation> = annotations.iter().filter(|a| a.accepted).collect();
    ordered.extend(annotations.iter().filter(|a| !a.accepted));
    for a in ordered.into_iter().take(max_frames) {
        let name = Path::new(&a.frame)
            .file_name()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_else(|| a.frame.clone());
        html.push_str("<div class=\"cell\"><div class=\"wrap\">");
        html.push_str(&format!("<img loading=\"lazy\" src=\"{frames_dir}/{name}\" alt=\"{name}\">"));
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
            html.push_str(&format!("<br><span class=\"no\">{r}</span>"));
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
            points2d: vec![],
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
        assert!((a.area_ratio - 0.0493).abs() < 0.005, "面積比應約 4.9%，實際 {}", a.area_ratio);
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
            points2d: vec![],
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
            points2d: vec![],
        };
        let far = Box3D::axis_aligned([0.0, 0.0, 5000.0], [2.0, 2.0, 2.0]);
        let a = annotate_frame(&image, &camera(), &far, &AnnotateOptions::default(), 10);
        assert!(!a.accepted && a.reason.as_deref().unwrap().contains("過小"), "{a:?}");

        // 「過大」的測法：盒子必須完整落在視錐內（8 個角都可見），但投影後佔滿畫面。
        // f=1000、畫面 1000px ⇒ 半視角 26.6°；邊長 8 的盒子放在 z=12.2 時，
        // 近面 (z=8.2) 的半寬投影為 4*1000/8.2 ≈ 488 px，框寬約 976 px（≈95% 畫面）→ 應判為「過大」。
        let near = Box3D::axis_aligned([0.0, 0.0, 12.2], [8.0, 8.0, 8.0]);
        let a = annotate_frame(&image, &camera(), &near, &AnnotateOptions::default(), 10);
        assert_eq!(a.visible_corners, 8, "8 個角都應在畫面內：{a:?}");
        assert!(!a.accepted && a.reason.as_deref().unwrap().contains("過大"), "{a:?}");

        // 反之，若盒子大到角點全部跑出畫面／跑到相機後方，會先被「可見角點」攔下——
        // 這是合理行為（連完整輪廓都看不到的影格不該當訓練樣本）。
        let absurd = Box3D::axis_aligned([0.0, 0.0, 1.0], [40.0, 40.0, 40.0]);
        let a = annotate_frame(&image, &camera(), &absurd, &AnnotateOptions::default(), 10);
        assert!(!a.accepted && a.reason.as_deref().unwrap().contains("可見角點"), "{a:?}");
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
            points2d: vec![],
        };
        let b = Box3D::axis_aligned([0.0, 0.0, 10.0], [2.0, 2.0, 2.0]);
        let a = annotate_frame(&image, &camera(), &b, &AnnotateOptions::default(), 0);
        assert!(!a.accepted && a.reason.as_deref().unwrap().contains("遮擋"), "{a:?}");
    }

    /// 上軸推定：相機繞同一水平圓環走一圈 → 上軸應接近世界 Y 軸且標記為「推定成功」。
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
                points2d: vec![],
            });
        }
        let (up, estimated) = infer_up_axis(&images);
        assert!(!estimated, "圓環軌跡應可推出上軸");
        assert!(up[1].abs() > 0.98, "上軸應接近世界 Y 軸，實際 {up:?}");
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
                    points2d: vec![],
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

    /// 點數太少時不應硬擬合。
    #[test]
    fn refuses_to_fit_with_too_few_points() {
        assert!(fit_box_from_xyz(&[[0.0; 3]; 3], &[], &FitOptions::default()).is_none());
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
                points2d: vec![],
            })
            .collect();
        let b = fit_box_from_xyz(&points, &images, &FitOptions::default()).expect("應可擬合");
        assert!(b.diagonal() < 12.0, "離群點未被修剪掉：尺寸 {:?}", b.size);
    }

    /// 摘要統計：拒絕原因應被聚合。
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
        let corners = s.rejection_reasons.iter().find(|(k, _)| k.contains("可見角點")).map(|(_, v)| *v);
        assert_eq!(corners, Some(2), "同類原因應聚合：{:?}", s.rejection_reasons);
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
        };
        let b = FrameAnnotation { frame: "frame_0002.jpg".into(), yolo: None, ..a.clone() };
        let n = write_yolo_labels(&dir, &[a, b], 5).expect("寫入應成功");
        assert_eq!(n, 1);
        let content = std::fs::read_to_string(dir.join("frame_0001.txt")).unwrap();
        assert!(content.starts_with("5 0.500000 0.400000 0.300000 0.600000"), "內容錯誤：{content}");
        assert!(!dir.join("frame_0002.txt").exists());
        let _ = std::fs::remove_dir_all(dir);
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
                points2d: vec![],
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
                points2d: vec![],
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
                error: 0.0,
                track: (1..=8).map(|i| (i, 0)).collect(),
            })
            .collect();
        let recon = SparseReconstruction { cameras, images, points };
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
}
