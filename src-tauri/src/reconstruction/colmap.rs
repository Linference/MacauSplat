//! COLMAP 稀疏重建（sparse reconstruction）讀取與相機投影。
//!
//! 為什麼需要這個模組：
//!     三維重建流程（COLMAP → Brush）完成後，專案目錄裡已經存有**每一張影格的相機位姿**
//!     與場景的稀疏三維點雲。這些資料正是「自動標註」的基礎：只要在三維空間定義建築的包圍盒，
//!     就能把它投影到每一張影格，得到幾何一致、零漏框的二維標註。
//!
//! 本模組只做兩件事，且都與既有引擎無關（純讀檔＋數學）：
//!     1. 解析 COLMAP 的二進位輸出（`cameras.bin` / `images.bin` / `points3D.bin`）
//!     2. 世界座標 → 相機座標 → 像素座標的投影（**針孔＋徑向／切向畸變**）
//!
//! ## 投影精度的界線（重要）
//! `camera_to_pixel` 實作的是針孔模型加上 Brown 型徑向／切向畸變。
//! COLMAP 的其他模型（魚眼、FOV、FullOpenCV 的高階項）**沒有**實作，
//! 對那些相機只能給出近似值。這件事由 `CameraModel::projection_is_approximate`
//! 明確表達，並由自動標註在影格上附帶警示——不能默默給出偏掉的框。
//!
//! ## 讀檔的兩個原則
//!   - **不解析不需要的欄位**：`images.bin` 每張影像的 `points2D`（每筆 24 位元組）在本專案沒有用途，
//!     因此只跳過（`Reader::skip`）而不進記憶體；但格式正確性仍要檢查（跳過時讀到檔尾即報錯）。
//!   - **容量一律先檢查**：檔案裡聲稱的數量是 u64，`Vec::with_capacity(u64::MAX as usize)` 會直接
//!     abort（無法用 `Result` 接住）。因此一律用 `checked_capacity` 以檔案長度反推上界。
//!
//! 座標約定（與 COLMAP 一致）：
//!     `X_cam = R * X_world + t`，其中 `R` 由四元數 `qvec = (w, x, y, z)` 轉換而來，
//!     `t = tvec`。相機朝向為 +Z，也就是深度必須為正才在相機前方。

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, Read};
use std::path::{Path, PathBuf};

/// COLMAP 相機模型（僅列出本模組支援的常見型號）。
///
/// `model_id` 與名稱的對應取自 COLMAP 的 `camera_models.h`；參數個數必須完全吻合，
/// 否則讀檔時會錯位——這是本模組最容易出錯的地方，因此每個型號都寫了單元測試。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CameraModel {
    /// id 0：`f, cx, cy`
    SimplePinhole,
    /// id 1：`fx, fy, cx, cy`
    Pinhole,
    /// id 2：`f, cx, cy, k`
    SimpleRadial,
    /// id 3：`f, cx, cy, k1, k2`
    Radial,
    /// id 4：`fx, fy, cx, cy, k1, k2, p1, p2`
    OpenCv,
    /// id 5：`fx, fy, cx, cy, k1, k2, k3, k4`
    ///
    /// 這是**魚眼**模型：COLMAP 對它的 k1~k4 是魚眼（等距）畸變多項式的係數，
    /// 與 Brown 徑向畸變不是同一回事。本模組的 `camera_to_pixel` 只會把它當成
    /// 「針孔＋徑向」處理，因此對這個模型算出的像素座標只是近似
    /// （見 `CameraModel::projection_is_approximate`）。
    OpenCvFisheye,
    /// id 6：`fx, fy, cx, cy, k1, k2, p1, p2, k3, k4, k5, k6`
    FullOpenCv,
    /// id 7：`fx, fy, cx, cy, omega`
    Fov,
    /// id 8：`f, cx, cy, k`
    SimpleRadialFisheye,
    /// id 9：`f, cx, cy, k1, k2`
    RadialFisheye,
    /// id 10：`fx, fy, cx, cy, k1, k2, p1, p2, k3, k4, sx1, sy1`
    ThinPrismFisheye,
}

impl CameraModel {
    /// 由 COLMAP 的 `model_id` 取得對應模型。
    pub fn from_id(id: i32) -> Option<Self> {
        Some(match id {
            0 => Self::SimplePinhole,
            1 => Self::Pinhole,
            2 => Self::SimpleRadial,
            3 => Self::Radial,
            4 => Self::OpenCv,
            5 => Self::OpenCvFisheye,
            6 => Self::FullOpenCv,
            7 => Self::Fov,
            8 => Self::SimpleRadialFisheye,
            9 => Self::RadialFisheye,
            10 => Self::ThinPrismFisheye,
            _ => return None,
        })
    }

    /// 該模型所需的參數個數（必須與 COLMAP 定義完全一致）。
    pub fn num_params(self) -> usize {
        match self {
            Self::SimplePinhole => 3,
            Self::Pinhole => 4,
            Self::SimpleRadial => 4,
            Self::Radial => 5,
            Self::OpenCv => 8,
            Self::OpenCvFisheye => 8,
            Self::FullOpenCv => 12,
            Self::Fov => 5,
            Self::SimpleRadialFisheye => 4,
            Self::RadialFisheye => 5,
            Self::ThinPrismFisheye => 12,
        }
    }

    /// 是否為魚眼類模型（畸變多項式為魚眼／等距形式，而非 Brown 徑向畸變）。
    pub fn is_fisheye(self) -> bool {
        matches!(
            self,
            Self::OpenCvFisheye
                | Self::SimpleRadialFisheye
                | Self::RadialFisheye
                | Self::ThinPrismFisheye
        )
    }

    /// 本模組的投影（針孔＋徑向／切向畸變）是否只能**近似**這個模型。
    ///
    /// 為什麼要與 `is_fisheye` 分開：兩者重疊但不相等。
    ///   - `Fov` 不是魚眼，但它有一個本模組未實作的 ω 畸變；
    ///   - `FullOpenCv` 也不是魚眼，而 `radial_distortion` 只取 k1、k2，k3~k6 被忽略；
    ///   - `ThinPrismFisheye` 的薄稜鏡項同樣未實作。
    ///
    /// 自動標註會用這個方法決定「這張影格的框要不要附帶近似警示」。
    pub fn projection_is_approximate(self) -> bool {
        self.is_fisheye() || matches!(self, Self::Fov | Self::FullOpenCv)
    }
}

/// 相機內參與尺寸。
#[derive(Debug, Clone)]
pub struct Camera {
    pub id: u32,
    pub model: CameraModel,
    pub width: u64,
    pub height: u64,
    pub params: Vec<f64>,
}

impl Camera {
    /// 取出 `(fx, fy, cx, cy)`，把各種模型統一成針孔形式。
    pub fn intrinsics(&self) -> (f64, f64, f64, f64) {
        let p = &self.params;
        match self.model {
            CameraModel::SimplePinhole
            | CameraModel::SimpleRadial
            | CameraModel::Radial
            | CameraModel::SimpleRadialFisheye
            | CameraModel::RadialFisheye => (p[0], p[0], p[1], p[2]),
            _ => (p[0], p[1], p[2], p[3]),
        }
    }

    /// 徑向畸變係數 `(k1, k2)`（沒有的話回傳 0）。
    pub fn radial_distortion(&self) -> (f64, f64) {
        let p = &self.params;
        match self.model {
            CameraModel::SimpleRadial | CameraModel::SimpleRadialFisheye => (p[3], 0.0),
            CameraModel::Radial | CameraModel::RadialFisheye => (p[3], p[4]),
            CameraModel::OpenCv | CameraModel::FullOpenCv | CameraModel::OpenCvFisheye => {
                (p[4], p[5])
            }
            _ => (0.0, 0.0),
        }
    }

    /// 切向畸變係數 `(p1, p2)`（沒有的話回傳 0）。
    pub fn tangential_distortion(&self) -> (f64, f64) {
        match self.model {
            CameraModel::OpenCv | CameraModel::FullOpenCv => (self.params[6], self.params[7]),
            _ => (0.0, 0.0),
        }
    }
}

/// 一張已註冊影格的姿態。
#[derive(Debug, Clone)]
pub struct Image {
    pub id: u32,
    /// 四元數 `(w, x, y, z)`。
    pub qvec: [f64; 4],
    pub tvec: [f64; 3],
    pub camera_id: u32,
    /// 相對於 `work/frames/` 的檔名（COLMAP 記錄的是它自己的相對路徑）。
    pub name: String,
}

impl Image {
    /// 由四元數取得旋轉矩陣（COLMAP 的 `qvec = (w, x, y, z)`）。
    pub fn rotation(&self) -> [[f64; 3]; 3] {
        quaternion_to_rotation(&self.qvec)
    }

    /// 世界座標 → 相機座標。
    pub fn to_camera(&self, world: [f64; 3]) -> [f64; 3] {
        let r = self.rotation();
        let t = self.tvec;
        [
            r[0][0] * world[0] + r[0][1] * world[1] + r[0][2] * world[2] + t[0],
            r[1][0] * world[0] + r[1][1] * world[1] + r[1][2] * world[2] + t[1],
            r[2][0] * world[0] + r[2][1] * world[1] + r[2][2] * world[2] + t[2],
        ]
    }

    /// 相機中心在世界座標的位置 `C = -R^T * t`（用於判斷影格與建築的相對關係）。
    pub fn camera_center(&self) -> [f64; 3] {
        let r = self.rotation();
        // R^T * t
        let rt = [
            r[0][0] * self.tvec[0] + r[1][0] * self.tvec[1] + r[2][0] * self.tvec[2],
            r[0][1] * self.tvec[0] + r[1][1] * self.tvec[1] + r[2][1] * self.tvec[2],
            r[0][2] * self.tvec[0] + r[1][2] * self.tvec[1] + r[2][2] * self.tvec[2],
        ];
        [-rt[0], -rt[1], -rt[2]]
    }
}

/// 稀疏點雲中的一個三維點。
///
/// 只保留本專案真的會用到的欄位：位置、顏色、可觀測列表（track）。
/// `error`（重投影誤差）在檔案裡仍然存在，但沒有任何地方讀取它，因此解析時直接跳過。
#[derive(Debug, Clone)]
pub struct Point3D {
    pub id: i64,
    pub xyz: [f64; 3],
    pub rgb: [u8; 3],
    /// `(image_id, point2D_idx)` 的可觀測列表，長度即 track length。
    pub track: Vec<(u32, u32)>,
}

/// 一個稀疏重建：相機、影格與三維點。
#[derive(Debug, Clone, Default)]
pub struct SparseReconstruction {
    pub cameras: HashMap<u32, Camera>,
    pub images: Vec<Image>,
    pub points: Vec<Point3D>,
}

impl SparseReconstruction {
    /// 由專案目錄載入稀疏重建。
    ///
    /// 會依序嘗試常見位置：`<root>/colmap/sparse/0`、`<root>/sparse/0`、`<root>/colmap`、`<root>`，
    /// 因此不論是 OOOSplat 的 `work/colmap` 或原始 COLMAP 輸出都能讀取。
    pub fn load(root: impl AsRef<Path>) -> Result<Self, ColmapError> {
        let root = root.as_ref();
        let candidates = [
            root.join("sparse").join("0"),
            root.join("colmap").join("sparse").join("0"),
            root.join("colmap"),
            root.to_path_buf(),
        ];
        let dir = candidates
            .iter()
            .find(|d| d.join("cameras.bin").is_file() && d.join("images.bin").is_file())
            .ok_or_else(|| ColmapError::MissingFiles(root.to_path_buf()))?
            .clone();
        Self::load_dir(&dir)
    }

    /// 由已確定含有 `.bin` 的目錄載入。
    pub fn load_dir(dir: impl AsRef<Path>) -> Result<Self, ColmapError> {
        let dir = dir.as_ref();
        let cameras = read_cameras_bin(&dir.join("cameras.bin"))?;
        let images = read_images_bin(&dir.join("images.bin"))?;
        let points_path = dir.join("points3D.bin");
        // 點雲可能不存在（例如只做了註冊），此時仍可進行投影標註，只是無法自動擬合包圍盒
        let points = if points_path.is_file() {
            read_points3d_bin(&points_path)?
        } else {
            Vec::new()
        };
        Ok(Self {
            cameras,
            images,
            points,
        })
    }
}

/// 把 COLMAP 記錄的檔名正規化，方便與影格檔名比對。
pub fn normalize_name(name: &str) -> String {
    name.replace('\\', "/")
        .rsplit('/')
        .next()
        .unwrap_or(name)
        .trim()
        .to_string()
}

#[derive(Debug, thiserror::Error)]
pub enum ColmapError {
    #[error("在 {0} 找不到 COLMAP 的 cameras.bin / images.bin")]
    MissingFiles(PathBuf),
    #[error("COLMAP 檔案格式不符（{file}）：{detail}")]
    Format { file: String, detail: String },
    #[error("COLMAP 使用了本模組尚未支援的相機模型 id={0}")]
    UnsupportedCameraModel(i32),
    #[error("檔案讀取失敗：{0}")]
    Io(#[from] std::io::Error),
}

// ---------------------------------------------------------------- 二進位讀取小工具

/// `images.bin` 中每一筆 `points2D` 的位元組數：`x: f64`、`y: f64`、`point3D_id: i64`。
const POINT2D_BYTES: u64 = 24;
/// `points3D.bin` 中每一筆 track entry 的位元組數：`image_id: u32`、`point2D_idx: u32`。
const TRACK_ENTRY_BYTES: u64 = 8;
/// 單一相機在 `cameras.bin` 中最少佔用的位元組數（`id+model+width+height` 加上最少 3 個參數）。
const MIN_CAMERA_BYTES: u64 = 4 + 4 + 8 + 8 + 3 * 8;
/// 單一影像在 `images.bin` 中最少佔用的位元組數（空檔名、零個特徵點）。
const MIN_IMAGE_BYTES: u64 = 4 + 4 * 8 + 3 * 8 + 4 + 1 + 8;
/// 單一三維點在 `points3D.bin` 中最少佔用的位元組數（零長度 track）。
const MIN_POINT_BYTES: u64 = 8 + 3 * 8 + 3 + 8 + 8;

/// 檢查檔頭聲稱的元素數量是否可能，並回傳可安全保留的容量。
///
/// 為什麼不能直接 `num as usize`：`num` 是檔案裡的 u64。檔案若損壞、或根本不是 COLMAP 的輸出，
/// 這個值可能是 `u64::MAX`；`Vec::with_capacity(u64::MAX as usize)` **配置失敗會直接 abort**，
/// 那是行程層級的死亡，`Result` 完全接不住（應用會直接消失，而不是顯示錯誤）。
///
/// 這裡用「檔案長度 ÷ 單一元素最少位元組」當上界：合法的檔案必定滿足這個不等式，
/// 因此不會誤擋正常檔案；超過上界就只可能是壞檔案，回 `ColmapError::Format`。
fn checked_capacity(
    reader: &Reader,
    num: u64,
    min_element_bytes: u64,
    what: &str,
) -> Result<usize, ColmapError> {
    /// 絕對上限：即使檔案真的很大，也不為超過此數量的元素預先配置記憶體
    /// （一億個元素已遠超任何合理的單一重建規模）。
    const MAX_ELEMENTS: u64 = 100_000_000;
    let bound = (reader.len / min_element_bytes.max(1)).min(MAX_ELEMENTS);
    if num > bound {
        return Err(ColmapError::Format {
            file: reader.path.display().to_string(),
            detail: format!(
                "{what}數量 {num} 與檔案長度 {} 不符（上限 {bound}），檔案可能損壞或不是 COLMAP 輸出",
                reader.len
            ),
        });
    }
    usize::try_from(num).map_err(|_| ColmapError::Format {
        file: reader.path.display().to_string(),
        detail: format!("{what}數量 {num} 超出本平台可表示的範圍"),
    })
}

struct Reader {
    inner: BufReader<File>,
    path: PathBuf,
    /// 開啟時的檔案長度（只用來檢查數量是否合理，`checked_capacity` 會用到）。
    len: u64,
}

impl Reader {
    fn open(path: &Path) -> Result<Self, ColmapError> {
        let file = File::open(path)?;
        let len = file.metadata()?.len();
        Ok(Self {
            inner: BufReader::new(file),
            path: path.to_path_buf(),
            len,
        })
    }

    fn take<const N: usize>(&mut self) -> Result<[u8; N], ColmapError> {
        let mut buf = [0u8; N];
        self.inner
            .read_exact(&mut buf)
            .map_err(|e| ColmapError::Format {
                file: self.path.display().to_string(),
                detail: format!("讀取 {N} 位元組失敗（檔案可能被截斷）：{e}"),
            })?;
        Ok(buf)
    }

    /// 跳過 `count` 個位元組。
    ///
    /// 為什麼不是 `seek`：`BufReader` 上的 seek 會丟掉整個緩衝區，之後每次讀取都要重新呼叫系統呼叫；
    /// 這裡改成「讀進固定大小的緩衝區再丟掉」——不配置與 `count` 等量的記憶體，
    /// 但**仍然讀到那些位元組**，因此檔案被截斷時會回 `Format` 錯誤，格式正確性不受影響。
    fn skip(&mut self, count: u64) -> Result<(), ColmapError> {
        let mut buffer = [0u8; 8192];
        let mut remaining = count;
        while remaining > 0 {
            let chunk = remaining.min(buffer.len() as u64) as usize;
            self.inner
                .read_exact(&mut buffer[..chunk])
                .map_err(|e| ColmapError::Format {
                    file: self.path.display().to_string(),
                    detail: format!("跳過 {count} 位元組時失敗（檔案可能被截斷）：{e}"),
                })?;
            remaining -= chunk as u64;
        }
        Ok(())
    }

    fn u32(&mut self) -> Result<u32, ColmapError> {
        Ok(u32::from_le_bytes(self.take::<4>()?))
    }

    fn i32(&mut self) -> Result<i32, ColmapError> {
        Ok(i32::from_le_bytes(self.take::<4>()?))
    }

    fn u64(&mut self) -> Result<u64, ColmapError> {
        Ok(u64::from_le_bytes(self.take::<8>()?))
    }

    fn i64(&mut self) -> Result<i64, ColmapError> {
        Ok(i64::from_le_bytes(self.take::<8>()?))
    }

    fn f64(&mut self) -> Result<f64, ColmapError> {
        Ok(f64::from_le_bytes(self.take::<8>()?))
    }

    fn u8(&mut self) -> Result<u8, ColmapError> {
        Ok(self.take::<1>()?[0])
    }

    /// 讀取以 NUL 結尾的字串（COLMAP 的檔名）。
    fn string(&mut self) -> Result<String, ColmapError> {
        let mut bytes = Vec::new();
        loop {
            let b = self.u8()?;
            if b == 0 {
                break;
            }
            bytes.push(b);
            if bytes.len() > 4096 {
                return Err(ColmapError::Format {
                    file: self.path.display().to_string(),
                    detail: "字串長度異常（超過 4096 位元組）".into(),
                });
            }
        }
        Ok(String::from_utf8_lossy(&bytes).into_owned())
    }
}

/// 讀取 `cameras.bin`。
pub fn read_cameras_bin(path: &Path) -> Result<HashMap<u32, Camera>, ColmapError> {
    let mut r = Reader::open(path)?;
    let num = r.u64()?;
    let mut out = HashMap::with_capacity(checked_capacity(&r, num, MIN_CAMERA_BYTES, "相機")?);
    for _ in 0..num {
        let id = r.u32()?;
        let model_id = r.i32()?;
        let model =
            CameraModel::from_id(model_id).ok_or(ColmapError::UnsupportedCameraModel(model_id))?;
        let width = r.u64()?;
        let height = r.u64()?;
        let mut params = Vec::with_capacity(model.num_params());
        for _ in 0..model.num_params() {
            params.push(r.f64()?);
        }
        out.insert(
            id,
            Camera {
                id,
                model,
                width,
                height,
                params,
            },
        );
    }
    Ok(out)
}

/// 讀取 `images.bin`。
///
/// **不解析**每張影像的 `points2D`：本專案的自動標註只需要相機位姿，而一張 1080p 影格可能有
/// 數千個特徵點——原本為每個點做一次 `Vec::push`，等於每兩秒（點雲檢視器的刷新間隔）
/// 就把整份特徵點表重新配置一次，卻沒有任何地方讀它。
/// 位元組仍然被完整跳過（`Reader::skip`），所以格式正確性與截斷檢查都保留。
pub fn read_images_bin(path: &Path) -> Result<Vec<Image>, ColmapError> {
    let mut r = Reader::open(path)?;
    let num = r.u64()?;
    let mut out = Vec::with_capacity(checked_capacity(&r, num, MIN_IMAGE_BYTES, "影像")?);
    for _ in 0..num {
        let id = r.u32()?;
        let mut qvec = [0.0; 4];
        for v in qvec.iter_mut() {
            *v = r.f64()?;
        }
        let mut tvec = [0.0; 3];
        for v in tvec.iter_mut() {
            *v = r.f64()?;
        }
        let camera_id = r.u32()?;
        let name = r.string()?;
        let points2d = r.u64()?;
        let bytes = points2d
            .checked_mul(POINT2D_BYTES)
            .ok_or_else(|| ColmapError::Format {
                file: path.display().to_string(),
                detail: format!("影像 {id} 的特徵點數 {points2d} 不合理（位元組數溢位）"),
            })?;
        r.skip(bytes)?;
        out.push(Image {
            id,
            qvec,
            tvec,
            camera_id,
            name,
        });
    }
    Ok(out)
}

/// 讀取 `points3D.bin`。
///
/// 每點的 `error`（重投影誤差，f64）在本專案沒有用途，因此跳過不存。
pub fn read_points3d_bin(path: &Path) -> Result<Vec<Point3D>, ColmapError> {
    let mut r = Reader::open(path)?;
    let num = r.u64()?;
    let mut out = Vec::with_capacity(checked_capacity(&r, num, MIN_POINT_BYTES, "三維點")?);
    for _ in 0..num {
        let id = r.i64()?;
        let mut xyz = [0.0; 3];
        for v in xyz.iter_mut() {
            *v = r.f64()?;
        }
        let rgb = [r.u8()?, r.u8()?, r.u8()?];
        r.skip(8)?; // error: f64（未使用）
        let track_len = r.u64()?;
        let mut track =
            Vec::with_capacity(checked_capacity(&r, track_len, TRACK_ENTRY_BYTES, "track")?);
        for _ in 0..track_len {
            let image_id = r.u32()?;
            let point2d_idx = r.u32()?;
            track.push((image_id, point2d_idx));
        }
        out.push(Point3D {
            id,
            xyz,
            rgb,
            track,
        });
    }
    Ok(out)
}

// ---------------------------------------------------------------- 數學
/// 四元數 `(w, x, y, z)` → 旋轉矩陣。
pub fn quaternion_to_rotation(q: &[f64; 4]) -> [[f64; 3]; 3] {
    let (w, x, y, z) = (q[0], q[1], q[2], q[3]);
    let n = (w * w + x * x + y * y + z * z).sqrt();
    if n <= f64::EPSILON {
        return [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    }
    let (w, x, y, z) = (w / n, x / n, y / n, z / n);
    [
        [
            1.0 - 2.0 * y * y - 2.0 * z * z,
            2.0 * x * y - 2.0 * z * w,
            2.0 * x * z + 2.0 * y * w,
        ],
        [
            2.0 * x * y + 2.0 * z * w,
            1.0 - 2.0 * x * x - 2.0 * z * z,
            2.0 * y * z - 2.0 * x * w,
        ],
        [
            2.0 * x * z - 2.0 * y * w,
            2.0 * y * z + 2.0 * x * w,
            1.0 - 2.0 * x * x - 2.0 * y * y,
        ],
    ]
}

/// 相機座標 → 像素座標（針孔模型＋Brown 徑向／切向畸變）。
///
/// 回傳 `None` 表示該點在相機後方（`z <= 0`），也就是不可見。
///
/// **這個函式對非針孔模型只是近似**：請先看 `CameraModel::projection_is_approximate`，
/// 需要對使用者誠實說明框位可能偏移時（自動標註就是如此），必須把近似講出來。
pub fn camera_to_pixel(point: [f64; 3], camera: &Camera) -> Option<(f64, f64, f64)> {
    let (z, x, y) = (point[2], point[0], point[1]);
    if !z.is_finite() || z <= 1e-9 {
        return None;
    }
    let (fx, fy, cx, cy) = camera.intrinsics();
    let (mut a, mut b) = (x / z, y / z);
    let (k1, k2) = camera.radial_distortion();
    if k1 != 0.0 || k2 != 0.0 {
        let r2 = a * a + b * b;
        let radial = 1.0 + k1 * r2 + k2 * r2 * r2;
        a *= radial;
        b *= radial;
    }
    let (p1, p2) = camera.tangential_distortion();
    if p1 != 0.0 || p2 != 0.0 {
        let r2 = a * a + b * b;
        let (ax, ay) = (a, b);
        a = ax + 2.0 * p1 * ax * ay + p2 * (r2 + 2.0 * ax * ax);
        b = ay + p1 * (r2 + 2.0 * ay * ay) + 2.0 * p2 * ax * ay;
    }
    Some((fx * a + cx, fy * b + cy, z))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn temp_path(name: &str) -> PathBuf {
        let mut p = std::env::temp_dir();
        p.push(format!("macausplat-test-{}-{}", std::process::id(), name));
        p
    }

    /// 檢查四元數轉旋轉矩陣：單位四元數 → 單位矩陣；90° 繞 Z 軸 → 已知矩陣。
    #[test]
    fn quaternion_rotation_is_correct() {
        let identity = quaternion_to_rotation(&[1.0, 0.0, 0.0, 0.0]);
        for (i, row) in identity.iter().enumerate() {
            for (j, v) in row.iter().enumerate() {
                assert!(
                    (v - if i == j { 1.0 } else { 0.0 }).abs() < 1e-12,
                    "identity 錯誤於 ({i},{j})"
                );
            }
        }
        // 繞 Z 軸 +90°：(x,y) → (-y,x)。四元數為 (cos(θ/2), 0, 0, sin(θ/2))，θ=90° → θ/2=45°
        let half = std::f64::consts::FRAC_PI_4;
        let rot = quaternion_to_rotation(&[half.cos(), 0.0, 0.0, half.sin()]);
        let p = [1.0, 0.0, 0.0];
        let x = rot[0][0] * p[0] + rot[0][1] * p[1] + rot[0][2] * p[2];
        let y = rot[1][0] * p[0] + rot[1][1] * p[1] + rot[1][2] * p[2];
        assert!(
            x.abs() < 1e-9 && (y - 1.0).abs() < 1e-9,
            "繞 Z 軸旋轉結果錯誤：({x}, {y})"
        );
    }

    /// 相機在正面 5 公尺、朝 +Z 時，畫面中心的點應投影到 (cx, cy)。
    #[test]
    fn projection_hits_principal_point() {
        let camera = Camera {
            id: 1,
            model: CameraModel::Pinhole,
            width: 640,
            height: 480,
            params: vec![500.0, 500.0, 320.0, 240.0],
        };
        let image = Image {
            id: 1,
            qvec: [1.0, 0.0, 0.0, 0.0],
            tvec: [0.0, 0.0, 0.0],
            camera_id: 1,
            name: "a.jpg".into(),
        };
        let (u, v, z) =
            camera_to_pixel(image.to_camera([0.0, 0.0, 5.0]), &camera).expect("應可投影");
        assert!((u - 320.0).abs() < 1e-9 && (v - 240.0).abs() < 1e-9 && (z - 5.0).abs() < 1e-9);
        // 相機後方的點不可見
        assert!(camera_to_pixel(image.to_camera([0.0, 0.0, -5.0]), &camera).is_none());
    }

    /// 投影近似性：哪些模型只能近似、哪些是真的針孔。
    ///
    /// 這是標註警示的依據，因此逐個模型列出期望值；`FullOpenCv` 也被視為近似，
    /// 因為 `radial_distortion` 只取 k1、k2，k3~k6 從未被使用。
    #[test]
    fn flags_models_whose_projection_is_only_approximate() {
        use CameraModel::*;
        for model in [
            OpenCvFisheye,
            SimpleRadialFisheye,
            RadialFisheye,
            ThinPrismFisheye,
        ] {
            assert!(model.is_fisheye(), "{model:?} 應被視為魚眼");
            assert!(
                model.projection_is_approximate(),
                "{model:?} 的投影只是近似，必須被標記"
            );
        }
        for model in [Fov, FullOpenCv] {
            assert!(!model.is_fisheye(), "{model:?} 不是魚眼");
            assert!(
                model.projection_is_approximate(),
                "{model:?} 有未實作的畸變項，必須被標記為近似"
            );
        }
        for model in [SimplePinhole, Pinhole, SimpleRadial, Radial, OpenCv] {
            assert!(
                !model.projection_is_approximate(),
                "{model:?} 是完整實作的模型，不該被標記為近似"
            );
        }
    }

    /// 相機中心公式：t = 0 時相機在世界原點；t 平移後位置應反向。
    #[test]
    fn camera_center_follows_translation() {
        let mut image = Image {
            id: 1,
            qvec: [1.0, 0.0, 0.0, 0.0],
            tvec: [1.0, 2.0, 3.0],
            camera_id: 1,
            name: "a.jpg".into(),
        };
        let c = image.camera_center();
        assert!(
            (c[0] + 1.0).abs() < 1e-12 && (c[1] + 2.0).abs() < 1e-12 && (c[2] + 3.0).abs() < 1e-12
        );
        image.tvec = [0.0, 0.0, 0.0];
        let c = image.camera_center();
        assert!(c.iter().all(|v| v.abs() < 1e-12));
    }

    /// 寫出一份最小的 cameras.bin / images.bin / points3D.bin，再讀回來比對。
    /// 這是本模組最容易出錯的地方（欄位順序與型別），所以用真實二進位做往返測試。
    ///
    /// `images.bin` 刻意寫**兩張**影像、且第一張帶有 2 個 2D 特徵點：
    /// 本模組跳過那些位元組而不解析，若跳過的長度算錯，第二張影像的欄位就會整組錯位，
    /// 這個測試因此同時守住「跳過」的正確性。
    #[test]
    fn binary_round_trip_matches_colmap_layout() {
        let cam_path = temp_path("cameras.bin");
        let img_path = temp_path("images.bin");
        let pts_path = temp_path("points3D.bin");

        // cameras.bin：1 台 PINHOLE（fx, fy, cx, cy）
        let mut f = File::create(&cam_path).unwrap();
        f.write_all(&1u64.to_le_bytes()).unwrap();
        f.write_all(&7u32.to_le_bytes()).unwrap(); // camera_id
        f.write_all(&1i32.to_le_bytes()).unwrap(); // model_id = PINHOLE
        f.write_all(&1920u64.to_le_bytes()).unwrap(); // width
        f.write_all(&1080u64.to_le_bytes()).unwrap(); // height
        for v in [1000.0f64, 1000.0, 960.0, 540.0] {
            f.write_all(&v.to_le_bytes()).unwrap();
        }
        drop(f);

        // images.bin：2 張影格；第一張有 2 個 2D 點（其中一個 point3D_id = -1），第二張 0 個。
        let mut f = File::create(&img_path).unwrap();
        f.write_all(&2u64.to_le_bytes()).unwrap();
        f.write_all(&3u32.to_le_bytes()).unwrap(); // image_id
        for v in [1.0f64, 0.0, 0.0, 0.0] {
            f.write_all(&v.to_le_bytes()).unwrap();
        }
        for v in [0.1f64, 0.2, 0.3] {
            f.write_all(&v.to_le_bytes()).unwrap();
        }
        f.write_all(&7u32.to_le_bytes()).unwrap(); // camera_id
        f.write_all(b"frame_0001.jpg\0").unwrap();
        f.write_all(&2u64.to_le_bytes()).unwrap();
        f.write_all(&12.5f64.to_le_bytes()).unwrap();
        f.write_all(&34.5f64.to_le_bytes()).unwrap();
        f.write_all(&99i64.to_le_bytes()).unwrap();
        f.write_all(&50.0f64.to_le_bytes()).unwrap();
        f.write_all(&60.0f64.to_le_bytes()).unwrap();
        f.write_all(&(-1i64).to_le_bytes()).unwrap();
        f.write_all(&4u32.to_le_bytes()).unwrap(); // image_id
        for v in [0.0f64, 1.0, 0.0, 0.0] {
            f.write_all(&v.to_le_bytes()).unwrap();
        }
        for v in [-1.0f64, -2.0, -3.0] {
            f.write_all(&v.to_le_bytes()).unwrap();
        }
        f.write_all(&7u32.to_le_bytes()).unwrap(); // camera_id
        f.write_all(b"frame_0002.jpg\0").unwrap();
        f.write_all(&0u64.to_le_bytes()).unwrap();
        drop(f);

        // points3D.bin：1 個點，track 長度 1
        let mut f = File::create(&pts_path).unwrap();
        f.write_all(&1u64.to_le_bytes()).unwrap();
        f.write_all(&99i64.to_le_bytes()).unwrap();
        for v in [1.5f64, -2.5, 3.5] {
            f.write_all(&v.to_le_bytes()).unwrap();
        }
        f.write_all(&[10u8, 20, 30]).unwrap();
        f.write_all(&0.25f64.to_le_bytes()).unwrap();
        f.write_all(&1u64.to_le_bytes()).unwrap();
        f.write_all(&3u32.to_le_bytes()).unwrap();
        f.write_all(&1u32.to_le_bytes()).unwrap();
        drop(f);

        let cameras = read_cameras_bin(&cam_path).expect("cameras.bin 應可讀取");
        let cam = cameras.get(&7).expect("camera_id 7 應存在");
        assert_eq!(cam.model, CameraModel::Pinhole);
        assert_eq!((cam.width, cam.height), (1920, 1080));
        assert_eq!(cam.intrinsics(), (1000.0, 1000.0, 960.0, 540.0));

        let images = read_images_bin(&img_path).expect("images.bin 應可讀取");
        assert_eq!(images.len(), 2);
        assert_eq!(images[0].id, 3);
        assert_eq!(images[0].camera_id, 7);
        assert_eq!(images[0].name, "frame_0001.jpg");
        assert_eq!(images[0].qvec, [1.0, 0.0, 0.0, 0.0]);
        assert_eq!(images[0].tvec, [0.1, 0.2, 0.3]);
        // 第二張影像的欄位必須完全正確——這是「跳過 2D 特徵點」沒有算錯長度的證據
        assert_eq!(images[1].id, 4);
        assert_eq!(images[1].name, "frame_0002.jpg");
        assert_eq!(images[1].qvec, [0.0, 1.0, 0.0, 0.0]);
        assert_eq!(images[1].tvec, [-1.0, -2.0, -3.0]);

        let points = read_points3d_bin(&pts_path).expect("points3D.bin 應可讀取");
        assert_eq!(points.len(), 1);
        assert_eq!(points[0].id, 99);
        assert_eq!(points[0].xyz, [1.5, -2.5, 3.5]);
        assert_eq!(points[0].rgb, [10, 20, 30]);
        assert_eq!(points[0].track, vec![(3, 1)]);

        let _ = std::fs::remove_file(cam_path);
        let _ = std::fs::remove_file(img_path);
        let _ = std::fs::remove_file(pts_path);
    }

    /// 檔案截斷時應回報格式錯誤，而不是 panic 或讀出垃圾資料。
    #[test]
    fn truncated_file_reports_error() {
        let path = temp_path("truncated.bin");
        let mut f = File::create(&path).unwrap();
        f.write_all(&2u64.to_le_bytes()).unwrap(); // 聲稱有 2 台相機，但後面沒資料
        drop(f);
        let err = read_cameras_bin(&path).expect_err("應報錯");
        assert!(
            matches!(err, ColmapError::Format { .. }),
            "錯誤型別應為 Format，實際：{err:?}"
        );
        let _ = std::fs::remove_file(path);
    }

    /// **關鍵回歸測試**：檔頭聲稱的數量離譜時必須回錯誤，而不是配置記憶體而 abort。
    ///
    /// 失效鏈：`num` 直接來自檔案的 u64。損壞的檔案（或根本不是 COLMAP 輸出的檔案）
    /// 只要前 8 位元組剛好是非零的大數，`Vec::with_capacity(num as usize)` 就會嘗試配置
    /// 天文數字般的記憶體。**配置失敗是 abort，`Result` 接不住**——UI 不會顯示錯誤，而是整個行程消失。
    /// 因此數量必須先與檔案長度交叉檢查。
    #[test]
    fn absurd_element_counts_are_rejected_without_allocating() {
        let cam_path = temp_path("absurd-cameras.bin");
        let mut f = File::create(&cam_path).unwrap();
        f.write_all(&u64::MAX.to_le_bytes()).unwrap();
        drop(f);
        let err = read_cameras_bin(&cam_path).expect_err("u64::MAX 台相機不可能存在");
        assert!(
            matches!(err, ColmapError::Format { .. }),
            "錯誤型別應為 Format，實際：{err:?}"
        );

        // 數量「看似合理」但遠超檔案長度所能容納 → 一樣要擋
        let mut f = File::create(&cam_path).unwrap();
        f.write_all(&10_000_000u64.to_le_bytes()).unwrap();
        drop(f);
        assert!(read_cameras_bin(&cam_path).is_err());

        // images.bin 與 points3D.bin 走同一條路徑，也要擋
        let img_path = temp_path("absurd-images.bin");
        let mut f = File::create(&img_path).unwrap();
        f.write_all(&u64::MAX.to_le_bytes()).unwrap();
        drop(f);
        assert!(read_images_bin(&img_path).is_err());

        let pts_path = temp_path("absurd-points.bin");
        let mut f = File::create(&pts_path).unwrap();
        f.write_all(&u64::MAX.to_le_bytes()).unwrap();
        drop(f);
        assert!(read_points3d_bin(&pts_path).is_err());

        // track 長度同樣來自檔案，也要擋（一個點宣稱有 u64::MAX 筆觀測）
        let track_path = temp_path("absurd-track.bin");
        let mut f = File::create(&track_path).unwrap();
        f.write_all(&1u64.to_le_bytes()).unwrap(); // 1 個點
        f.write_all(&1i64.to_le_bytes()).unwrap();
        for v in [0.0f64, 0.0, 0.0] {
            f.write_all(&v.to_le_bytes()).unwrap();
        }
        f.write_all(&[1u8, 2, 3]).unwrap();
        f.write_all(&0.0f64.to_le_bytes()).unwrap();
        f.write_all(&u64::MAX.to_le_bytes()).unwrap();
        drop(f);
        assert!(read_points3d_bin(&track_path).is_err());

        let _ = std::fs::remove_file(cam_path);
        let _ = std::fs::remove_file(img_path);
        let _ = std::fs::remove_file(pts_path);
        let _ = std::fs::remove_file(track_path);
    }

    /// 即使 2D 特徵點不再被解析，那些位元組仍必須真的存在：
    /// 檔案在特徵點區段被截斷時要報錯，而不是安靜地讀出錯位的資料。
    #[test]
    fn truncated_feature_section_still_reports_error() {
        let path = temp_path("truncated-points2d.bin");
        let mut f = File::create(&path).unwrap();
        f.write_all(&1u64.to_le_bytes()).unwrap();
        f.write_all(&3u32.to_le_bytes()).unwrap();
        for v in [1.0f64, 0.0, 0.0, 0.0] {
            f.write_all(&v.to_le_bytes()).unwrap();
        }
        for v in [0.0f64, 0.0, 0.0] {
            f.write_all(&v.to_le_bytes()).unwrap();
        }
        f.write_all(&7u32.to_le_bytes()).unwrap();
        f.write_all(b"a.jpg\0").unwrap();
        f.write_all(&5u64.to_le_bytes()).unwrap(); // 聲稱 5 個 2D 點（需要 120 位元組）
        f.write_all(&[0u8; 24]).unwrap(); // 只給 1 個
        drop(f);
        let err = read_images_bin(&path).expect_err("截斷應報錯");
        assert!(
            matches!(err, ColmapError::Format { .. }),
            "錯誤型別應為 Format，實際：{err:?}"
        );
        let _ = std::fs::remove_file(path);
    }

    /// 檔名正規化：Windows 反斜線與子目錄都應被剝除。
    #[test]
    fn name_normalization() {
        assert_eq!(normalize_name("images\\frame_1.jpg"), "frame_1.jpg");
        assert_eq!(normalize_name("a/b/frame_2.jpg"), "frame_2.jpg");
        assert_eq!(normalize_name("frame_3.jpg"), "frame_3.jpg");
    }
}
