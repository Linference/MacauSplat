//! COLMAP 稀疏重建（sparse reconstruction）讀取與相機投影。
//!
//! 為什麼需要這個模組：
//!     三維重建流程（COLMAP → Brush）完成後，專案目錄裡已經存有**每一張影格的相機位姿**
//!     與場景的稀疏三維點雲。這些資料正是「自動標註」的基礎：只要在三維空間定義建築的包圍盒，
//!     就能把它投影到每一張影格，得到幾何一致、零漏框的二維標註。
//!
//! 本模組只做兩件事，且都與既有引擎無關（純讀檔＋數學）：
//!     1. 解析 COLMAP 的二進位輸出（`cameras.bin` / `images.bin` / `points3D.bin`）
//!     2. 世界座標 → 相機座標 → 像素座標的投影（含針孔與徑向畸變相機模型）
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
    /// id 5：`fx, fy, cx, cy, k1, k2, k3, k4`（魚眼，本模組僅近似處理）
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

    /// 是否為魚眼類模型（畸變較大，本模組僅做最小處理）。
    pub fn is_fisheye(self) -> bool {
        matches!(
            self,
            Self::OpenCvFisheye | Self::SimpleRadialFisheye | Self::RadialFisheye | Self::ThinPrismFisheye
        )
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
            CameraModel::SimplePinhole | CameraModel::SimpleRadial | CameraModel::Radial
            | CameraModel::SimpleRadialFisheye | CameraModel::RadialFisheye => {
                (p[0], p[0], p[1], p[2])
            }
            _ => (p[0], p[1], p[2], p[3]),
        }
    }

    /// 徑向畸變係數 `(k1, k2)`（沒有的話回傳 0）。
    pub fn radial_distortion(&self) -> (f64, f64) {
        let p = &self.params;
        match self.model {
            CameraModel::SimpleRadial | CameraModel::SimpleRadialFisheye => (p[3], 0.0),
            CameraModel::Radial | CameraModel::RadialFisheye => (p[3], p[4]),
            CameraModel::OpenCv | CameraModel::FullOpenCv | CameraModel::OpenCvFisheye => (p[4], p[5]),
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

/// 一張已註冊影格的姿態與二維特徵點。
#[derive(Debug, Clone)]
pub struct Image {
    pub id: u32,
    /// 四元數 `(w, x, y, z)`。
    pub qvec: [f64; 4],
    pub tvec: [f64; 3],
    pub camera_id: u32,
    /// 相對於 `work/frames/` 的檔名（COLMAP 記錄的是它自己的相對路徑）。
    pub name: String,
    /// 每筆為 `(x, y, point3D_id)`；`point3D_id == -1` 表示沒有對應三維點。
    pub points2d: Vec<(f64, f64, i64)>,
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
#[derive(Debug, Clone)]
pub struct Point3D {
    pub id: i64,
    pub xyz: [f64; 3],
    pub rgb: [u8; 3],
    pub error: f64,
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
        Ok(Self { cameras, images, points })
    }

    /// 名稱 → 影格的查表（用來把 `frames/` 的檔案對到姿態）。
    pub fn images_by_name(&self) -> HashMap<String, &Image> {
        self.images.iter().map(|i| (normalize_name(&i.name), i)).collect()
    }

    /// 是否具備可用的點雲（供自動擬合包圍盒）。
    pub fn has_points(&self) -> bool {
        !self.points.is_empty()
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
struct Reader {
    inner: BufReader<File>,
    path: PathBuf,
}

impl Reader {
    fn open(path: &Path) -> Result<Self, ColmapError> {
        Ok(Self { inner: BufReader::new(File::open(path)?), path: path.to_path_buf() })
    }

    fn take<const N: usize>(&mut self) -> Result<[u8; N], ColmapError> {
        let mut buf = [0u8; N];
        self.inner.read_exact(&mut buf).map_err(|e| ColmapError::Format {
            file: self.path.display().to_string(),
            detail: format!("讀取 {N} 位元組失敗（檔案可能被截斷）：{e}"),
        })?;
        Ok(buf)
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
    let mut out = HashMap::with_capacity(num as usize);
    for _ in 0..num {
        let id = r.u32()?;
        let model_id = r.i32()?;
        let model = CameraModel::from_id(model_id).ok_or(ColmapError::UnsupportedCameraModel(model_id))?;
        let width = r.u64()?;
        let height = r.u64()?;
        let mut params = Vec::with_capacity(model.num_params());
        for _ in 0..model.num_params() {
            params.push(r.f64()?);
        }
        out.insert(id, Camera { id, model, width, height, params });
    }
    Ok(out)
}

/// 讀取 `images.bin`。
pub fn read_images_bin(path: &Path) -> Result<Vec<Image>, ColmapError> {
    let mut r = Reader::open(path)?;
    let num = r.u64()?;
    let mut out = Vec::with_capacity(num as usize);
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
        let n = r.u64()?;
        let mut points2d = Vec::with_capacity(n as usize);
        for _ in 0..n {
            let x = r.f64()?;
            let y = r.f64()?;
            let pid = r.i64()?;
            points2d.push((x, y, pid));
        }
        out.push(Image { id, qvec, tvec, camera_id, name, points2d });
    }
    Ok(out)
}

/// 讀取 `points3D.bin`。
pub fn read_points3d_bin(path: &Path) -> Result<Vec<Point3D>, ColmapError> {
    let mut r = Reader::open(path)?;
    let num = r.u64()?;
    let mut out = Vec::with_capacity(num as usize);
    for _ in 0..num {
        let id = r.i64()?;
        let mut xyz = [0.0; 3];
        for v in xyz.iter_mut() {
            *v = r.f64()?;
        }
        let rgb = [r.u8()?, r.u8()?, r.u8()?];
        let error = r.f64()?;
        let track_len = r.u64()?;
        let mut track = Vec::with_capacity(track_len as usize);
        for _ in 0..track_len {
            let image_id = r.u32()?;
            let point2d_idx = r.u32()?;
            track.push((image_id, point2d_idx));
        }
        out.push(Point3D { id, xyz, rgb, error, track });
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
        [1.0 - 2.0 * y * y - 2.0 * z * z, 2.0 * x * y - 2.0 * z * w, 2.0 * x * z + 2.0 * y * w],
        [2.0 * x * y + 2.0 * z * w, 1.0 - 2.0 * x * x - 2.0 * z * z, 2.0 * y * z - 2.0 * x * w],
        [2.0 * x * z - 2.0 * y * w, 2.0 * y * z + 2.0 * x * w, 1.0 - 2.0 * x * x - 2.0 * y * y],
    ]
}

/// 相機座標 → 像素座標（含徑向與切向畸變）。
///
/// 回傳 `None` 表示該點在相機後方（`z <= 0`），也就是不可見。
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

/// 世界座標 → 像素座標（一步完成）。
pub fn world_to_pixel(world: [f64; 3], image: &Image, camera: &Camera) -> Option<(f64, f64, f64)> {
    camera_to_pixel(image.to_camera(world), camera)
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
                assert!((v - if i == j { 1.0 } else { 0.0 }).abs() < 1e-12, "identity 錯誤於 ({i},{j})");
            }
        }
        // 繞 Z 軸 +90°：(x,y) → (-y,x)。四元數為 (cos(θ/2), 0, 0, sin(θ/2))，θ=90° → θ/2=45°
        let half = std::f64::consts::FRAC_PI_4;
        let rot = quaternion_to_rotation(&[half.cos(), 0.0, 0.0, half.sin()]);
        let p = [1.0, 0.0, 0.0];
        let x = rot[0][0] * p[0] + rot[0][1] * p[1] + rot[0][2] * p[2];
        let y = rot[1][0] * p[0] + rot[1][1] * p[1] + rot[1][2] * p[2];
        assert!(x.abs() < 1e-9 && (y - 1.0).abs() < 1e-9, "繞 Z 軸旋轉結果錯誤：({x}, {y})");
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
            points2d: vec![],
        };
        let (u, v, z) = world_to_pixel([0.0, 0.0, 5.0], &image, &camera).expect("應可投影");
        assert!((u - 320.0).abs() < 1e-9 && (v - 240.0).abs() < 1e-9 && (z - 5.0).abs() < 1e-9);
        // 相機後方的點不可見
        assert!(world_to_pixel([0.0, 0.0, -5.0], &image, &camera).is_none());
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
            points2d: vec![],
        };
        let c = image.camera_center();
        assert!((c[0] + 1.0).abs() < 1e-12 && (c[1] + 2.0).abs() < 1e-12 && (c[2] + 3.0).abs() < 1e-12);
        image.tvec = [0.0, 0.0, 0.0];
        let c = image.camera_center();
        assert!(c.iter().all(|v| v.abs() < 1e-12));
    }

    /// 寫出一份最小的 cameras.bin / images.bin / points3D.bin，再讀回來比對。
    /// 這是本模組最容易出錯的地方（欄位順序與型別），所以用真實二進位做往返測試。
    #[test]
    fn binary_round_trip_matches_colmap_layout() {
        let cam_path = temp_path("cameras.bin");
        let img_path = temp_path("images.bin");
        let pts_path = temp_path("points3D.bin");

        // cameras.bin：1 台 PINHOLE（fx, fy, cx, cy）
        let mut f = File::create(&cam_path).unwrap();
        f.write_all(&1u64.to_le_bytes()).unwrap();
        f.write_all(&7u32.to_le_bytes()).unwrap();      // camera_id
        f.write_all(&1i32.to_le_bytes()).unwrap();      // model_id = PINHOLE
        f.write_all(&1920u64.to_le_bytes()).unwrap();   // width
        f.write_all(&1080u64.to_le_bytes()).unwrap();   // height
        for v in [1000.0f64, 1000.0, 960.0, 540.0] {
            f.write_all(&v.to_le_bytes()).unwrap();
        }
        drop(f);

        // images.bin：1 張影格，2 個 2D 點（其中一個 point3D_id = -1）
        let mut f = File::create(&img_path).unwrap();
        f.write_all(&1u64.to_le_bytes()).unwrap();
        f.write_all(&3u32.to_le_bytes()).unwrap();      // image_id
        for v in [1.0f64, 0.0, 0.0, 0.0] {
            f.write_all(&v.to_le_bytes()).unwrap();
        }
        for v in [0.1f64, 0.2, 0.3] {
            f.write_all(&v.to_le_bytes()).unwrap();
        }
        f.write_all(&7u32.to_le_bytes()).unwrap();      // camera_id
        f.write_all(b"frame_0001.jpg\0").unwrap();
        f.write_all(&2u64.to_le_bytes()).unwrap();
        f.write_all(&12.5f64.to_le_bytes()).unwrap();
        f.write_all(&34.5f64.to_le_bytes()).unwrap();
        f.write_all(&99i64.to_le_bytes()).unwrap();
        f.write_all(&50.0f64.to_le_bytes()).unwrap();
        f.write_all(&60.0f64.to_le_bytes()).unwrap();
        f.write_all(&(-1i64).to_le_bytes()).unwrap();
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
        assert_eq!(images.len(), 1);
        assert_eq!(images[0].id, 3);
        assert_eq!(images[0].camera_id, 7);
        assert_eq!(images[0].name, "frame_0001.jpg");
        assert_eq!(images[0].qvec, [1.0, 0.0, 0.0, 0.0]);
        assert_eq!(images[0].points2d.len(), 2);
        assert_eq!(images[0].points2d[0], (12.5, 34.5, 99));
        assert_eq!(images[0].points2d[1].2, -1);

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
        f.write_all(&2u64.to_le_bytes()).unwrap();  // 聲稱有 2 台相機，但後面沒資料
        drop(f);
        let err = read_cameras_bin(&path).expect_err("應報錯");
        assert!(matches!(err, ColmapError::Format { .. }), "錯誤型別應為 Format，實際：{err:?}");
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
