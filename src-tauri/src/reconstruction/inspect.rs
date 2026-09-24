//! 專案審計：讀取一個已完成的三維重建專案，回報「可用於自動標註的素材」有多少。
//!
//! 這是自動標註前的第一步，也是資料品質的第一道關卡：
//!     - 影格數與已註冊（有相機位姿）的張數 → 註冊率太低就別浪費時間做標註
//!     - 相機模型與內參是否合理 → 內參異常會讓投影框整個歪掉
//!     - 點雲規模與三維範圍 → 決定自動擬合包圍盒是否可行
//!     - 影格檔名與 COLMAP 記錄是否對得上 → 對不上就找不到對應的影像

use std::collections::HashMap;
use std::path::{Component, Path, PathBuf};

use serde::Serialize;

use super::colmap::{normalize_name, SparseReconstruction};

/// 影格目錄的候選位置（依優先順序）。
///
/// 為什麼要集中成一個常數：這個清單原本散在 `audit_project`、`monitor` 與 QA 表的呼叫端，
/// 而 QA 表更直接寫死 `../work/frames`——於是「影格其實放在 `source/images`」的專案
/// 會產生一整頁破圖，卻沒有任何錯誤訊息。現在只有一處定義，其他人只能引用它。
pub const FRAME_DIRECTORY_CANDIDATES: [&str; 3] = ["work/frames", "frames", "source/images"];

/// QA 表在找不到影格目錄時的後備連結（維持舊行為，但由呼叫端明確選用）。
pub const FALLBACK_FRAMES_HREF: &str = FRAME_DIRECTORY_CANDIDATES[0];

/// 實際存在的影格目錄（相對於專案根目錄）。
pub fn frames_directory(project_dir: impl AsRef<Path>) -> Option<PathBuf> {
    let project_dir = project_dir.as_ref();
    FRAME_DIRECTORY_CANDIDATES
        .iter()
        .map(PathBuf::from)
        .find(|candidate| project_dir.join(candidate).is_dir())
}

/// 由 QA 表（位於 `html_dir`）指向實際影格目錄的相對連結。
///
/// 回傳 `None` 表示三個候選位置都不存在；呼叫端此時應使用 `FALLBACK_FRAMES_HREF`
/// 並讓使用者知道連結可能失效，而不是讓整份標註失敗。
/// 分隔符一律用 `/`：這是 HTML 屬性的慣例，Windows 的反斜線在瀏覽器裡會被當成一般字元。
pub fn frames_href(project_dir: impl AsRef<Path>, html_dir: impl AsRef<Path>) -> Option<String> {
    let project_dir = project_dir.as_ref();
    let frames = frames_directory(project_dir)?;
    let target = project_dir.join(&frames);
    let relative = relative_path(html_dir.as_ref(), &target)?;
    Some(relative.to_string_lossy().replace('\\', "/"))
}

/// 由 `from_dir` 走到 `to_dir` 的相對路徑。
///
/// 兩邊都必須相對於同一個基準（相對路徑會先接上目前工作目錄）；
/// 不同磁碟機（Windows）或無法比較時回 `None`。
fn relative_path(from_dir: &Path, to_dir: &Path) -> Option<PathBuf> {
    let from = absolute(from_dir);
    let to = absolute(to_dir);
    let from_parts: Vec<Component> = from.components().collect();
    let to_parts: Vec<Component> = to.components().collect();
    // Windows 的 `C:` 與 UNC 前綴必須完全相同才能比較
    if from_parts.first() != to_parts.first() {
        return None;
    }
    let common = from_parts
        .iter()
        .zip(to_parts.iter())
        .take_while(|(a, b)| a == b)
        .count();
    let mut out = PathBuf::new();
    for _ in common..from_parts.len() {
        out.push("..");
    }
    for part in &to_parts[common..] {
        out.push(part.as_os_str());
    }
    if out.as_os_str().is_empty() {
        out.push(".");
    }
    Some(out)
}

/// 把相對路徑接上目前工作目錄（已在絕對路徑時原樣回傳）。
fn absolute(path: &Path) -> PathBuf {
    if path.is_absolute() {
        path.to_path_buf()
    } else {
        std::env::current_dir()
            .map(|cwd| cwd.join(path))
            .unwrap_or_else(|_| path.to_path_buf())
    }
}

/// 三維包圍盒（軸對齊），用於快速描述點雲範圍。
#[derive(Debug, Clone, Copy, Serialize)]
pub struct Bounds3 {
    pub min: [f64; 3],
    pub max: [f64; 3],
}

/// 單一相機的使用統計。
#[derive(Debug, Clone, Serialize)]
pub struct CameraUsage {
    pub camera_id: u32,
    pub model: String,
    pub resolution: [u64; 2],
    pub intrinsics: [f64; 4],
    /// 使用這台相機的已註冊影格數。
    pub registered_images: usize,
}

/// 專案審計結果。
#[derive(Debug, Clone, Serialize)]
pub struct ProjectAudit {
    pub project_dir: PathBuf,
    pub project_name: Option<String>,
    pub quality: Option<String>,
    /// 影格目錄（見 `frames_dir`）內的影格檔案數。
    pub frames_on_disk: usize,
    /// 實際採用的影格目錄（相對於專案根目錄）；`None` 表示三個候選位置都不存在。
    ///
    /// 為什麼要回報：後續的 QA 表必須連到「真正的那個目錄」，
    /// 而使用者也需要知道這份審計是從哪裡數影格的。
    pub frames_dir: Option<String>,
    /// COLMAP 已註冊（有相機位姿）的影格數。
    pub registered_frames: usize,
    /// 註冊率 = 已註冊 / 影格數。
    pub registration_ratio: f64,
    /// 已註冊但磁碟上找不到檔案的影格（代表 frames 與 colmap 不同步）。
    pub missing_files: Vec<String>,
    /// 磁碟上有、但未被註冊的影格數。
    pub unregistered_frames: usize,
    pub cameras: Vec<CameraUsage>,
    pub points: usize,
    /// 點雲的軌跡長度中位數（越高代表該點被越多影格看到，越可靠）。
    pub median_track: f64,
    pub point_bounds: Option<Bounds3>,
    /// 是否具備自動標註所需的最低條件。
    pub ready_for_annotation: bool,
    /// 不具備時的原因。
    pub blockers: Vec<String>,
}

fn classify_quality(project: &serde_json::Value) -> Option<String> {
    project
        .get("quality")
        .and_then(|v| v.as_str())
        .map(str::to_string)
}

/// 掃描專案並產生審計報告。
pub fn audit_project(
    project_dir: impl AsRef<Path>,
) -> Result<ProjectAudit, super::colmap::ColmapError> {
    let project_dir = project_dir.as_ref().to_path_buf();
    let mut blockers = Vec::new();

    // 專案中介資料（可能不存在；不視為致命）
    let project_json = std::fs::read_to_string(project_dir.join("project.json")).ok();
    let project_value: Option<serde_json::Value> = project_json
        .as_deref()
        .and_then(|s| serde_json::from_str(s).ok());

    // 影格：依候選清單找第一個存在的目錄（清單定義在模組常數，避免各處各寫一份）
    let frames_dir = frames_directory(&project_dir);
    let mut frame_names: Vec<String> = Vec::new();
    if let Some(relative) = &frames_dir {
        if let Ok(entries) = std::fs::read_dir(project_dir.join(relative)) {
            for e in entries.flatten() {
                if e.path().is_file() {
                    frame_names.push(e.file_name().to_string_lossy().to_string());
                }
            }
        }
    } else {
        blockers.push(format!(
            "找不到影格目錄（候選：{}）",
            FRAME_DIRECTORY_CANDIDATES.join("、")
        ));
    }

    // 稀疏重建
    let reconstruction = SparseReconstruction::load(project_dir.join("work/colmap"))
        .or_else(|_| SparseReconstruction::load(&project_dir))?;

    let frame_set: std::collections::HashSet<String> =
        frame_names.iter().map(|n| normalize_name(n)).collect();
    let registered_names: Vec<String> = reconstruction
        .images
        .iter()
        .map(|i| normalize_name(&i.name))
        .collect();
    let missing_files: Vec<String> = registered_names
        .iter()
        .filter(|n| !frame_set.contains(*n))
        .cloned()
        .collect();
    let registered_set: std::collections::HashSet<&String> = registered_names.iter().collect();
    let unregistered = frame_names
        .iter()
        .filter(|n| !registered_set.contains(&normalize_name(n)))
        .count();

    // 相機使用統計
    let mut usage: HashMap<u32, usize> = HashMap::new();
    for img in &reconstruction.images {
        *usage.entry(img.camera_id).or_insert(0) += 1;
    }
    let mut cameras: Vec<CameraUsage> = reconstruction
        .cameras
        .values()
        .map(|c| {
            let (fx, fy, cx, cy) = c.intrinsics();
            CameraUsage {
                camera_id: c.id,
                model: format!("{:?}", c.model),
                resolution: [c.width, c.height],
                intrinsics: [fx, fy, cx, cy],
                registered_images: usage.get(&c.id).copied().unwrap_or(0),
            }
        })
        .collect();
    cameras.sort_by_key(|c| c.camera_id);

    // 點雲統計
    let mut tracks: Vec<usize> = reconstruction
        .points
        .iter()
        .map(|p| p.track.len())
        .collect();
    tracks.sort_unstable();
    let median_track = if tracks.is_empty() {
        0.0
    } else {
        tracks[tracks.len() / 2] as f64
    };
    let point_bounds = if reconstruction.points.is_empty() {
        None
    } else {
        let mut min = [f64::INFINITY; 3];
        let mut max = [f64::NEG_INFINITY; 3];
        for p in &reconstruction.points {
            for i in 0..3 {
                min[i] = min[i].min(p.xyz[i]);
                max[i] = max[i].max(p.xyz[i]);
            }
        }
        Some(Bounds3 { min, max })
    };

    let registration_ratio = if frame_names.is_empty() {
        0.0
    } else {
        reconstruction.images.len() as f64 / frame_names.len() as f64
    };

    // 可用性判斷
    // 注意：註冊率低代表相機位姿不可靠，投影出來的框會整個歪掉——因此「有資料」不等於「可用」。
    // 這裡要求註冊率達 50% 且至少 20 張已註冊影格，才算真的能做自動標註。
    const MIN_REGISTERED: usize = 20;
    if reconstruction.images.is_empty() {
        blockers.push("COLMAP 沒有任何已註冊影格".into());
    }
    if reconstruction.cameras.is_empty() {
        blockers.push("COLMAP 沒有相機內參".into());
    }
    if registration_ratio < 0.5 && !frame_names.is_empty() {
        blockers.push(format!(
            "註冊率僅 {:.0}%（低於 50%），投影標註的覆蓋率會很差，建議重拍或重跑重建",
            registration_ratio * 100.0
        ));
    }
    if reconstruction.images.len() < MIN_REGISTERED {
        blockers.push(format!(
            "已註冊影格僅 {} 張（低於 {} 張），不足以構成訓練資料集",
            reconstruction.images.len(),
            MIN_REGISTERED
        ));
    }
    if reconstruction.points.is_empty() {
        // 沒有點雲仍可手動指定包圍盒，只是無法自動擬合
        blockers.push("缺少 points3D.bin（可手動指定包圍盒，但無法自動擬合）".into());
    }
    // 只有「相機與影格齊備、註冊率足夠、影格數足夠」才適合自動標註
    let ready_for_annotation = !reconstruction.images.is_empty()
        && !reconstruction.cameras.is_empty()
        && registration_ratio >= 0.5
        && reconstruction.images.len() >= MIN_REGISTERED;

    Ok(ProjectAudit {
        project_dir,
        project_name: project_value
            .as_ref()
            .and_then(|v| v.get("name"))
            .and_then(|v| v.as_str())
            .map(str::to_string),
        quality: project_value.as_ref().and_then(classify_quality),
        frames_on_disk: frame_names.len(),
        frames_dir: frames_dir.map(|p| p.to_string_lossy().replace('\\', "/")),
        registered_frames: reconstruction.images.len(),
        registration_ratio,
        missing_files: missing_files.into_iter().take(20).collect(),
        unregistered_frames: unregistered,
        cameras,
        points: reconstruction.points.len(),
        median_track,
        point_bounds,
        ready_for_annotation,
        blockers,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::io::Write;

    /// 造一個最小可用的專案。
    ///
    /// `frames` 張影格；`registered` 張進入 COLMAP（`registered < frames` 可模擬低註冊率）；
    /// `name_offset` 可讓 COLMAP 記錄的檔名與磁碟不同（模擬不同步）。
    fn make_project(
        tag: &str,
        frames: usize,
        registered: usize,
        with_points: bool,
        name_offset: bool,
    ) -> PathBuf {
        let dir =
            std::env::temp_dir().join(format!("macausplat-audit-{}-{}", std::process::id(), tag));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(dir.join("work/frames")).unwrap();
        fs::create_dir_all(dir.join("work/colmap")).unwrap();
        for i in 0..frames {
            fs::write(
                dir.join("work/frames").join(format!("frame_{i:04}.jpg")),
                b"x",
            )
            .unwrap();
        }
        fs::write(
            dir.join("project.json"),
            br#"{"name":"demo","quality":"balanced","status":"completed"}"#,
        )
        .unwrap();

        // cameras.bin：SIMPLE_PINHOLE
        let mut f = fs::File::create(dir.join("work/colmap/cameras.bin")).unwrap();
        f.write_all(&1u64.to_le_bytes()).unwrap();
        f.write_all(&1u32.to_le_bytes()).unwrap();
        f.write_all(&0i32.to_le_bytes()).unwrap();
        f.write_all(&640u64.to_le_bytes()).unwrap();
        f.write_all(&480u64.to_le_bytes()).unwrap();
        for v in [500.0f64, 320.0, 240.0] {
            f.write_all(&v.to_le_bytes()).unwrap();
        }
        drop(f);

        // images.bin：registered 張
        let mut f = fs::File::create(dir.join("work/colmap/images.bin")).unwrap();
        f.write_all(&(registered as u64).to_le_bytes()).unwrap();
        for i in 0..registered {
            f.write_all(&((i + 1) as u32).to_le_bytes()).unwrap();
            for v in [1.0f64, 0.0, 0.0, 0.0] {
                f.write_all(&v.to_le_bytes()).unwrap();
            }
            for v in [0.0f64, 0.0, (i as f64) * 0.5] {
                f.write_all(&v.to_le_bytes()).unwrap();
            }
            f.write_all(&1u32.to_le_bytes()).unwrap();
            let name = if name_offset && i == registered - 1 {
                format!("registered_only_{i:04}.jpg")
            } else {
                format!("frame_{i:04}.jpg")
            };
            f.write_all(format!("{name}\0").as_bytes()).unwrap();
            f.write_all(&0u64.to_le_bytes()).unwrap();
        }
        drop(f);

        if with_points {
            let mut f = fs::File::create(dir.join("work/colmap/points3D.bin")).unwrap();
            f.write_all(&2u64.to_le_bytes()).unwrap();
            for (id, xyz) in [(1i64, [0.0f64, 0.0, 0.0]), (2, [1.0, 2.0, 3.0])] {
                f.write_all(&id.to_le_bytes()).unwrap();
                for v in xyz {
                    f.write_all(&v.to_le_bytes()).unwrap();
                }
                f.write_all(&[9u8, 9, 9]).unwrap();
                f.write_all(&0.1f64.to_le_bytes()).unwrap();
                f.write_all(&1u64.to_le_bytes()).unwrap();
                f.write_all(&1u32.to_le_bytes()).unwrap();
                f.write_all(&0u32.to_le_bytes()).unwrap();
            }
            drop(f);
        }
        dir
    }

    #[test]
    fn audit_reports_ready_project() {
        let dir = make_project("ready", 24, 24, true, false);
        let a = audit_project(&dir).expect("審計應成功");
        assert_eq!(a.frames_on_disk, 24);
        assert_eq!(a.registered_frames, 24);
        assert!((a.registration_ratio - 1.0).abs() < 1e-9);
        assert_eq!(a.cameras.len(), 1);
        assert_eq!(a.cameras[0].model, "SimplePinhole");
        assert_eq!(a.cameras[0].intrinsics, [500.0, 500.0, 320.0, 240.0]);
        assert_eq!(a.cameras[0].registered_images, 24);
        assert_eq!(a.points, 2);
        assert!(a.point_bounds.is_some());
        assert!(
            a.ready_for_annotation,
            "應可用於標註；blockers={:?}",
            a.blockers
        );
        assert!(a.blockers.is_empty(), "不應有 blocker：{:?}", a.blockers);
        assert_eq!(a.project_name.as_deref(), Some("demo"));
        assert_eq!(a.quality.as_deref(), Some("balanced"));
        // 審計要說出影格是從哪個目錄數來的（後續 QA 表要用同一個）
        assert_eq!(a.frames_dir.as_deref(), Some("work/frames"));
        let _ = fs::remove_dir_all(dir);
    }

    /// 影格目錄的三個候選位置：找到第一個存在的，並能算出 QA 表用的相對連結。
    ///
    /// 這是 QA 表破圖的根因：舊版寫死 `../work/frames`，
    /// 但專案的影格也可能在 `frames/` 或 `source/images/`。
    #[test]
    fn locates_the_actual_frames_directory_and_builds_a_relative_link() {
        let root = std::env::temp_dir().join(format!("macausplat-frames-{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(&root).unwrap();

        // 三個都沒有 → 沒有可用的影格目錄
        assert_eq!(frames_directory(&root), None);
        assert_eq!(frames_href(&root, root.join("annotation")), None);
        assert_eq!(FALLBACK_FRAMES_HREF, "work/frames");

        // 只有 source/images → 必須選它，而不是 work/frames
        fs::create_dir_all(root.join("source/images")).unwrap();
        assert_eq!(
            frames_directory(&root).as_deref(),
            Some(Path::new("source/images"))
        );
        assert_eq!(
            frames_href(&root, root.join("annotation")).as_deref(),
            Some("../source/images")
        );

        // 加上 frames/ → 優先序較高的 frames 勝出
        fs::create_dir_all(root.join("frames")).unwrap();
        assert_eq!(
            frames_directory(&root).as_deref(),
            Some(Path::new("frames"))
        );

        // 最後加上 work/frames → 最高優先
        fs::create_dir_all(root.join("work/frames")).unwrap();
        assert_eq!(
            frames_directory(&root).as_deref(),
            Some(Path::new("work/frames"))
        );
        assert_eq!(
            frames_href(&root, root.join("annotation")).as_deref(),
            Some("../work/frames")
        );
        // QA 表放在專案根目錄時，連結就是相對的子目錄
        assert_eq!(
            frames_href(&root, &root).as_deref(),
            Some("work/frames"),
            "同一層時不該出現多餘的 .."
        );
        // 相對路徑會以目前工作目錄為基準（命令列的 `--out` 常常是相對路徑）
        let cwd = std::env::current_dir().unwrap();
        assert_eq!(
            relative_path(Path::new("."), &cwd.join("work/frames")).as_deref(),
            Some(Path::new("work/frames"))
        );
        assert_eq!(
            relative_path(&cwd.join("annotation"), Path::new("work/frames")).as_deref(),
            Some(Path::new("../work/frames"))
        );
        // 不同磁碟機（Windows）無法表示成相對路徑 → 回 None 讓呼叫端改用後備值
        #[cfg(windows)]
        assert_eq!(
            relative_path(Path::new("C:/a"), Path::new("D:/b")),
            None,
            "跨磁碟機不該硬湊出相對路徑"
        );
        // 連結一律用 `/`（HTML 屬性裡的反斜線不是分隔符）
        assert!(!frames_href(&root, root.join("annotation"))
            .unwrap()
            .contains('\\'));

        let _ = fs::remove_dir_all(root);
    }

    /// 審計會指出影格目錄不存在，而不是安靜地回報 0 張。
    #[test]
    fn audit_reports_missing_frames_directory() {
        let dir = make_project("noframes", 24, 24, true, false);
        fs::remove_dir_all(dir.join("work/frames")).unwrap();
        let a = audit_project(&dir).expect("審計應成功");
        assert_eq!(a.frames_on_disk, 0);
        assert!(a.frames_dir.is_none());
        assert!(
            a.blockers.iter().any(|b| b.contains("找不到影格目錄")),
            "應指出找不到影格目錄：{:?}",
            a.blockers
        );
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn audit_flags_missing_points_as_blocker_but_still_usable() {
        let dir = make_project("nopoints", 24, 24, false, false);
        let a = audit_project(&dir).expect("審計應成功");
        assert_eq!(a.points, 0);
        assert!(a.point_bounds.is_none());
        assert!(
            a.blockers.iter().any(|b| b.contains("points3D")),
            "應指出缺少點雲：{:?}",
            a.blockers
        );
        // 沒有點雲仍可手動給包圍盒，所以仍算可用
        assert!(a.ready_for_annotation);
        let _ = fs::remove_dir_all(dir);
    }

    /// 低註冊率必須被判為「不可用」——這是真實專案裡最常見的失敗模式
    /// （實測 8 個專案中有 3 個註冊率只有 15%、2%、1%）。
    #[test]
    fn audit_rejects_low_registration_project() {
        let dir = make_project("lowreg", 200, 3, true, false);
        let a = audit_project(&dir).expect("審計應成功");
        assert!(a.registration_ratio < 0.5);
        assert!(!a.ready_for_annotation, "低註冊率不應被判為可用");
        assert!(
            a.blockers.iter().any(|b| b.contains("註冊率")),
            "應指出註冊率過低：{:?}",
            a.blockers
        );
        assert!(
            a.blockers.iter().any(|b| b.contains("已註冊影格僅")),
            "應指出已註冊影格數不足：{:?}",
            a.blockers
        );
        let _ = fs::remove_dir_all(dir);
    }

    /// 註冊率高但影格數太少（例如 10 張）同樣不足以構成資料集。
    #[test]
    fn audit_rejects_too_few_frames() {
        let dir = make_project("fewframes", 10, 10, true, false);
        let a = audit_project(&dir).expect("審計應成功");
        assert!(a.registration_ratio > 0.9);
        assert!(!a.ready_for_annotation, "影格數不足時不應判為可用");
        assert!(a
            .blockers
            .iter()
            .any(|b| b.contains("不足以構成訓練資料集")));
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn audit_detects_name_mismatch_between_frames_and_colmap() {
        // 24 張影格、24 張註冊，但最後一張 COLMAP 記錄的是磁碟上沒有的檔名
        let dir = make_project("mismatch", 24, 24, true, true);
        let a = audit_project(&dir).expect("審計應成功");
        assert_eq!(
            a.missing_files,
            vec!["registered_only_0023.jpg".to_string()]
        );
        assert_eq!(a.unregistered_frames, 1);
        let _ = fs::remove_dir_all(dir);
    }
}
