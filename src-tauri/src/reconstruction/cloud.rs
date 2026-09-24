//! 建圖過程的三維點雲：讀取 COLMAP 的中間快照，供應用內即時檢視「點雲正在長大」。
//!
//! ## 為什麼需要
//! COLMAP 的增量式重建（mapper）在執行時**不會**提供任何三維預覽——這是它的設計，
//! 不是缺陷。使用者在長達數分鐘的建圖階段只能看主控台滾動的文字。
//!
//! 但 COLMAP 支援 `--Mapper.snapshot_path`：每註冊 N 張影像就把當下的重建寫成一份完整模型
//! （cameras.bin／images.bin／points3D.bin）。因此只要**持續讀取最新的一份快照**，
//! 就能在應用內畫出點雲增長的過程——不需要修改 COLMAP，也不需要額外的服務。
//!
//! ## 設計要點
//! - **只回傳渲染需要的資料**：位置（f32）與顏色（u8），不把 `Point3D` 的 track 資料送到前端。
//! - **均勻降採樣**：均勻間隔取樣，保證點數上限可控且分布不失真（不是取前 N 個）。
//! - **相機也畫出來**：顯示每張已註冊影像的相機中心，讓「拍攝軌跡」與點雲一起成長，
//!   這比單獨畫點雲更能說明「現在註冊到哪裡了」。

use std::path::{Path, PathBuf};

use serde::Serialize;

use super::colmap::{normalize_name, SparseReconstruction};

/// 一份可渲染的點雲快照。
#[derive(Debug, Clone, Serialize)]
pub struct SnapshotCloud {
    /// 交錯的 xyz 座標（長度 = 3 × 點數）。
    pub positions: Vec<f32>,
    /// 交錯的 rgb 顏色（長度 = 3 × 點數）。
    pub colors: Vec<u8>,
    /// 實際送出的點數（降採樣後）。
    pub point_count: usize,
    /// 快照中的總點數（降採樣前）。
    pub total_points: usize,
    /// 已註冊的相機中心座標（交錯 xyz）。
    pub cameras: Vec<f32>,
    /// 世界座標包圍盒 `[min_x, min_y, min_z, max_x, max_y, max_z]`。
    pub bounds: [f32; 6],
    /// 讀取來源的快照目錄名稱（例如 `3`）；`None` 表示還沒有任何快照。
    pub snapshot: Option<String>,
    /// 可用的快照總數。
    pub snapshot_count: usize,
    /// 是否已是最終模型（而非中間快照）。
    pub is_final: bool,
}

impl SnapshotCloud {
    pub fn empty() -> Self {
        Self {
            positions: Vec::new(),
            colors: Vec::new(),
            point_count: 0,
            total_points: 0,
            cameras: Vec::new(),
            bounds: [0.0; 6],
            snapshot: None,
            snapshot_count: 0,
            is_final: false,
        }
    }
}

/// 找出所有可用的快照目錄（每個目錄都必須具備三個 `.bin`）。
///
/// 依目錄名稱的數字大小排序，因此 `2` 會排在 `10` 之前（不是字典序）。
pub fn snapshot_directories(project_dir: &Path) -> Vec<PathBuf> {
    let roots = [
        project_dir.join("work/colmap/sparse/snapshots"),
        project_dir.join("work/colmap/snapshots"),
        project_dir.join("sparse/snapshots"),
    ];
    let mut found: Vec<(u64, PathBuf)> = Vec::new();
    for root in roots {
        let Ok(entries) = std::fs::read_dir(&root) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if !path.is_dir() {
                continue;
            }
            if !path.join("images.bin").is_file() || !path.join("points3D.bin").is_file() {
                continue;
            }
            let order = entry
                .file_name()
                .to_string_lossy()
                .parse::<u64>()
                .unwrap_or(u64::MAX);
            found.push((order, path));
        }
    }
    found.sort_by_key(|(order, _)| *order);
    found.into_iter().map(|(_, p)| p).collect()
}

/// 找出最終稀疏模型目錄（重建完成後使用）。
pub fn final_model_directory(project_dir: &Path) -> Option<PathBuf> {
    [
        project_dir.join("work/colmap/sparse/0"),
        project_dir.join("work/colmap"),
        project_dir.join("sparse/0"),
    ]
    .into_iter()
    .find(|p| p.join("images.bin").is_file() && p.join("points3D.bin").is_file())
}

/// 均勻降採樣：以固定間隔取樣，保持分布在整個點雲上。
///
/// 為什麼不用「取前 N 個」：COLMAP 的點是按三角化順序排列的，
/// 取前 N 個會只畫出重建初期、集中在起始影像附近的那一團，看起來像失敗。
pub fn downsample_indices(total: usize, max_points: usize) -> Vec<usize> {
    if max_points == 0 || total <= max_points {
        return (0..total).collect();
    }
    let stride = total as f64 / max_points as f64;
    (0..max_points)
        .map(|i| ((i as f64) * stride) as usize)
        .collect()
}

/// 載入最新的快照（若沒有快照則載入最終模型）。
///
/// `max_points` 為送到前端的點數上限（0 表示不限制）。
pub fn load_snapshot_cloud(
    project_dir: &Path,
    max_points: usize,
) -> Result<SnapshotCloud, super::colmap::ColmapError> {
    let snapshots = snapshot_directories(project_dir);
    let snapshot_count = snapshots.len();
    // 優先看最新快照；沒有快照時退回最終模型（讓「完成後」也能看到結果）
    let (dir, label, is_final) = match snapshots.last() {
        Some(dir) => (
            dir.clone(),
            dir.file_name().map(|s| s.to_string_lossy().to_string()),
            false,
        ),
        None => match final_model_directory(project_dir) {
            Some(dir) => (dir, Some("final".to_string()), true),
            None => return Ok(SnapshotCloud::empty()),
        },
    };
    let reconstruction = SparseReconstruction::load_dir(&dir)?;
    Ok(build_cloud(
        &reconstruction,
        max_points,
        label,
        snapshot_count,
        is_final,
    ))
}

/// 由稀疏重建組出可渲染的點雲（抽出來以便測試）。
pub fn build_cloud(
    reconstruction: &SparseReconstruction,
    max_points: usize,
    snapshot: Option<String>,
    snapshot_count: usize,
    is_final: bool,
) -> SnapshotCloud {
    let total = reconstruction.points.len();
    let indices = downsample_indices(total, max_points);
    let mut positions = Vec::with_capacity(indices.len() * 3);
    let mut colors = Vec::with_capacity(indices.len() * 3);
    let mut min = [f32::INFINITY; 3];
    let mut max = [f32::NEG_INFINITY; 3];
    for &i in &indices {
        let p = &reconstruction.points[i];
        for k in 0..3 {
            let v = p.xyz[k] as f32;
            positions.push(v);
            min[k] = min[k].min(v);
            max[k] = max[k].max(v);
        }
        colors.push(p.rgb[0]);
        colors.push(p.rgb[1]);
        colors.push(p.rgb[2]);
    }
    if positions.is_empty() {
        min = [0.0; 3];
        max = [0.0; 3];
    }
    // 相機中心：讓「拍攝軌跡」跟著一起成長
    let mut cameras = Vec::with_capacity(reconstruction.images.len() * 3);
    for image in &reconstruction.images {
        let c = image.camera_center();
        cameras.push(c[0] as f32);
        cameras.push(c[1] as f32);
        cameras.push(c[2] as f32);
    }
    SnapshotCloud {
        point_count: positions.len() / 3,
        total_points: total,
        positions,
        colors,
        cameras,
        bounds: [min[0], min[1], min[2], max[0], max[1], max[2]],
        snapshot,
        snapshot_count,
        is_final,
    }
}

/// 快照的簡要資訊（給前端列出、選擇用）。
#[derive(Debug, Clone, Serialize)]
pub struct SnapshotInfo {
    pub name: String,
    pub path: String,
    pub is_final: bool,
}

/// 列出專案中所有可檢視的快照（含最終模型）。
pub fn list_snapshots(project_dir: &Path) -> Vec<SnapshotInfo> {
    let mut out: Vec<SnapshotInfo> = snapshot_directories(project_dir)
        .into_iter()
        .map(|p| SnapshotInfo {
            name: p
                .file_name()
                .map(|s| s.to_string_lossy().to_string())
                .unwrap_or_default(),
            path: p.display().to_string(),
            is_final: false,
        })
        .collect();
    if let Some(dir) = final_model_directory(project_dir) {
        out.push(SnapshotInfo {
            name: "final".into(),
            path: dir.display().to_string(),
            is_final: true,
        });
    }
    out
}

/// 目前註冊影像的檔名清單（給前端顯示「最新註冊的是哪張」）。
pub fn registered_image_names(project_dir: &Path, limit: usize) -> Vec<String> {
    let dir = snapshot_directories(project_dir)
        .last()
        .cloned()
        .or_else(|| final_model_directory(project_dir));
    let Some(dir) = dir else {
        return Vec::new();
    };
    let Ok(reconstruction) = SparseReconstruction::load_dir(&dir) else {
        return Vec::new();
    };
    let mut names: Vec<String> = reconstruction
        .images
        .iter()
        .map(|i| normalize_name(&i.name))
        .collect();
    names.sort();
    names.truncate(limit);
    names
}

#[cfg(test)]
mod tests {
    use super::super::colmap::{Camera, CameraModel, Image, Point3D};
    use super::*;
    use std::collections::HashMap;
    use std::io::Write;

    fn rgb_palette(n: usize) -> Vec<Point3D> {
        (0..n)
            .map(|i| Point3D {
                id: i as i64,
                xyz: [i as f64, (i * 2) as f64, (i % 7) as f64],
                rgb: [(i % 256) as u8, ((i * 3) % 256) as u8, 200],
                error: 0.1,
                track: vec![(1, 0)],
            })
            .collect()
    }

    fn reconstruction_with(points: usize) -> SparseReconstruction {
        let mut cameras = HashMap::new();
        cameras.insert(
            1,
            Camera {
                id: 1,
                model: CameraModel::SimplePinhole,
                width: 640,
                height: 480,
                params: vec![500.0, 320.0, 240.0],
            },
        );
        let images = vec![
            Image {
                id: 1,
                qvec: [1.0, 0.0, 0.0, 0.0],
                tvec: [0.0, 0.0, 0.0],
                camera_id: 1,
                name: "a.jpg".into(),
                points2d: vec![],
            },
            Image {
                id: 2,
                qvec: [1.0, 0.0, 0.0, 0.0],
                tvec: [-1.0, 0.0, 0.0],
                camera_id: 1,
                name: "b.jpg".into(),
                points2d: vec![],
            },
        ];
        SparseReconstruction {
            cameras,
            images,
            points: rgb_palette(points),
        }
    }

    /// 降採樣：不超過上限、且均勻覆蓋整個範圍（而非只取前面一小段）。
    #[test]
    fn downsample_covers_the_whole_range() {
        let idx = downsample_indices(1000, 100);
        assert_eq!(idx.len(), 100);
        assert_eq!(idx[0], 0);
        assert!(idx[99] >= 980, "最後一個索引應接近尾端，實際 {}", idx[99]);
        // 間隔均勻
        let gaps: Vec<usize> = idx.windows(2).map(|w| w[1] - w[0]).collect();
        let min = *gaps.iter().min().unwrap();
        let max = *gaps.iter().max().unwrap();
        assert!(max - min <= 1, "間隔應均勻，實際 {min}..{max}");
    }

    /// 不超過上限時應全部保留。
    #[test]
    fn downsample_keeps_everything_when_under_limit() {
        assert_eq!(downsample_indices(50, 100).len(), 50);
        assert_eq!(downsample_indices(0, 100).len(), 0);
    }

    /// 建出的點雲：座標／顏色長度一致、包圍盒正確、相機中心被帶上。
    #[test]
    fn builds_cloud_with_bounds_and_cameras() {
        let recon = reconstruction_with(500);
        let cloud = build_cloud(&recon, 100, Some("3".into()), 4, false);
        assert_eq!(cloud.point_count, 100);
        assert_eq!(cloud.positions.len(), 300);
        assert_eq!(cloud.colors.len(), 300);
        assert_eq!(cloud.total_points, 500);
        assert_eq!(cloud.snapshot.as_deref(), Some("3"));
        assert_eq!(cloud.snapshot_count, 4);
        assert!(!cloud.is_final);
        // 包圍盒：x 從 0 到接近 499、y 從 0 到接近 998
        assert!(
            cloud.bounds[0] >= 0.0 && cloud.bounds[3] > 400.0,
            "bounds={:?}",
            cloud.bounds
        );
        assert!(
            cloud.bounds[1] >= 0.0 && cloud.bounds[4] > 800.0,
            "bounds={:?}",
            cloud.bounds
        );
        // 兩台相機：中心分別為 (0,0,0) 與 (1,0,0)
        assert_eq!(cloud.cameras.len(), 6);
        assert!((cloud.cameras[0] - 0.0).abs() < 1e-6);
        assert!(
            (cloud.cameras[3] - 1.0).abs() < 1e-6,
            "第二台相機中心應在 x=1"
        );
    }

    /// 空的重建不應 panic，且包圍盒退化為 0。
    #[test]
    fn empty_reconstruction_is_safe() {
        let recon = SparseReconstruction {
            cameras: HashMap::new(),
            images: vec![],
            points: vec![],
        };
        let cloud = build_cloud(&recon, 100, None, 0, false);
        assert_eq!(cloud.point_count, 0);
        assert!(cloud.positions.is_empty());
        assert_eq!(cloud.bounds, [0.0; 6]);
        assert!(cloud.cameras.is_empty());
    }

    /// 讀取專案：沒有快照時應退回最終模型；完全沒有模型時回傳空雲。
    #[test]
    fn load_prefers_snapshots_then_final_model() {
        let root = std::env::temp_dir().join(format!("macausplat-cloud-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(root.join("work/colmap/sparse/0")).unwrap();
        // 一開始什麼都沒有
        let cloud = load_snapshot_cloud(&root, 1000).expect("不應報錯");
        assert_eq!(cloud.point_count, 0);
        assert!(cloud.snapshot.is_none());

        // 寫一份最終模型（1 張影像、2 個點）
        write_model(&root.join("work/colmap/sparse/0"), &["a.jpg"], 2);
        let cloud = load_snapshot_cloud(&root, 1000).expect("應讀到最終模型");
        assert!(cloud.is_final, "沒有快照時應標記為最終模型");
        assert_eq!(cloud.snapshot.as_deref(), Some("final"));
        assert_eq!(cloud.point_count, 2);
        assert_eq!(cloud.cameras.len(), 3);

        // 再加兩份中間快照 → 應優先採用「最新」的那一份（數字大者，不是字典序）
        for name in ["2", "10"] {
            let dir = root.join("work/colmap/sparse/snapshots").join(name);
            std::fs::create_dir_all(&dir).unwrap();
            write_model(&dir, &["a.jpg"], if name == "10" { 5 } else { 3 });
        }
        let cloud = load_snapshot_cloud(&root, 1000).expect("應讀到最新快照");
        assert_eq!(
            cloud.snapshot.as_deref(),
            Some("10"),
            "應選數字最大的快照（10 而非 2）"
        );
        assert_eq!(cloud.snapshot_count, 2);
        assert_eq!(cloud.point_count, 5);
        assert!(!cloud.is_final);

        let list = list_snapshots(&root);
        assert_eq!(list.len(), 3, "兩份快照 + 最終模型：{list:?}");
        assert!(list.last().unwrap().is_final);
        let names = registered_image_names(&root, 10);
        assert_eq!(names, vec!["a.jpg".to_string()]);

        let _ = std::fs::remove_dir_all(root);
    }

    /// 寫出一份最小可讀的 COLMAP 模型。
    fn write_model(dir: &Path, images: &[&str], points: usize) {
        std::fs::create_dir_all(dir).unwrap();
        let mut f = std::fs::File::create(dir.join("cameras.bin")).unwrap();
        f.write_all(&1u64.to_le_bytes()).unwrap();
        f.write_all(&1u32.to_le_bytes()).unwrap();
        f.write_all(&0i32.to_le_bytes()).unwrap();
        f.write_all(&640u64.to_le_bytes()).unwrap();
        f.write_all(&480u64.to_le_bytes()).unwrap();
        for v in [500.0f64, 320.0, 240.0] {
            f.write_all(&v.to_le_bytes()).unwrap();
        }
        drop(f);

        let mut f = std::fs::File::create(dir.join("images.bin")).unwrap();
        f.write_all(&(images.len() as u64).to_le_bytes()).unwrap();
        for (i, name) in images.iter().enumerate() {
            f.write_all(&((i + 1) as u32).to_le_bytes()).unwrap();
            for v in [1.0f64, 0.0, 0.0, 0.0] {
                f.write_all(&v.to_le_bytes()).unwrap();
            }
            for v in [-(i as f64), 0.0, 0.0] {
                f.write_all(&v.to_le_bytes()).unwrap();
            }
            f.write_all(&1u32.to_le_bytes()).unwrap();
            f.write_all(format!("{name}\0").as_bytes()).unwrap();
            f.write_all(&0u64.to_le_bytes()).unwrap();
        }
        drop(f);

        let mut f = std::fs::File::create(dir.join("points3D.bin")).unwrap();
        f.write_all(&(points as u64).to_le_bytes()).unwrap();
        for i in 0..points {
            f.write_all(&(i as i64).to_le_bytes()).unwrap();
            for v in [i as f64, 0.0, 0.0] {
                f.write_all(&v.to_le_bytes()).unwrap();
            }
            f.write_all(&[10u8, 20, 30]).unwrap();
            f.write_all(&0.5f64.to_le_bytes()).unwrap();
            f.write_all(&1u64.to_le_bytes()).unwrap();
            f.write_all(&1u32.to_le_bytes()).unwrap();
            f.write_all(&0u32.to_le_bytes()).unwrap();
        }
        drop(f);
    }
}
