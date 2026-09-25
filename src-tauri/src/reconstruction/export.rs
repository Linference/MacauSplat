//! 把 MacauSplat 產出的標註匯出成 YOLO 訓練資料集。
//!
//! ## 為什麼需要這個模組
//! 自動標註（`annotate`）產生的是「某個專案底下的標籤」；訓練需要的是「一個合併好的資料集」。
//! 這兩者之間原本要靠人工複製檔案、改名、換算類別索引、更新 data.yaml——容易漏、容易錯，
//! 而且每次重拍都要重做一次。
//!
//! 本模組把這一步變成一條命令，是「拍攝 → 重建 → 標註 → 訓練」這條資料閉環的收口。
//!
//! ## 資料集契約（由 `tools/yolo-dataset/refine_labels.py` 的 `class_of()` 決定）
//! - 影像檔名必須以**類別名開頭**，後接底線與任意描述：`<class>_<描述>.jpg`
//!   （類別名可能含底線，例如 `casino_hotel`，因此比對用**最長前綴**，不能用 `split("_")[0]`）
//! - 標籤與影像**同名**、只換副檔名：`<同名>.txt`
//! - 標籤內容為 YOLO 格式：`<class_id> <cx> <cy> <w> <h>`（皆為 0~1 正規化）
//! - `class_id` 是 `data.yaml` 裡 `names:` 的索引——**來源專案的類別索引與資料集的不一定相同，
//!   因此匯出時必須換算**，這是本模組最容易出錯、也最必要的一步。

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use serde::Serialize;

/// 匯出時發生的錯誤。
#[derive(Debug)]
pub enum ExportError {
    /// 找不到 `data.yaml`。
    MissingDataYaml(PathBuf),
    /// `data.yaml` 裡沒有 `names:` 區塊，或格式無法解析。
    UnreadableClassMap(PathBuf),
    /// 指定的類別不在資料集的類別表裡。
    UnknownClass {
        requested: String,
        available: Vec<String>,
    },
    /// 找不到該專案的標註輸出（尚未執行過 `annotate`）。
    MissingAnnotations(PathBuf),
    /// 指定的 split 不是 train／val。
    UnknownSplit(String),
    Io(std::io::Error),
}

impl std::fmt::Display for ExportError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MissingDataYaml(p) => write!(f, "找不到資料集設定檔：{}", p.display()),
            Self::UnreadableClassMap(p) => {
                write!(f, "{} 裡找不到可解析的 names: 區塊", p.display())
            }
            Self::UnknownClass {
                requested,
                available,
            } => write!(
                f,
                "資料集沒有類別「{requested}」；現有類別：{}",
                available.join("、")
            ),
            Self::MissingAnnotations(p) => write!(
                f,
                "找不到標註輸出（{}）——請先對此專案執行 annotate",
                p.display()
            ),
            Self::UnknownSplit(s) => write!(f, "split 只能是 train 或 val，收到「{s}」"),
            Self::Io(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for ExportError {}

impl From<std::io::Error> for ExportError {
    fn from(value: std::io::Error) -> Self {
        Self::Io(value)
    }
}

/// 匯出參數。
#[derive(Debug, Clone)]
pub struct ExportOptions {
    /// 目標類別名（必須存在於資料集的 names 表）。
    pub class_name: String,
    /// 目標 split：train 或 val。
    pub split: String,
    /// 來源標記，會放進檔名以便日後追溯（例如專案短名或 `splat`）。
    pub source: String,
    /// 只列出會做什麼，不實際寫檔。
    pub dry_run: bool,
}

/// 匯出結果。
#[derive(Debug, Clone, Serialize)]
pub struct ExportReport {
    pub dataset: String,
    pub class_name: String,
    pub class_id: usize,
    pub split: String,
    /// 實際寫入的影像數。
    pub images_written: usize,
    /// 實際寫入的標籤數。
    pub labels_written: usize,
    /// 標籤存在但找不到對應影像而略過。
    pub skipped_missing_frame: usize,
    /// 目標已存在同名檔案而略過。
    pub skipped_exists: usize,
    /// 標籤內容不是合法 YOLO 格式而略過。
    pub skipped_invalid_label: usize,
    /// 前幾個寫入的檔名（供人工抽查）。
    pub samples: Vec<String>,
}

/// 解析 `data.yaml` 的 `names:` 區塊。
///
/// 只支援本專案實際使用的形式（`  0: portuguese` 逐行），不引入 YAML 相依。
pub fn read_class_map(data_yaml: &Path) -> Result<BTreeMap<usize, String>, ExportError> {
    if !data_yaml.is_file() {
        return Err(ExportError::MissingDataYaml(data_yaml.to_path_buf()));
    }
    let text = std::fs::read_to_string(data_yaml)?;
    let mut map = BTreeMap::new();
    let mut in_names = false;
    for raw in text.lines() {
        let line = raw.trim_end();
        if line.trim_start().starts_with("names:") {
            in_names = true;
            continue;
        }
        if !in_names {
            continue;
        }
        // names 區塊結束：遇到頂層鍵（行首非空白且非清單項）
        if !line.is_empty() && !line.starts_with(' ') && !line.starts_with('\t') {
            break;
        }
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        if let Some((key, value)) = trimmed.split_once(':') {
            if let Ok(index) = key.trim().parse::<usize>() {
                map.insert(index, value.trim().to_string());
            }
        }
    }
    if map.is_empty() {
        return Err(ExportError::UnreadableClassMap(data_yaml.to_path_buf()));
    }
    Ok(map)
}

/// 找出專案的影格目錄（與 `inspect` 用同一組候選）。
fn frames_directory(project: &Path) -> Option<PathBuf> {
    ["work/frames", "frames", "source/images"]
        .iter()
        .map(|rel| project.join(rel))
        .find(|dir| dir.is_dir())
}

/// 解析單行 YOLO 標籤，回傳 `(cx, cy, w, h)`。類別索引會被忽略（由呼叫端決定新索引）。
fn parse_label_line(line: &str) -> Option<[f64; 4]> {
    let mut parts = line.split_whitespace();
    let _class = parts.next()?;
    let mut values = [0.0f64; 4];
    for slot in values.iter_mut() {
        *slot = parts.next()?.parse().ok()?;
    }
    if parts.next().is_some() {
        return None;
    }
    if values.iter().any(|v| !v.is_finite()) {
        return None;
    }
    Some(values)
}

/// 把某個專案的自動標註併進 YOLO 資料集。
pub fn export_annotations(
    project: &Path,
    dataset: &Path,
    options: &ExportOptions,
) -> Result<ExportReport, ExportError> {
    let split = options.split.to_ascii_lowercase();
    if split != "train" && split != "val" {
        return Err(ExportError::UnknownSplit(options.split.clone()));
    }

    let class_map = read_class_map(&dataset.join("data.yaml"))?;
    let available: Vec<String> = class_map.values().cloned().collect();
    let class_id = class_map
        .iter()
        .find(|(_, name)| name.as_str() == options.class_name)
        .map(|(id, _)| *id)
        .ok_or_else(|| ExportError::UnknownClass {
            requested: options.class_name.clone(),
            available: available.clone(),
        })?;

    // 來源：annotate 的輸出
    let labels_src = project.join("annotation/labels");
    if !labels_src.is_dir() {
        return Err(ExportError::MissingAnnotations(labels_src));
    }
    let frames_dir = frames_directory(project);

    let images_dst = dataset.join("images").join(&split);
    let labels_dst = dataset.join("labels").join(&split);
    if !options.dry_run {
        std::fs::create_dir_all(&images_dst)?;
        std::fs::create_dir_all(&labels_dst)?;
    }

    let source_tag = if options.source.is_empty() {
        "splat".to_string()
    } else {
        options.source.clone()
    };
    // 來源標記只保留安全字元，避免把路徑或引號帶進檔名
    let source_tag: String = source_tag
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '-' {
                c
            } else {
                '_'
            }
        })
        .collect();

    let mut report = ExportReport {
        dataset: dataset.display().to_string(),
        class_name: options.class_name.clone(),
        class_id,
        split: split.clone(),
        images_written: 0,
        labels_written: 0,
        skipped_missing_frame: 0,
        skipped_exists: 0,
        skipped_invalid_label: 0,
        samples: Vec::new(),
    };

    let mut entries: Vec<PathBuf> = std::fs::read_dir(&labels_src)?
        .flatten()
        .map(|e| e.path())
        .filter(|p| {
            p.extension()
                .is_some_and(|ext| ext.eq_ignore_ascii_case("txt"))
        })
        .collect();
    entries.sort();

    for label_path in entries {
        let Some(stem) = label_path
            .file_stem()
            .map(|s| s.to_string_lossy().to_string())
        else {
            continue;
        };
        let content = std::fs::read_to_string(&label_path)?;
        let Some(values) = content.lines().find_map(parse_label_line) else {
            report.skipped_invalid_label += 1;
            continue;
        };

        // 目標檔名：<類別>_<來源>_<原影格名>，符合 class_of() 的最長前綴契約
        let new_stem = format!("{}_{}_{}", options.class_name, source_tag, stem);
        let image_dst = images_dst.join(format!("{new_stem}.jpg"));
        let label_dst = labels_dst.join(format!("{new_stem}.txt"));
        if image_dst.exists() || label_dst.exists() {
            report.skipped_exists += 1;
            continue;
        }

        // 找來源影像（抽幀輸出多為 jpg；一併接受 png 以免序列輸入時漏掉）
        let Some(frames_dir) = frames_dir.as_ref() else {
            report.skipped_missing_frame += 1;
            continue;
        };
        let source_image = ["jpg", "jpeg", "png"]
            .iter()
            .map(|ext| frames_dir.join(format!("{stem}.{ext}")))
            .find(|p| p.is_file());
        let Some(source_image) = source_image else {
            report.skipped_missing_frame += 1;
            continue;
        };

        if !options.dry_run {
            std::fs::copy(&source_image, &image_dst)?;
            // 關鍵：類別索引要換算成資料集的索引，不能沿用來源的
            std::fs::write(
                &label_dst,
                format!(
                    "{} {:.6} {:.6} {:.6} {:.6}\n",
                    class_id, values[0], values[1], values[2], values[3]
                ),
            )?;
        }
        report.images_written += 1;
        report.labels_written += 1;
        if report.samples.len() < 5 {
            report.samples.push(format!("{new_stem}.jpg"));
        }
    }

    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write_dataset(root: &Path) {
        std::fs::create_dir_all(root.join("images/train")).unwrap();
        std::fs::create_dir_all(root.join("labels/train")).unwrap();
        std::fs::write(
            root.join("data.yaml"),
            "path: datasets/demo\ntrain: images/train\nval: images/val\n\nnc: 3\nnames:\n  0: portuguese\n  1: tong_lau\n  2: market\n",
        )
        .unwrap();
    }

    /// 造一個已完成 annotate 的專案：3 張影格 + 3 個標籤，其中一個標籤格式壞掉。
    fn write_project(root: &Path) {
        std::fs::create_dir_all(root.join("annotation/labels")).unwrap();
        std::fs::create_dir_all(root.join("work/frames")).unwrap();
        for name in ["frame_0001", "frame_0002", "frame_0003"] {
            std::fs::write(
                root.join("work/frames").join(format!("{name}.jpg")),
                b"jpeg",
            )
            .unwrap();
        }
        std::fs::write(
            root.join("annotation/labels/frame_0001.txt"),
            "0 0.5 0.5 0.2 0.2\n",
        )
        .unwrap();
        std::fs::write(
            root.join("annotation/labels/frame_0002.txt"),
            "0 0.4 0.6 0.3 0.1\n",
        )
        .unwrap();
        std::fs::write(root.join("annotation/labels/frame_0003.txt"), "壞掉的行\n").unwrap();
        // 這一張沒有對應影格
        std::fs::write(
            root.join("annotation/labels/frame_9999.txt"),
            "0 0.1 0.1 0.1 0.1\n",
        )
        .unwrap();
    }

    fn temp(tag: &str) -> PathBuf {
        let dir =
            std::env::temp_dir().join(format!("macausplat-export-{}-{}", std::process::id(), tag));
        let _ = std::fs::remove_dir_all(&dir);
        dir
    }

    #[test]
    fn parses_class_map_from_data_yaml() {
        let root = temp("yaml");
        write_dataset(&root);
        let map = read_class_map(&root.join("data.yaml")).unwrap();
        assert_eq!(map.get(&0).map(String::as_str), Some("portuguese"));
        assert_eq!(map.get(&1).map(String::as_str), Some("tong_lau"));
        assert_eq!(map.get(&2).map(String::as_str), Some("market"));
        assert_eq!(map.len(), 3);
        let _ = std::fs::remove_dir_all(root);
    }

    /// **關鍵測試**：來源標籤的類別索引（0）必須換算成資料集裡的索引（1）。
    #[test]
    fn rewrites_class_index_to_dataset_index() {
        let root = temp("index");
        let dataset = root.join("dataset");
        let project = root.join("project");
        write_dataset(&dataset);
        write_project(&project);

        let report = export_annotations(
            &project,
            &dataset,
            &ExportOptions {
                class_name: "tong_lau".into(), // 資料集索引 1，來源標籤寫的是 0
                split: "train".into(),
                source: "tujian".into(),
                dry_run: false,
            },
        )
        .unwrap();

        assert_eq!(report.class_id, 1);
        assert_eq!(report.images_written, 2, "兩張有效標籤都應匯出：{report:?}");
        assert_eq!(report.skipped_invalid_label, 1, "壞掉的那行應略過");
        assert_eq!(report.skipped_missing_frame, 1, "沒有影格的那張應略過");

        let label =
            std::fs::read_to_string(dataset.join("labels/train/tong_lau_tujian_frame_0001.txt"))
                .unwrap();
        assert!(
            label.starts_with("1 "),
            "類別索引必須換算成資料集索引 1，實際：{label}"
        );
        assert!(dataset
            .join("images/train/tong_lau_tujian_frame_0001.jpg")
            .is_file());
        // 檔名必須以類別名開頭（class_of 的最長前綴契約）
        assert!(report.samples.iter().all(|s| s.starts_with("tong_lau_")));
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn rejects_unknown_class_and_split() {
        let root = temp("reject");
        let dataset = root.join("dataset");
        let project = root.join("project");
        write_dataset(&dataset);
        write_project(&project);

        let err = export_annotations(
            &project,
            &dataset,
            &ExportOptions {
                class_name: "casino_hotel".into(),
                split: "train".into(),
                source: "x".into(),
                dry_run: false,
            },
        )
        .unwrap_err();
        assert!(matches!(err, ExportError::UnknownClass { .. }), "{err:?}");

        let err = export_annotations(
            &project,
            &dataset,
            &ExportOptions {
                class_name: "tong_lau".into(),
                split: "test".into(),
                source: "x".into(),
                dry_run: false,
            },
        )
        .unwrap_err();
        assert!(matches!(err, ExportError::UnknownSplit(_)), "{err:?}");
        let _ = std::fs::remove_dir_all(root);
    }

    /// dry-run 不得寫入任何檔案，但報告要正確。
    #[test]
    fn dry_run_writes_nothing() {
        let root = temp("dryrun");
        let dataset = root.join("dataset");
        let project = root.join("project");
        write_dataset(&dataset);
        write_project(&project);

        let report = export_annotations(
            &project,
            &dataset,
            &ExportOptions {
                class_name: "market".into(),
                split: "train".into(),
                source: "p".into(),
                dry_run: true,
            },
        )
        .unwrap();
        assert_eq!(report.images_written, 2);
        assert!(!dataset
            .join("images/train/market_p_frame_0001.jpg")
            .exists());
        assert!(!dataset
            .join("labels/train/market_p_frame_0001.txt")
            .exists());
        let _ = std::fs::remove_dir_all(root);
    }

    /// 重複匯出同一批資料時，第二次應全部略過（不覆蓋、不重複）。
    #[test]
    fn second_export_skips_existing() {
        let root = temp("again");
        let dataset = root.join("dataset");
        let project = root.join("project");
        write_dataset(&dataset);
        write_project(&project);
        let options = ExportOptions {
            class_name: "tong_lau".into(),
            split: "train".into(),
            source: "s".into(),
            dry_run: false,
        };
        let first = export_annotations(&project, &dataset, &options).unwrap();
        assert_eq!(first.images_written, 2);
        let second = export_annotations(&project, &dataset, &options).unwrap();
        assert_eq!(second.images_written, 0);
        assert_eq!(second.skipped_exists, 2, "{second:?}");
        let _ = std::fs::remove_dir_all(root);
    }

    /// 尚未 annotate 的專案要給清楚的錯誤，而不是靜默成功。
    #[test]
    fn reports_missing_annotations() {
        let root = temp("noannot");
        let dataset = root.join("dataset");
        write_dataset(&dataset);
        std::fs::create_dir_all(&root).unwrap();
        let err = export_annotations(
            &root,
            &dataset,
            &ExportOptions {
                class_name: "market".into(),
                split: "train".into(),
                source: "x".into(),
                dry_run: false,
            },
        )
        .unwrap_err();
        assert!(matches!(err, ExportError::MissingAnnotations(_)), "{err:?}");
        let _ = std::fs::remove_dir_all(root);
    }
}
