use std::{
    ffi::OsString,
    path::{Path, PathBuf},
};

use crate::{
    error::{Result, SplatError},
    presets::QualityPreset,
    process::{ProcessManager, ProcessObserver, ProcessSpec},
};

pub fn require_verified_cli(executable: &Path) -> Result<()> {
    if executable.is_file() {
        Ok(())
    } else {
        Err(SplatError::EngineMissing(executable.display().to_string()))
    }
}

/// 展示模式是否開啟（現場示範用）。
///
/// 為什麼用環境變數而不是介面選項：攤位展示只需要在啟動應用前設定一個變數，
/// 不必改動設定結構與 UI；而且**預設關閉**才能在無顯示環境（CI、無頭機器）安全執行
/// —— Brush 的查看器需要圖形後端，開了會直接失敗。
///
/// 用法：`set MACAU_SPLAT_TRAIN_VIEWER=1` 後啟動，訓練階段會另外彈出一個 Brush 視窗，
/// 即時顯示 splats 成形過程，並可在訓練中互動旋轉視角、對照輸入視角。
pub fn demo_viewer_enabled() -> bool {
    matches!(
        std::env::var("MACAU_SPLAT_TRAIN_VIEWER").as_deref(),
        Ok("1") | Ok("true") | Ok("yes") | Ok("on")
    )
}

/// rerun.io 即時訓練統計是否開啟（loss 曲線、splat 點雲、輸入影像）。
///
/// 需要另外安裝 rerun viewer（`cargo install rerun-cli`），詳見 Brush 的 README。
pub fn rerun_logging_enabled() -> bool {
    matches!(
        std::env::var("MACAU_SPLAT_TRAIN_RERUN").as_deref(),
        Ok("1") | Ok("true") | Ok("yes") | Ok("on")
    )
}

/// 組出 Brush 的訓練參數。
///
/// 抽成純函式是為了可以單元測試「展示模式的旗標有沒有正確附加」——
/// 這種字串拼接最容易在重構時無聲壞掉。
pub fn build_train_args(
    preset: QualityPreset,
    output_directory: &Path,
    dataset: &Path,
    viewer: bool,
    rerun: bool,
) -> Vec<OsString> {
    let mut args: Vec<OsString> = vec![
        OsString::from("--total-steps"),
        preset.brush_iterations.to_string().into(),
        OsString::from("--max-resolution"),
        preset.brush_max_resolution.to_string().into(),
        OsString::from("--export-every"),
        preset.brush_iterations.to_string().into(),
        OsString::from("--export-path"),
        output_directory.into(),
        OsString::from("--export-name"),
        OsString::from("final.ply.tmp"),
    ];
    // 展示模式：彈出即時訓練查看器（Brush 內建）
    if viewer {
        args.push(OsString::from("--with-viewer"));
    }
    // 訓練動態視覺化：把訓練統計與點雲即時推送到 rerun viewer
    if rerun {
        args.push(OsString::from("--rerun-enabled"));
        args.push(OsString::from("--rerun-log-train-stats-every"));
        args.push(OsString::from("100"));
        // 點雲每隔較久才送一次（Brush 官方標註此項為 heavy）
        args.push(OsString::from("--rerun-log-splats-every"));
        args.push(OsString::from("5000"));
        args.push(OsString::from("--rerun-max-img-size"));
        args.push(OsString::from("512"));
    }
    args.push(dataset.into());
    args
}

pub async fn train(
    executable: &Path,
    dataset: &Path,
    output_directory: &Path,
    preset: QualityPreset,
    log_path: PathBuf,
    manager: &ProcessManager,
    observer: Option<ProcessObserver>,
) -> Result<PathBuf> {
    tokio::fs::create_dir_all(output_directory).await?;
    let candidate = output_directory.join("final.ply.tmp");
    if candidate.exists() {
        tokio::fs::remove_file(&candidate).await?;
    }
    let viewer = demo_viewer_enabled();
    let rerun = rerun_logging_enabled();
    if viewer || rerun {
        tracing::info!(
            viewer,
            rerun,
            "展示模式已開啟：訓練期間會顯示即時狀態（設定 MACAU_SPLAT_TRAIN_VIEWER／MACAU_SPLAT_TRAIN_RERUN 可關閉）"
        );
    }
    let args = build_train_args(preset, output_directory, dataset, viewer, rerun);
    let output = manager
        .run(ProcessSpec {
            executable: executable.to_path_buf(),
            args,
            working_directory: Some(output_directory.to_path_buf()),
            log_path: Some(log_path),
            observer,
        })
        .await?;
    if !output.success {
        let detail = output.failure_detail();
        return Err(SplatError::Process(format!(
            "Brush 退出码 {:?}{}",
            output.exit_code,
            if detail.is_empty() {
                String::new()
            } else {
                format!("\n{detail}")
            }
        )));
    }
    let candidate = if candidate.is_file() {
        candidate
    } else {
        let alternate = output_directory.join("final.ply.tmp.ply");
        if alternate.is_file() {
            alternate
        } else {
            candidate
        }
    };
    if !candidate.is_file() {
        return Err(SplatError::Process(format!(
            "Brush 未生成预期文件：{}",
            candidate.display()
        )));
    }
    Ok(candidate)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn preset() -> QualityPreset {
        // 用專案內既有的品質預設，避免測試與實際參數脫節
        crate::presets::Quality::Balanced.preset()
    }

    fn as_strings(args: &[OsString]) -> Vec<String> {
        args.iter().map(|a| a.to_string_lossy().to_string()).collect()
    }

    /// 預設（非展示）：不得出現任何 viewer／rerun 旗標，否則無頭環境會直接失敗。
    #[test]
    fn default_args_have_no_viewer_flags() {
        let args = build_train_args(preset(), Path::new("out"), Path::new("dataset"), false, false);
        let s = as_strings(&args);
        assert!(!s.iter().any(|a| a.contains("viewer")), "不應含 viewer：{s:?}");
        assert!(!s.iter().any(|a| a.contains("rerun")), "不應含 rerun：{s:?}");
        // 既有參數必須完整保留
        for expected in ["--total-steps", "--max-resolution", "--export-every", "--export-path", "--export-name"] {
            assert!(s.contains(&expected.to_string()), "缺少必要參數 {expected}：{s:?}");
        }
        assert_eq!(s.last().map(String::as_str), Some("dataset"), "資料集路徑應在最後：{s:?}");
    }

    /// 展示模式：必須附上 `--with-viewer`，且仍保留全部既有參數。
    #[test]
    fn viewer_flag_is_appended_for_demo_mode() {
        let args = build_train_args(preset(), Path::new("out"), Path::new("dataset"), true, false);
        let s = as_strings(&args);
        assert!(s.contains(&"--with-viewer".to_string()), "應含 --with-viewer：{s:?}");
        assert!(s.contains(&"--total-steps".to_string()));
        assert_eq!(s.last().map(String::as_str), Some("dataset"), "旗標必須在資料集路徑之前：{s:?}");
    }

    /// rerun：附上啟用與取樣間隔參數（點雲間隔須比統計間隔大，避免拖慢訓練）。
    #[test]
    fn rerun_flags_include_sampling_intervals() {
        let args = build_train_args(preset(), Path::new("out"), Path::new("dataset"), false, true);
        let s = as_strings(&args);
        assert!(s.contains(&"--rerun-enabled".to_string()), "應含 --rerun-enabled：{s:?}");
        let stats = s.iter().position(|a| a == "--rerun-log-train-stats-every");
        let splats = s.iter().position(|a| a == "--rerun-log-splats-every");
        assert!(stats.is_some() && splats.is_some(), "應含兩個取樣間隔：{s:?}");
        let stats_every: u32 = s[stats.unwrap() + 1].parse().expect("間隔應為數字");
        let splats_every: u32 = s[splats.unwrap() + 1].parse().expect("間隔應為數字");
        assert!(
            splats_every >= stats_every,
            "點雲取樣間隔不應小於統計間隔（Brush 標註點雲為 heavy）：{stats_every} vs {splats_every}"
        );
        assert_eq!(s.last().map(String::as_str), Some("dataset"));
    }

    /// 環境變數解析：只有明確的真值才開啟，未設定或奇怪的值一律關閉。
    #[test]
    fn env_gate_accepts_only_explicit_truthy_values() {
        // 這個測試會動到共用環境，因此逐一設定後立刻還原
        for (value, expected) in [
            ("1", true),
            ("true", true),
            ("on", true),
            ("0", false),
            ("false", false),
            ("", false),
        ] {
            if value.is_empty() {
                std::env::remove_var("MACAU_SPLAT_TRAIN_VIEWER");
            } else {
                std::env::set_var("MACAU_SPLAT_TRAIN_VIEWER", value);
            }
            assert_eq!(demo_viewer_enabled(), expected, "值 {value:?} 的判定不符預期");
        }
        std::env::remove_var("MACAU_SPLAT_TRAIN_VIEWER");
        assert!(!demo_viewer_enabled(), "測試結束後應為關閉狀態");
    }
}
