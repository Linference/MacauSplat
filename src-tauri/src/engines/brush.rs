use std::{
    ffi::OsString,
    path::{Path, PathBuf},
};

use crate::{
    engines::ColmapAccelerationStatus,
    error::{Result, SplatError},
    presets::{BrushTrainingPreset,QualityPreset},
    process::{ProcessManager, ProcessObserver, ProcessOutput, ProcessSpec},
};

const BRUSH_GPU_LOG_FILTER: &str =
    "brush_cli=info,brush_process=info,cubecl_wgpu=info,burn_wgpu=info";
pub(super) const REQUIRED_CLI_FLAGS: &[&str] = &[
    "--total-train-iters",
    "--max-resolution",
    "--refine-every",
    "--max-splats",
    "--growth-grad-threshold",
    "--growth-select-fraction",
    "--growth-stop-iter",
    "--export-every",
    "--export-path",
    "--export-name",
];

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BrushGpuLaunchPolicy {
    expected_device_name: Option<String>,
    force_first_discrete_gpu: bool,
    disable_amd_switchable_graphics: bool,
    selection_reason: &'static str,
}

pub struct BrushTrainingOptions<'a> {
    pub preset: BrushTrainingPreset,
    pub precision: bool,
    pub validation: bool,
    pub log_path: PathBuf,
    pub gpu_launch_policy: &'a BrushGpuLaunchPolicy,
}

impl BrushGpuLaunchPolicy {
    pub fn from_acceleration(acceleration: &ColmapAccelerationStatus) -> Self {
        Self::resolve(cfg!(windows), acceleration)
    }

    fn resolve(is_windows: bool, acceleration: &ColmapAccelerationStatus) -> Self {
        let selected = acceleration.device.as_ref();
        let unambiguous_single_nvidia = is_windows
            && acceleration.use_gpu()
            && acceleration.detected_nvidia_device_count == 1
            && selected.is_some_and(|device| device.index == 0);
        let selection_reason = if unambiguous_single_nvidia {
            "single_nvidia_discrete_gpu"
        } else if !is_windows {
            "non_windows_default"
        } else if !acceleration.use_gpu() {
            "no_compatible_nvidia_gpu"
        } else {
            "ambiguous_multi_gpu_topology"
        };

        Self {
            expected_device_name: if unambiguous_single_nvidia {
                selected.map(|device| device.name.clone())
            } else {
                None
            },
            force_first_discrete_gpu: unambiguous_single_nvidia,
            disable_amd_switchable_graphics: unambiguous_single_nvidia,
            selection_reason,
        }
    }

    fn environment(&self) -> Vec<(OsString, OsString)> {
        self.environment_with_rust_log(std::env::var_os("RUST_LOG"))
    }

    fn environment_with_rust_log(
        &self,
        existing_rust_log: Option<OsString>,
    ) -> Vec<(OsString, OsString)> {
        let mut environment = Vec::new();
        if self.force_first_discrete_gpu {
            environment.push((
                OsString::from("CUBECL_WGPU_DEFAULT_DEVICE"),
                OsString::from("DiscreteGpu(0)"),
            ));
        }
        if self.disable_amd_switchable_graphics {
            environment.push((
                OsString::from("DISABLE_LAYER_AMD_SWITCHABLE_GRAPHICS_1"),
                OsString::from("1"),
            ));
        }
        let rust_log = match existing_rust_log {
            Some(value) if !value.is_empty() => {
                let mut combined = value;
                combined.push(",");
                combined.push(BRUSH_GPU_LOG_FILTER);
                combined
            }
            _ => OsString::from(BRUSH_GPU_LOG_FILTER),
        };
        environment.push((OsString::from("RUST_LOG"), rust_log));
        if std::env::var_os("RUST_BACKTRACE").is_none() {
            environment.push((OsString::from("RUST_BACKTRACE"), OsString::from("1")));
        }
        environment
    }

    pub fn log_summary(&self) -> String {
        format!(
            "backend=Vulkan selection={} expectedDevice={} cubeclDevice={} amdSwitchableLayerDisabled={}",
            self.selection_reason,
            self.expected_device_name.as_deref().unwrap_or("unverified"),
            if self.force_first_discrete_gpu {
                "DiscreteGpu(0)"
            } else {
                "automatic"
            },
            self.disable_amd_switchable_graphics
        )
    }
}

fn train_args(
    dataset: &Path,
    output_directory: &Path,
    preset: BrushTrainingPreset,
) -> Vec<OsString> {
    let mut args = vec![
        OsString::from("--total-train-iters"),
        preset.total_steps.to_string().into(),
        OsString::from("--max-resolution"),
        preset.max_resolution.to_string().into(),
        OsString::from("--refine-every"),
        preset.refine_every.to_string().into(),
    ];
    if let Some(max_splats) = preset.max_splats {
        args.extend([
            OsString::from("--max-splats"),
            max_splats.to_string().into(),
        ]);
    }
    if let Some(densification) = preset.densification {
        args.extend([
            OsString::from("--growth-grad-threshold"),
            densification.growth_grad_threshold.to_string().into(),
            OsString::from("--growth-select-fraction"),
            densification.growth_select_fraction.to_string().into(),
            OsString::from("--growth-stop-iter"),
            densification.growth_stop_iter.to_string().into(),
        ]);
    }
    args.extend([
        OsString::from("--export-every"),
        preset.total_steps.to_string().into(),
        OsString::from("--export-path"),
        output_directory.into(),
        OsString::from("--export-name"),
        OsString::from("final.ply.tmp"),
        dataset.into(),
    ]);
    args
}

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
    let resolved = BrushTrainingPreset{total_steps:preset.brush_iterations,max_resolution:preset.brush_max_resolution,refine_every:200,max_splats:None,densification:preset.brush_densification};
    let mut args = train_args(dataset, output_directory, resolved);
    args.pop();
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

pub fn precision_train_args(mut args:Vec<OsString>,precision:bool,validation:bool)->Vec<OsString>{
    let dataset=args.pop().expect("Training arguments include the dataset");
    if precision{args.extend(["--render-mode".into(),"mip".into(),"--sh-degree".into(),"3".into()]);}
    if validation{args.extend(["--eval-split-every".into(),"10".into(),"--eval-every".into(),"1000".into(),"--eval-save-to-disk".into()]);}
    args.push(dataset);args
}

pub async fn train(
    executable: &Path,
    dataset: &Path,
    output_directory: &Path,
    options: BrushTrainingOptions<'_>,
    manager: &ProcessManager,
    observer: Option<ProcessObserver>,
) -> Result<PathBuf> {
    tokio::fs::create_dir_all(output_directory).await?;
    let candidate = output_directory.join("final.ply.tmp");
    let alternate = output_directory.join("final.ply.tmp.ply");
    for partial in [&candidate, &alternate] {
        if partial.exists() {
            tokio::fs::remove_file(partial).await?;
        }
    }
    let environment = options.gpu_launch_policy.environment();
    let mut args=precision_train_args(train_args(dataset,output_directory,options.preset),options.precision,options.validation);
    args.pop();
    if demo_viewer_enabled(){args.push("--with-viewer".into());}
    if rerun_logging_enabled(){args.extend(["--rerun-enabled".into(),"--rerun-log-train-stats-every".into(),"100".into()]);}
    args.push(dataset.into());
    let output = manager
        .run_with_environment(
            ProcessSpec {
                executable: executable.to_path_buf(),
                args,
                working_directory: Some(output_directory.to_path_buf()),
                log_path: Some(options.log_path),
                observer,
            },
            &environment,
        )
        .await?;
    resolve_train_output(&output, candidate, alternate)
}

fn resolve_train_output(
    output: &ProcessOutput,
    candidate: PathBuf,
    alternate: PathBuf,
) -> Result<PathBuf> {
    if !output.success {
        let detail = output.failure_detail();
        if is_out_of_memory_detail(&detail) {
            return Err(SplatError::BrushOutOfMemory(detail));
        }
        if is_device_lost_detail(&detail) {
            return Err(SplatError::BrushDeviceLost(detail));
        }
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

fn is_out_of_memory_detail(detail: &str) -> bool {
    let normalized = detail.to_ascii_lowercase();
    [
        "out of memory",
        "outofmemory",
        "buffertoobig",
        "buffer too big",
        "failed to allocate",
        "allocation failed",
    ]
    .iter()
    .any(|token| normalized.contains(token))
}

fn is_device_lost_detail(detail: &str) -> bool {
    let normalized = detail.to_ascii_lowercase();
    [
        "devicelost",
        "device lost",
        "device_lost",
        "parent device is lost",
        "vk_error_device_lost",
        "dxgi_error_device_removed",
    ]
    .iter()
    .any(|token| normalized.contains(token))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        engines::{
            AccelerationReasonCode, AccelerationRequirements, ColmapBackend, GpuDetectionState,
            GpuDeviceInfo,
        },
        presets::{resolve_brush_training_preset, Quality},
    };

    fn args_as_strings(args: Vec<OsString>) -> Vec<String> {
        args.into_iter()
            .map(|arg| arg.to_string_lossy().into_owned())
            .collect()
    }

    fn acceleration(
        backend: ColmapBackend,
        count: usize,
        index: Option<u32>,
    ) -> ColmapAccelerationStatus {
        ColmapAccelerationStatus {
            backend,
            detection_state: if backend == ColmapBackend::Gpu {
                GpuDetectionState::Ready
            } else {
                GpuDetectionState::Unavailable
            },
            reason_code: if backend == ColmapBackend::Gpu {
                AccelerationReasonCode::GpuReady
            } else {
                AccelerationReasonCode::NoNvidiaGpu
            },
            reason: String::new(),
            device: index.map(|index| GpuDeviceInfo {
                index,
                name: "NVIDIA GeForce RTX 3060 Laptop GPU".into(),
                driver_version: "560.81".into(),
                compute_capability: "8.6".into(),
                total_memory_mb: Some(6_144),
            }),
            requirements: AccelerationRequirements {
                minimum_driver_version: "528.33".into(),
                minimum_compute_capability: "5.0".into(),
            },
            detected_nvidia_device_count: count,
        }
    }

    fn environment_value<'a>(
        environment: &'a [(OsString, OsString)],
        key: &str,
    ) -> Option<&'a str> {
        environment
            .iter()
            .find(|(candidate, _)| candidate == key)
            .and_then(|(_, value)| value.to_str())
    }

    #[test]
    fn windows_single_nvidia_policy_forces_first_discrete_gpu_for_brush_only() {
        let policy =
            BrushGpuLaunchPolicy::resolve(true, &acceleration(ColmapBackend::Gpu, 1, Some(0)));
        let environment = policy.environment_with_rust_log(Some("app=debug".into()));

        assert_eq!(
            environment_value(&environment, "CUBECL_WGPU_DEFAULT_DEVICE"),
            Some("DiscreteGpu(0)")
        );
        assert_eq!(
            environment_value(&environment, "DISABLE_LAYER_AMD_SWITCHABLE_GRAPHICS_1"),
            Some("1")
        );
        assert_eq!(
            environment_value(&environment, "RUST_LOG"),
            Some("app=debug,brush_cli=info,brush_process=info,cubecl_wgpu=info,burn_wgpu=info")
        );
        assert!(!environment.iter().any(|(key, _)| key == "WGPU_BACKEND"));
        assert!(policy.log_summary().contains("backend=Vulkan"));
        assert!(policy
            .log_summary()
            .contains("expectedDevice=NVIDIA GeForce RTX 3060 Laptop GPU"));
    }

    #[test]
    fn ambiguous_or_cpu_topologies_keep_automatic_device_selection() {
        for status in [
            acceleration(ColmapBackend::Gpu, 2, Some(0)),
            acceleration(ColmapBackend::Gpu, 1, Some(1)),
            acceleration(ColmapBackend::Cpu, 0, None),
        ] {
            let policy = BrushGpuLaunchPolicy::resolve(true, &status);
            let environment = policy.environment_with_rust_log(None);
            assert_eq!(
                environment_value(&environment, "CUBECL_WGPU_DEFAULT_DEVICE"),
                None
            );
            assert_eq!(
                environment_value(&environment, "DISABLE_LAYER_AMD_SWITCHABLE_GRAPHICS_1"),
                None
            );
            assert_eq!(
                environment_value(&environment, "RUST_LOG"),
                Some(BRUSH_GPU_LOG_FILTER)
            );
        }
    }

    #[test]
    fn non_windows_policy_never_forces_windows_gpu_workarounds() {
        let policy =
            BrushGpuLaunchPolicy::resolve(false, &acceleration(ColmapBackend::Gpu, 1, Some(0)));
        let environment = policy.environment_with_rust_log(None);
        assert_eq!(
            environment_value(&environment, "CUBECL_WGPU_DEFAULT_DEVICE"),
            None
        );
        assert_eq!(
            environment_value(&environment, "DISABLE_LAYER_AMD_SWITCHABLE_GRAPHICS_1"),
            None
        );
    }

    #[test]
    fn new_cli_preserves_iterations_exports_and_unicode_paths() {
        let dataset = Path::new("中文 数据集");
        let output = Path::new("中文 导出目录");
        let preset = resolve_brush_training_preset(Quality::Balanced, false, None, 1_600, 0).preset;
        let args = train_args(dataset, output, preset);
        assert_eq!(args.last(), Some(&dataset.as_os_str().to_owned()));
        assert!(args.windows(2).any(|pair| pair[0] == "--total-train-iters"
            && pair[1] == OsString::from(preset.total_steps.to_string())));
        assert!(args
            .windows(2)
            .any(|pair| pair[0] == "--export-path" && pair[1] == output.as_os_str()));
        assert!(args
            .windows(2)
            .any(|pair| pair[0] == "--export-name" && pair[1] == "final.ply.tmp"));
        assert!(!args
            .iter()
            .any(|arg| arg == "--total-steps" || arg == "--lod-levels"));
    }

    #[test]
    fn completed_iterations_or_zero_exit_do_not_establish_successful_export() {
        let directory = tempfile::tempdir().unwrap();
        let candidate = directory.path().join("final.ply.tmp");
        let alternate = directory.path().join("final.ply.tmp.ply");
        let output = ProcessOutput {
            success: true,
            exit_code: Some(0),
            stdout: "Training progress: iteration=4 total=4\nDone training!".into(),
            stderr: "Export at iteration 4 failed".into(),
        };
        assert!(resolve_train_output(&output, candidate.clone(), alternate.clone()).is_err());
        std::fs::write(&candidate, "candidate is validated as PLY by the pipeline").unwrap();
        assert_eq!(
            resolve_train_output(&output, candidate.clone(), alternate.clone()).unwrap(),
            candidate
        );
        let failed = ProcessOutput {
            success: false,
            exit_code: Some(1),
            stderr: "VK_ERROR_DEVICE_LOST".into(),
            ..output
        };
        assert!(matches!(
            resolve_train_output(&failed, candidate, alternate),
            Err(SplatError::BrushDeviceLost(_))
        ));
    }

    #[test]
    fn optimization_off_passes_compatible_densification_for_all_qualities() {
        for (quality, steps, resolution) in [
            (Quality::Fast, "8000", "1200"),
            (Quality::Balanced, "15000", "1600"),
            (Quality::High, "30000", "2000"),
        ] {
            let args = args_as_strings(train_args(
                Path::new("dataset"),
                Path::new("output"),
                resolve_brush_training_preset(quality, false, None, 3_840, 0).preset,
            ));
            for pair in [
                ["--total-train-iters", steps],
                ["--max-resolution", resolution],
                ["--refine-every", "200"],
                ["--growth-grad-threshold", "0.0025"],
                ["--growth-select-fraction", "0.1"],
                ["--growth-stop-iter", "15000"],
            ] {
                assert!(
                    args.windows(2).any(|actual| actual == pair),
                    "{quality}: {pair:?}"
                );
            }
            assert!(!args.iter().any(|arg| arg == "--max-splats"));
            assert!(!args.iter().any(|arg| arg == "--split-at-screen-size"));
            assert!(!args.iter().any(|arg| arg == "--min-scale-factor"));
        }
    }

    #[test]
    fn high_profile_passes_all_bounded_training_overrides() {
        let preset =
            resolve_brush_training_preset(Quality::High, true, Some(8_192), 3_840, 60_000).preset;
        let args = args_as_strings(train_args(
            Path::new("dataset"),
            Path::new("output"),
            preset,
        ));

        assert!(args
            .windows(2)
            .any(|pair| pair == ["--growth-grad-threshold", "0.0016"]));
        assert!(args
            .windows(2)
            .any(|pair| pair == ["--growth-select-fraction", "0.14"]));
        assert!(args
            .windows(2)
            .any(|pair| pair == ["--growth-stop-iter", "23000"]));
        assert!(args
            .windows(2)
            .any(|pair| pair == ["--refine-every", "200"]));
        assert!(args
            .windows(2)
            .any(|pair| pair == ["--max-splats", "1500000"]));
    }

    #[test]
    fn fast_and_balanced_pass_efficiency_densification_profiles() {
        let fast = args_as_strings(train_args(
            Path::new("dataset"),
            Path::new("output"),
            resolve_brush_training_preset(Quality::Fast, true, Some(8_192), 3_840, 0).preset,
        ));
        assert!(fast
            .windows(2)
            .any(|pair| pair == ["--growth-grad-threshold", "0.0022"]));
        assert!(fast
            .windows(2)
            .any(|pair| pair == ["--growth-select-fraction", "0.11"]));
        assert!(fast
            .windows(2)
            .any(|pair| pair == ["--growth-stop-iter", "7000"]));
        assert!(fast
            .windows(2)
            .any(|pair| pair == ["--max-splats", "500000"]));

        let balanced = args_as_strings(train_args(
            Path::new("dataset"),
            Path::new("output"),
            resolve_brush_training_preset(Quality::Balanced, true, Some(8_192), 3_840, 0).preset,
        ));
        assert!(balanced
            .windows(2)
            .any(|pair| pair == ["--growth-grad-threshold", "0.002"]));
        assert!(balanced
            .windows(2)
            .any(|pair| pair == ["--growth-select-fraction", "0.12"]));
        assert!(balanced
            .windows(2)
            .any(|pair| pair == ["--growth-stop-iter", "13000"]));
        assert!(balanced
            .windows(2)
            .any(|pair| pair == ["--max-splats", "1000000"]));
    }

    #[test]
    fn gpu_failure_classifiers_keep_oom_and_device_loss_separate() {
        assert!(is_out_of_memory_detail("OutOfMemory while allocating"));
        assert!(is_out_of_memory_detail("BufferTooBig(2290420416)"));
        assert!(is_out_of_memory_detail("GPU allocation failed"));
        assert!(!is_out_of_memory_detail("DeviceLost: driver reset"));
        assert!(is_device_lost_detail("DeviceLost: driver reset"));
        assert!(is_device_lost_detail("Parent device is lost"));
        assert!(is_device_lost_detail("VK_ERROR_DEVICE_LOST"));
        assert!(!is_device_lost_detail("process exited with code 101"));
        assert!(!is_device_lost_detail(
            "process exited with code -1073740791"
        ));
        assert!(!is_out_of_memory_detail("process exited with code 1"));
    }
}

#[cfg(test)]mod precision_tests{
 use super::*;
 #[test]fn native_precision_flags_preserve_dataset_and_default_arguments(){
  let base=vec!["--total-train-iters".into(),"30000".into(),"data set".into()];
  assert_eq!(precision_train_args(base.clone(),false,false),base);
  let args=precision_train_args(base,true,true);let text:Vec<_>=args.iter().map(|x|x.to_string_lossy()).collect();
  assert_eq!(text.last().unwrap(),"data set");assert!(text.windows(2).any(|v|v==["--render-mode","mip"]));assert!(text.windows(2).any(|v|v==["--sh-degree","3"]));assert!(text.windows(2).any(|v|v==["--eval-split-every","10"]));
 }
}
