use std::path::PathBuf;

use clap::{Parser, Subcommand};
use ooo_splat::{
    engines::{ffmpeg::extract_uniform_frames, ffprobe::probe_video},
    error::{Result, SplatError},
    pipeline::runner::{default_engine_paths, PipelineRunner},
    presets::Quality,
    process::ProcessManager,
    video::{
        analyze_image_sequence, create_image_plan, prepare_image_sequence, FrameSelectionStrategy,
        UniformRatioFrameSelection,
    },
};

#[derive(Debug, Parser)]
#[command(name = "splatstudio", version, about = "MacauSplat local pipeline CLI")]
struct Cli {
    /// Override the bundled engine directory (also supports OOOSPLAT_ENGINE_DIR).
    #[arg(long, global = true)]
    engine_dir: Option<PathBuf>,
    #[command(subcommand)]
    command: Commands,
}

#[derive(Debug, Subcommand)]
enum Commands {
    /// Validate FFmpeg, FFprobe, CPU COLMAP and Brush.
    Health,
    /// Read video metadata or image-sequence information.
    Probe { input: PathBuf },
    /// Show the video frame plan or image-sequence plan.
    Plan {
        input: PathBuf,
        #[arg(long, value_enum, default_value_t = Quality::Balanced)]
        quality: Quality,
    },
    /// Prepare video frames or an image sequence and optional COLMAP masks.
    Extract {
        input: PathBuf,
        output: PathBuf,
        #[arg(long, value_enum, default_value_t = Quality::Balanced)]
        quality: Quality,
    },
    /// Run the end-to-end pipeline after all fixed engine CLIs are verified.
    Generate {
        input: PathBuf,
        /// Override the remembered projects root (useful for diagnostics).
        #[arg(long)]
        projects_root: Option<PathBuf>,
        #[arg(long, value_enum, default_value_t = Quality::Balanced)]
        quality: Quality,
    },
    /// Audit a finished project: frames, registered camera poses, sparse points.
    ///
    /// This is the first step of the dataset pipeline — it reports whether a
    /// reconstruction contains what automatic annotation needs.
    Inspect {
        /// Project directory (the one containing `work/colmap`).
        project: PathBuf,
    },
    /// Fit a 3D bounding box around the reconstructed scene (writes `box.json`).
    ///
    /// The up axis is inferred from the camera trajectory, because COLMAP's world
    /// frame is arbitrary and has no gravity direction.
    FitBox {
        project: PathBuf,
        #[arg(long, default_value = ".")]
        out: PathBuf,
        /// Percentile trim per axis (0.02 keeps the 2%..98% range, max 0.2).
        ///
        /// The range is enforced **here and only here**: `fit_box_from_xyz` no longer
        /// clamps the value, because a silent clamp made `--trim 0.9` mean the same
        /// thing as `--trim 0.2` while looking like it had an effect.
        #[arg(long, default_value_t = 0.02, value_parser = parse_trim)]
        trim: f64,
        /// Use the world Y axis as "up" instead of inferring it from the trajectory.
        #[arg(long)]
        no_infer_up: bool,
        /// Human seed rectangles from seed-editor.html; at least two registered views.
        #[arg(long)]
        seeds: Option<PathBuf>,
        #[arg(long, default_value_t = 2)]
        min_seed_support: usize,
    },
    /// Project a 3D box into every registered frame and write YOLO labels.
    ///
    /// Outputs: `labels/*.txt`, `annotations.json`, `annotations.csv`, `qa.html`.
    Annotate {
        project: PathBuf,
        /// Path to `box.json` produced by `fit-box`.
        #[arg(long)]
        box_path: PathBuf,
        /// Class index written into the YOLO labels.
        #[arg(long, default_value_t = 0)]
        class_id: usize,
        #[arg(long, default_value = ".")]
        out: PathBuf,
        /// Minimum visible box corners (of 8) for a frame to be kept.
        #[arg(long, default_value_t = 4)]
        min_corners: usize,
        /// Minimum number of in-box sparse points observed in the frame (occlusion proxy).
        #[arg(long, default_value_t = 3)]
        min_observed: usize,
        /// Strict mode: keep only frames where all 8 box corners are visible.
        ///
        /// When a corner lies behind the camera it cannot be projected, so the
        /// projected box may underestimate the true outline. Enable this when the
        /// label must fully contain the object.
        #[arg(long)]
        require_all_corners: bool,
        /// Frames shown in the HTML review sheet.
        #[arg(long, default_value_t = 60)]
        qa_limit: usize,
    },
    /// Live progress dashboard for a project (FFmpeg extraction / COLMAP / Brush).
    ///
    /// Writes a self-refreshing HTML page (no web server required) and prints a
    /// text summary. Use `--watch` to keep regenerating it while the pipeline runs.
    Monitor {
        project: PathBuf,
        /// Where to write the dashboard (default: `<project>/monitor.html`).
        #[arg(long)]
        out: Option<PathBuf>,
        /// Keep regenerating the page (pair with the page's meta refresh).
        #[arg(long)]
        watch: bool,
        /// Refresh interval in seconds.
        #[arg(long, default_value_t = 2)]
        interval: u64,
        /// Frames directory as seen from the HTML file.
        #[arg(long, default_value = "work/frames")]
        frames_href: String,
    },
    /// Merge this project's auto-annotations into a YOLO dataset.
    ///
    /// Closes the loop from capture to training: reads `annotation/labels/`,
    /// copies the matching frames, and rewrites the class index to the
    /// dataset's own numbering (source and dataset indices often differ).
    ExportDataset {
        /// Project directory that has already been annotated.
        project: PathBuf,
        /// Dataset root containing `data.yaml`, `images/`, `labels/`.
        #[arg(long)]
        into: PathBuf,
        /// Target class name; must exist in the dataset's `names:` list.
        #[arg(long)]
        class: String,
        /// Target split.
        #[arg(long, default_value = "train")]
        split: String,
        /// Source tag to embed in filenames (for provenance).
        #[arg(long, default_value = "splat")]
        source: String,
        /// Report what would happen without writing anything.
        #[arg(long)]
        dry_run: bool,
    },
    /// Inspect the COLMAP snapshot point cloud (grows during mapping).
    ///
    /// This is what the in-app 3D view uses; the CLI form prints statistics so
    /// the data path can be verified without launching the GUI.
    Cloud {
        project: PathBuf,
        /// Maximum number of points sent to the viewer.
        #[arg(long, default_value_t = 60000)]
        max_points: usize,
        /// Write the cloud as a PLY file for external viewing.
        #[arg(long)]
        ply: Option<PathBuf>,
    },
}

/// `--trim` 的驗證（**唯一的驗證點**）：只接受 0.0..=0.2。
///
/// 為什麼要自己做而不用 `clap` 的 `.range()`：`clap` 的範圍建構器只支援整數型別
/// （`RangedI64ValueParser`／`RangedU64ValueParser`），`--trim` 是比例值。
/// 用 `value_parser` 自訂函式一樣是把驗證放在命令列層——超出範圍會直接被擋下並回報，
/// 而不是像舊版那樣被 `fit_box_from_xyz` 裡的 `clamp(0.0, 0.2)` 悄悄吞掉，
/// 讓 `--trim 0.9` 與 `--trim 0.2` 同義（使用者以為自己調了參數卻毫無效果）。
fn parse_trim(raw: &str) -> std::result::Result<f64, String> {
    let value: f64 = raw
        .parse()
        .map_err(|_| format!("--trim 需要一個數字（0.0..=0.2），收到 {raw:?}"))?;
    if !value.is_finite() {
        return Err(format!("--trim 必須是有限數字，收到 {raw:?}"));
    }
    if !(0.0..=0.2).contains(&value) {
        return Err(format!(
            "--trim 必須介於 0.0 與 0.2 之間（收到 {value}）；0.02 代表各軸取 2%~98% 範圍"
        ));
    }
    Ok(value)
}

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .with_target(false)
        .init();
    if let Err(error) = execute(Cli::parse()).await {
        eprintln!("{error}");
        std::process::exit(1);
    }
}

async fn execute(cli: Cli) -> Result<()> {
    let engines = default_engine_paths(cli.engine_dir);
    match cli.command {
        Commands::Health => {
            println!(
                "{}",
                serde_json::to_string_pretty(&engines.check_all().await)?
            );
        }
        Commands::Probe { input } => {
            if input.is_dir() {
                println!(
                    "{}",
                    serde_json::to_string_pretty(&analyze_image_sequence(&input)?)?
                );
            } else {
                let video =
                    probe_video(&engines.ffprobe, &input, None, &ProcessManager::new()).await?;
                println!("{}", serde_json::to_string_pretty(&video)?);
            }
        }
        Commands::Plan { input, quality } => {
            let plan = if input.is_dir() {
                create_image_plan(&analyze_image_sequence(&input)?, &quality.preset())
            } else {
                let video =
                    probe_video(&engines.ffprobe, &input, None, &ProcessManager::new()).await?;
                UniformRatioFrameSelection.create_plan(&video, &quality.preset())
            };
            println!("{}", serde_json::to_string_pretty(&plan)?);
        }
        Commands::Extract {
            input,
            output,
            quality,
        } => {
            let masks = output
                .parent()
                .unwrap_or_else(|| std::path::Path::new("."))
                .join("masks");
            if input.is_dir() {
                let extraction = prepare_image_sequence(&input, &output, &masks)?;
                println!(
                    "prepared {} images in {} and {} masks in {}",
                    extraction.image_count,
                    output.display(),
                    extraction.mask_count,
                    masks.display()
                );
                return Ok(());
            }
            ensure_engine(&engines.ffprobe)?;
            ensure_engine(&engines.ffmpeg)?;
            let video = probe_video(&engines.ffprobe, &input, None, &ProcessManager::new()).await?;
            let plan = UniformRatioFrameSelection.create_plan(&video, &quality.preset());
            let extraction = extract_uniform_frames(
                &engines.ffmpeg,
                &input,
                &output,
                &masks,
                &plan,
                video.has_alpha,
                None,
                &ProcessManager::new(),
                None,
            )
            .await?;
            if extraction.has_alpha {
                println!(
                    "extracted {} RGBA frames to {} and {} masks to {}",
                    extraction.frame_count,
                    output.display(),
                    extraction.mask_count,
                    masks.display()
                );
            } else {
                println!(
                    "extracted {} frames to {}",
                    extraction.frame_count,
                    output.display()
                );
            }
        }
        Commands::Generate {
            input,
            projects_root,
            quality,
        } => {
            let runner = PipelineRunner::new(engines, |event| {
                eprintln!(
                    "{:>6.2}% {:?}: {}",
                    event.progress, event.stage, event.message
                );
            });
            let result = match projects_root {
                Some(root) => {
                    runner
                        .generate_for_diagnostics(&input, quality, &root)
                        .await?
                }
                None => {
                    let root = ooo_splat::project::catalog::load_settings()
                        .await?
                        .projects_root;
                    runner.generate(&input, quality, &root).await?
                }
            };
            println!("{}", serde_json::to_string_pretty(&result)?);
        }
        Commands::Inspect { project } => {
            let audit = ooo_splat::reconstruction::inspect::audit_project(&project)
                .map_err(|e| SplatError::Process(format!("專案審計失敗：{e}")))?;
            println!("{}", serde_json::to_string_pretty(&audit)?);
            if !audit.ready_for_annotation {
                eprintln!("⚠ 此專案目前不適合自動標註（尚有 blocker）");
            }
        }
        Commands::FitBox {
            project,
            out,
            trim,
            no_infer_up,
            seeds,
            min_seed_support,
        } => {
            use ooo_splat::reconstruction::autolabel::{fit_box_from_points, FitOptions};
            use ooo_splat::reconstruction::colmap::SparseReconstruction;

            let reconstruction = SparseReconstruction::load(project.join("work/colmap"))
                .or_else(|_| SparseReconstruction::load(&project))
                .map_err(|e| SplatError::Process(format!("讀取稀疏重建失敗：{e}")))?;
            let options = FitOptions {
                trim,
                infer_up: !no_infer_up,
                ..FitOptions::default()
            };
            let selection = seeds.as_ref().map(|path| -> Result<_> {
                let seeds = serde_json::from_slice::<ooo_splat::reconstruction::target_selection::SeedDocument>(&std::fs::read(path)?)?;
                ooo_splat::reconstruction::target_selection::select_target_points(&reconstruction, &seeds, min_seed_support).map_err(SplatError::Process)
            }).transpose()?;
            let box3d = if let Some(target) = &selection {
                ooo_splat::reconstruction::autolabel::fit_box_from_xyz(&target.points, &reconstruction.images, &options)
            } else { fit_box_from_points(&reconstruction, &options) }
                .map_err(|e| SplatError::Process(format!("自動擬合包圍盒失敗：{e}")))?;
            std::fs::create_dir_all(&out)?;
            let path = out.join("box.json");
            std::fs::write(&path, serde_json::to_string_pretty(&box3d)?)?;
            if let Some(target) = selection { std::fs::write(out.join("target-selection.json"), serde_json::to_vec_pretty(&target)?)?; }
            println!(
                "包圍盒：{:.2} × {:.2} × {:.2} 重建單位（未校準），中心 [{:.2}, {:.2}, {:.2}]，上軸：{}，取自 {} 個點",
                box3d.size[0],
                box3d.size[1],
                box3d.size[2],
                box3d.center[0],
                box3d.center[1],
                box3d.center[2],
                box3d.up_source.label(),
                box3d.fitted_from_points
            );
            if box3d.up_source.is_estimated() {
                println!(
                    "⚠ 上軸是後備值（世界 Y 軸），建築高度可能不準；請用 qa.html 人工複核框位"
                );
            }
            println!("已寫入 {}", path.display());
        }
        Commands::Annotate {
            project,
            box_path,
            class_id,
            out,
            min_corners,
            min_observed,
            require_all_corners,
            qa_limit,
        } => {
            use ooo_splat::reconstruction::autolabel::{
                annotate_reconstruction, summarize, write_annotations_csv, write_qa_html,
                write_yolo_labels, AnnotateOptions, Box3D,
            };
            use ooo_splat::reconstruction::colmap::SparseReconstruction;
            use ooo_splat::reconstruction::inspect;

            let reconstruction = SparseReconstruction::load(project.join("work/colmap"))
                .or_else(|_| SparseReconstruction::load(&project))
                .map_err(|e| SplatError::Process(format!("讀取稀疏重建失敗：{e}")))?;
            let box3d: Box3D = serde_json::from_str(&std::fs::read_to_string(&box_path)?)
                .map_err(|e| SplatError::Process(format!("box.json 解析失敗：{e}")))?;
            let options = AnnotateOptions {
                min_visible_corners: min_corners,
                min_observed_points: min_observed,
                require_all_corners,
                ..AnnotateOptions::default()
            };
            let annotations = annotate_reconstruction(&reconstruction, &box3d, &options);
            let mut summary = summarize(&annotations);
            std::fs::create_dir_all(&out)?;
            let labels_dir = out.join("labels");
            let labels = write_yolo_labels(&labels_dir, &annotations, class_id)?;
            summary.labels_removed = labels.removed;
            std::fs::write(
                out.join("annotations.json"),
                serde_json::to_string_pretty(&annotations)?,
            )?;
            write_annotations_csv(out.join("annotations.csv"), &annotations)?;
            // QA 表以 out/ 為基準指向影格目錄：用實際存在的那一個（三個候選之一），
            // 不可以寫死 ../work/frames
            let frames_href = inspect::frames_href(&project, &out)
                .unwrap_or_else(|| inspect::FALLBACK_FRAMES_HREF.to_string());
            write_qa_html(
                out.join("qa.html"),
                &frames_href,
                &annotations,
                &summary,
                &box3d,
                qa_limit,
            )?;
            let quality = ooo_splat::reconstruction::annotation_quality::assess(&reconstruction, &box3d, &annotations);
            ooo_splat::reconstruction::annotation_quality::write_report(&out, &frames_href, &quality, qa_limit)?;
            println!("複核分級：候選 {} · 需複核 {} · 暫不使用 {}（啟發式，不是準確率）", quality.candidate_frames, quality.review_frames, quality.rejected_frames);
            println!(
                "影格 {} 張 → 採用 {} 張（{:.1}%），已寫出 {} 個 YOLO 標籤",
                summary.frames_total,
                summary.frames_accepted,
                summary.acceptance_ratio * 100.0,
                labels.written
            );
            if labels.removed > 0 {
                println!(
                    "（重跑：已清掉上一輪的 {} 個舊標籤，避免不再採用的影格留下錯誤正樣本）",
                    labels.removed
                );
            }
            if !summary.rejection_reasons.is_empty() {
                println!("主要略過原因：");
                for (reason, n) in summary.rejection_reasons.iter().take(5) {
                    println!("  {n:4} 張：{reason}");
                }
            }
            if !summary.warnings.is_empty() {
                println!("警示（仍會寫出標籤，但請人工複核）：");
                for (message, n) in summary.warnings.iter().take(5) {
                    println!("  {n:4} 張：{message}");
                }
            }
            println!("QA 影格連結：{frames_href}");
            println!(
                "輸出：{}（labels/、annotations.json、annotations.csv、qa.html）",
                out.display()
            );
        }
        Commands::Monitor {
            project,
            out,
            watch,
            interval,
            frames_href,
        } => {
            use ooo_splat::pipeline::monitor::{default_output, read_snapshot, render_html};

            let out_path = out.unwrap_or_else(|| default_output(&project));
            if let Some(parent) = out_path.parent() {
                if !parent.as_os_str().is_empty() {
                    std::fs::create_dir_all(parent)?;
                }
            }
            let title = format!(
                "{} · 即時監視",
                project
                    .file_name()
                    .map(|s| s.to_string_lossy().to_string())
                    .unwrap_or_default()
            );
            loop {
                let snap = read_snapshot(&project);
                std::fs::write(
                    &out_path,
                    render_html(&snap, &frames_href, interval, &title),
                )?;
                // 每次迭代都印出文字摘要，方便在終端機直接看（不必開瀏覽器）
                println!(
                    "[{}] 階段：{}\n    抽幀 {} 張（速度 {}）｜特徵 {} 張｜匹配 {}/{}｜已註冊 {}（{}）｜三維點 {}｜快照 {}",
                    chrono::Local::now().format("%H:%M:%S"),
                    snap.stage,
                    snap.frames_on_disk,
                    snap.ffmpeg.speed.map(|s| format!("{s:.1}x")).unwrap_or_else(|| "—".into()),
                    snap.colmap.feature_images,
                    snap.colmap.matching_done,
                    snap.colmap.matching_total,
                    snap.colmap.registered_images,
                    if snap.colmap.registered_source == "model" { "讀自模型" } else { "日誌估算" },
                    snap.colmap.points,
                    snap.colmap.snapshots,
                );
                if !watch {
                    println!("已寫入 {}", out_path.display());
                    println!(
                        "用瀏覽器開啟即可（頁面每 {interval} 秒自動重整）；要持續更新請加 --watch"
                    );
                    break;
                }
                tokio::time::sleep(std::time::Duration::from_secs(interval.max(1))).await;
            }
        }
        Commands::Cloud {
            project,
            max_points,
            ply,
        } => {
            use ooo_splat::reconstruction::cloud::{list_snapshots, load_snapshot_cloud};

            let cloud = load_snapshot_cloud(&project, max_points)
                .map_err(|e| SplatError::Process(format!("讀取快照失敗：{e}")))?;
            let snaps = list_snapshots(&project);
            println!(
                "快照 {} 份｜目前檢視：{}｜點數 {} / {}｜相機 {} 台",
                snaps.len(),
                cloud.snapshot.clone().unwrap_or_else(|| "（無）".into()),
                cloud.point_count,
                cloud.total_points,
                cloud.cameras.len() / 3,
            );
            println!(
                "包圍盒：x {:.2}..{:.2}｜y {:.2}..{:.2}｜z {:.2}..{:.2}{}",
                cloud.bounds[0],
                cloud.bounds[3],
                cloud.bounds[1],
                cloud.bounds[4],
                cloud.bounds[2],
                cloud.bounds[5],
                if cloud.is_final {
                    "｜來源：最終模型"
                } else {
                    "｜來源：中間快照"
                },
            );
            if let Some(path) = ply {
                let mut out = String::from("ply\nformat ascii 1.0\n");
                out.push_str(&format!("element vertex {}\n", cloud.point_count));
                out.push_str("property float x\nproperty float y\nproperty float z\n");
                out.push_str(
                    "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n",
                );
                for i in 0..cloud.point_count {
                    out.push_str(&format!(
                        "{:.4} {:.4} {:.4} {} {} {}\n",
                        cloud.positions[i * 3],
                        cloud.positions[i * 3 + 1],
                        cloud.positions[i * 3 + 2],
                        cloud.colors[i * 3],
                        cloud.colors[i * 3 + 1],
                        cloud.colors[i * 3 + 2],
                    ));
                }
                std::fs::write(&path, out)?;
                println!("已輸出 PLY：{}", path.display());
            }
        }
        Commands::ExportDataset {
            project,
            into,
            class,
            split,
            source,
            dry_run,
        } => {
            use ooo_splat::reconstruction::export::{export_annotations, ExportOptions};

            let report = export_annotations(
                &project,
                &into,
                &ExportOptions {
                    class_name: class,
                    split,
                    source,
                    dry_run,
                },
            )
            .map_err(|e| SplatError::Process(format!("匯出失敗：{e}")))?;

            println!(
                "{}類別「{}」（資料集索引 {}）→ {}/{}",
                if dry_run { "[dry-run] " } else { "" },
                report.class_name,
                report.class_id,
                report.dataset,
                report.split
            );
            println!(
                "  寫入影像 {}｜標籤 {}｜略過：已存在 {}、找不到影格 {}、標籤無效 {}",
                report.images_written,
                report.labels_written,
                report.skipped_exists,
                report.skipped_missing_frame,
                report.skipped_invalid_label
            );
            if !report.samples.is_empty() {
                println!("  範例：{}", report.samples.join("、"));
            }
            if !dry_run && report.images_written > 0 {
                println!("  下一步：重新訓練 detector，資料集已更新");
            }
        }
    }
    Ok(())
}

fn ensure_engine(path: &std::path::Path) -> Result<()> {
    if path.is_file() {
        Ok(())
    } else {
        Err(SplatError::EngineMissing(path.display().to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `--trim` 的驗證必須在命令列層擋下（範圍 0.0..=0.2）。
    ///
    /// 舊版沒有任何驗證，而 `fit_box_from_xyz` 又會 `clamp(0.0, 0.2)`：
    /// 於是 `--trim 0.9` 與 `--trim 0.2` 完全同義，使用者以為參數生效了卻毫無差別。
    /// 現在超範圍會直接讓命令列解析失敗（而不是被悄悄改掉）。
    #[test]
    fn trim_argument_rejects_values_outside_the_supported_range() {
        assert_eq!(parse_trim("0.02").unwrap(), 0.02);
        assert_eq!(parse_trim("0").unwrap(), 0.0);
        assert_eq!(parse_trim("0.2").unwrap(), 0.2);
        for invalid in ["0.9", "-0.1", "0.200001", "abc", "", "NaN", "inf"] {
            let error = parse_trim(invalid).expect_err("這個值不該被接受");
            assert!(!error.is_empty(), "錯誤訊息不可為空");
        }
    }

    /// 經由真正的 `clap` 解析路徑驗證：超範圍的 `--trim` 會被擋下，合法值可以通過。
    #[test]
    fn cli_parse_blocks_an_out_of_range_trim() {
        let ok = Cli::try_parse_from(["splatstudio", "fit-box", "C:/project", "--trim", "0.2"]);
        assert!(ok.is_ok(), "0.2 應該被接受：{ok:?}");

        let refused =
            Cli::try_parse_from(["splatstudio", "fit-box", "C:/project", "--trim", "0.9"]);
        let error = refused.expect_err("0.9 必須被命令列擋下");
        let message = error.to_string();
        assert!(
            message.contains("--trim"),
            "錯誤訊息應指出是哪個參數：{message}"
        );
    }
}
