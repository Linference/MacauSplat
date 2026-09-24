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
        /// Percentile trim per axis (0.02 keeps the 2%..98% range).
        #[arg(long, default_value_t = 0.02)]
        trim: f64,
        /// Use the world Y axis as "up" instead of inferring it from the trajectory.
        #[arg(long)]
        no_infer_up: bool,
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
            let box3d = fit_box_from_points(&reconstruction, &options)
                .ok_or_else(|| SplatError::Process("點雲不足，無法自動擬合包圍盒（可用 --no-infer-up 或人工指定）".into()))?;
            std::fs::create_dir_all(&out)?;
            let path = out.join("box.json");
            std::fs::write(&path, serde_json::to_string_pretty(&box3d)?)?;
            println!(
                "包圍盒：{:.2} × {:.2} × {:.2} m，中心 [{:.2}, {:.2}, {:.2}]，上軸{}，取自 {} 個點",
                box3d.size[0],
                box3d.size[1],
                box3d.size[2],
                box3d.center[0],
                box3d.center[1],
                box3d.center[2],
                if box3d.up_axis_estimated { "為推定值（軌跡退化）" } else { "由相機軌跡推定" },
                box3d.fitted_from_points
            );
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
                annotate_reconstruction, summarize, write_annotations_csv, write_qa_html, write_yolo_labels,
                AnnotateOptions, Box3D,
            };
            use ooo_splat::reconstruction::colmap::SparseReconstruction;

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
            let summary = summarize(&annotations);
            std::fs::create_dir_all(&out)?;
            let labels_dir = out.join("labels");
            let written = write_yolo_labels(&labels_dir, &annotations, class_id)?;
            std::fs::write(out.join("annotations.json"), serde_json::to_string_pretty(&annotations)?)?;
            write_annotations_csv(out.join("annotations.csv"), &annotations)?;
            // QA 表以 out/ 為基準指向影格目錄
            write_qa_html(
                out.join("qa.html"),
                "../work/frames",
                &annotations,
                &summary,
                &box3d,
                qa_limit,
            )?;
            println!(
                "影格 {} 張 → 採用 {} 張（{:.1}%），已寫出 {} 個 YOLO 標籤",
                summary.frames_total,
                summary.frames_accepted,
                summary.acceptance_ratio * 100.0,
                written
            );
            if !summary.rejection_reasons.is_empty() {
                println!("主要略過原因：");
                for (reason, n) in summary.rejection_reasons.iter().take(5) {
                    println!("  {n:4} 張：{reason}");
                }
            }
            println!("輸出：{}（labels/、annotations.json、annotations.csv、qa.html）", out.display());
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
            let title = format!("{} · 即時監視", project.file_name().map(|s| s.to_string_lossy().to_string()).unwrap_or_default());
            loop {
                let snap = read_snapshot(&project);
                std::fs::write(&out_path, render_html(&snap, &frames_href, interval, &title))?;
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
                    println!("用瀏覽器開啟即可（頁面每 {interval} 秒自動重整）；要持續更新請加 --watch");
                    break;
                }
                tokio::time::sleep(std::time::Duration::from_secs(interval.max(1))).await;
            }
        }
        Commands::Cloud { project, max_points, ply } => {
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
                cloud.bounds[0], cloud.bounds[3], cloud.bounds[1], cloud.bounds[4],
                cloud.bounds[2], cloud.bounds[5],
                if cloud.is_final { "｜來源：最終模型" } else { "｜來源：中間快照" },
            );
            if let Some(path) = ply {
                let mut out = String::from("ply\nformat ascii 1.0\n");
                out.push_str(&format!("element vertex {}\n", cloud.point_count));
                out.push_str("property float x\nproperty float y\nproperty float z\n");
                out.push_str("property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n");
                for i in 0..cloud.point_count {
                    out.push_str(&format!(
                        "{:.4} {:.4} {:.4} {} {} {}\n",
                        cloud.positions[i * 3], cloud.positions[i * 3 + 1], cloud.positions[i * 3 + 2],
                        cloud.colors[i * 3], cloud.colors[i * 3 + 1], cloud.colors[i * 3 + 2],
                    ));
                }
                std::fs::write(&path, out)?;
                println!("已輸出 PLY：{}", path.display());
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
