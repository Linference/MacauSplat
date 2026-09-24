//! 即時監視：把管線各階段的進度解析出來，產生可自動刷新的展示頁。
//!
//! ## 為什麼需要這個模組
//! 重建流程有四個階段，各自的「看得見的程度」差很多：
//!
//! | 階段 | 原始工具是否提供即時視圖 | 本模組的做法 |
//! |---|---|---|
//! | FFmpeg 抽幀 | 只有文字進度（`frame=`、`speed=`） | 解析日誌得到影格數與速度，並列出最新影格縮圖 |
//! | COLMAP 特徵／匹配 | 只有主控台輸出 | 解析 `Features:`、`Processing image [N/M]` 得到進度 |
//! | COLMAP 建圖 | **沒有**即時三維視圖（設計上就沒有） | 解析 `=> Image sees ...` 得到已註冊影格數與點數；若啟用了中間快照，另可顯示點雲統計 |
//! | Brush 訓練 | **有**：Brush 內建即時查看器（`--with-viewer`） | 本模組顯示訓練參數與耗時，並提示「三維畫面請看 Brush 視窗」 |
//!
//! 因此本模組不是要取代任何工具，而是把**散落在日誌裡的進度**集中成一頁，
//! 讓現場觀眾看得到「現在跑到哪裡」。
//!
//! ## 設計
//! 解析器全部是純函式（輸入日誌文字、輸出結構），因此可以用合成日誌做單元測試，
//! 不需要真的跑一次重建就能驗證格式是否對得上。

use std::path::{Path, PathBuf};

use serde::Serialize;

/// FFmpeg 抽幀進度。
#[derive(Debug, Clone, Default, Serialize)]
pub struct FfmpegProgress {
    /// FFmpeg 回報的已完成影格數（`frame=`）。
    pub frames_reported: usize,
    /// FFmpeg 回報的處理速度（`speed=61.8x` → 61.8）。
    pub speed: Option<f64>,
    pub finished: bool,
}

/// COLMAP 進度。
#[derive(Debug, Clone, Default, Serialize)]
pub struct ColmapProgress {
    /// 目前執行到哪個子命令：feature_extractor／sequential_matcher／mapper…
    pub command: Option<String>,
    /// 已抽出的特徵影像數（`Features: N (SIFT)` 的行數）。
    pub feature_images: usize,
    /// 最後一張影像的特徵點數。
    pub last_feature_count: usize,
    /// 匹配進度（`Processing image [N/M]`）。
    pub matching_done: usize,
    pub matching_total: usize,
    /// **當前**重建的已註冊影格數（`Registering image #N (num_reg_frames=M)` 的 M）。
    pub registered_images: usize,
    /// **最佳**重建的已註冊影格數（整份日誌的最大值）。
    ///
    /// 為什麼要區分「當前」與「最佳」：COLMAP 的 mapper 會嘗試多個重建，
    /// 失敗的被丟棄後重試，因此同一份日誌會出現很多次 `Registering image`。
    /// **用行數當註冊數會嚴重高估**（實測某專案行數 811、實際只註冊 15 張）；
    /// 完成後要看「最終模型有多大」應取最大值。
    /// 但請注意：日誌仍只是「過程」的近似——**最終模型檔才是真相**，
    /// 因此 `read_snapshot` 會在模型存在時改用模型內的影像數（見 `registered_source`）。
    pub best_registered: usize,
    /// 已註冊影格數的來源：`"model"`（讀最終模型，精確）或 `"log"`（日誌近似，執行中）。
    pub registered_source: String,
    /// 當前的三維點數（`=> Image sees N / M points` 的 M）。
    pub points: usize,
    /// 最佳重建的三維點數（最大值）。
    pub best_points: usize,
    /// 中間快照數量（`--Mapper.snapshot_path` 產生的）。
    pub snapshots: usize,
}

/// Brush 訓練進度（Brush 不寫進度到日誌，因此只有參數與耗時）。
#[derive(Debug, Clone, Default, Serialize)]
pub struct BrushProgress {
    pub total_steps: Option<usize>,
    pub max_resolution: Option<usize>,
    pub running: bool,
    pub elapsed_ms: Option<u64>,
    /// 最終模型是否已產出。
    pub produced_model: bool,
}

/// 一次監視快照。
#[derive(Debug, Clone, Serialize)]
pub struct MonitorSnapshot {
    pub project_name: String,
    /// 目前階段（人類可讀）。
    pub stage: String,
    pub frames_on_disk: usize,
    /// 最新的影格檔名（用於縮圖牆）。
    pub newest_frames: Vec<String>,
    pub ffmpeg: FfmpegProgress,
    pub colmap: ColmapProgress,
    pub brush: BrushProgress,
    /// 最終 `final.ply` 是否已存在。
    pub final_ply: Option<u64>,
}

fn tail_lines(text: &str, n: usize) -> Vec<&str> {
    let mut v: Vec<&str> = text.lines().filter(|l| !l.trim().is_empty()).collect();
    if v.len() > n {
        v.drain(0..v.len() - n);
    }
    v
}

/// 解析 FFmpeg 的 `-progress` 輸出。
pub fn parse_ffmpeg_progress(text: &str) -> FfmpegProgress {
    let mut p = FfmpegProgress::default();
    for line in text.lines() {
        let line = line.trim();
        if let Some(v) = line.strip_prefix("frame=") {
            if let Ok(n) = v.trim().parse::<usize>() {
                p.frames_reported = n;
            }
        } else if let Some(v) = line.strip_prefix("speed=") {
            let v = v.trim().trim_end_matches('x');
            p.speed = v.parse::<f64>().ok();
        } else if line == "progress=end" {
            p.finished = true;
        }
    }
    p
}

/// 解析 COLMAP 的日誌。
///
/// 認得的行：
///   - `arguments:` 之後的第一行是子命令名稱（feature_extractor／sequential_matcher／mapper）
///   - `Features: 2027 (SIFT)`            每張影像的特徵點數
///   - `Processing image [12/273]`        匹配進度
///   - `=> Image sees 1309 / 2397 points` 每註冊一張影像後的點數
pub fn parse_colmap_progress(text: &str) -> ColmapProgress {
    let mut p = ColmapProgress::default();
    let mut expect_command = false;
    for raw in text.lines() {
        let line = raw.trim();
        if line.starts_with("arguments:") {
            expect_command = true;
            continue;
        }
        if expect_command {
            // 子命令名稱是 arguments: 之後第一個非選項、非空行
            if !line.is_empty() && !line.starts_with('-') && !line.contains(":\\") && !line.contains(":/") {
                p.command = Some(line.to_string());
                // 每次新的子命令都把上一階段的計數歸零，避免把前後階段混在一起
                match line {
                    "feature_extractor" => {
                        p.feature_images = 0;
                        p.last_feature_count = 0;
                    }
                    "sequential_matcher" | "exhaustive_matcher" | "vocab_tree_matcher" => {
                        p.matching_done = 0;
                        p.matching_total = 0;
                    }
                    "mapper" => {
                        p.registered_images = 0;
                        p.points = 0;
                    }                    _ => {}
                }
            }
            expect_command = false;
            continue;
        }
        if line.contains("Features:") && line.contains("(SIFT)") {
            p.feature_images += 1;
            if let Some(rest) = line.split("Features:").nth(1) {
                if let Some(num) = rest.trim().split_whitespace().next() {
                    if let Ok(n) = num.parse::<usize>() {
                        p.last_feature_count = n;
                    }
                }
            }
        } else if let Some(idx) = line.find("Processing image [") {
            if let Some(close) = line[idx..].find(']') {
                let inside = &line[idx + "Processing image [".len()..idx + close];
                let mut parts = inside.split('/');
                if let (Some(a), Some(b)) = (parts.next(), parts.next()) {
                    p.matching_done = a.trim().parse().unwrap_or(p.matching_done);
                    p.matching_total = b.trim().parse().unwrap_or(p.matching_total);
                }
            }
        } else if let Some(idx) = line.find("Registering image #") {
            // 真正的註冊事件：行內附帶 (num_reg_frames=M)，M 才是目前重建的影格數。
            // 不能數行數——mapper 失敗重試會產生大量重複行。
            if let Some(mpos) = line.find("num_reg_frames=") {
                let rest = &line[mpos + "num_reg_frames=".len()..];
                let num: String = rest.chars().take_while(|c| c.is_ascii_digit()).collect();
                if let Ok(n) = num.parse::<usize>() {
                    p.registered_images = n;
                    p.best_registered = p.best_registered.max(n);
                }
            }
        } else if let Some(idx) = line.find("=> Image sees") {
            let rest = &line[idx + "=> Image sees".len()..];
            // 形如 " 1309 / 2397 points"；每張影像在每次 bundle adjustment 都會印一次，
            // 因此只取最大值當作「目前最佳重建的點數」。
            let nums: Vec<usize> = rest
                .split_whitespace()
                .filter_map(|t| t.parse::<usize>().ok())
                .collect();
            if nums.len() >= 2 {
                p.points = nums[1];
                p.best_points = p.best_points.max(nums[1]);
            }
        }
    }
    p
}

/// 解析 Brush 的日誌（只有命令列參數與結束狀態；進度要看 Brush 視窗）。
pub fn parse_brush_progress(text: &str, model_exists: bool) -> BrushProgress {
    let mut p = BrushProgress::default();
    let mut expect_value: Option<String> = None;
    for raw in text.lines() {
        let line = raw.trim();
        if let Some(flag) = expect_value.take() {
            match flag.as_str() {
                "--total-steps" => p.total_steps = line.parse().ok(),
                "--max-resolution" => p.max_resolution = line.parse().ok(),
                _ => {}
            }
            continue;
        }
        match line {
            "--total-steps" => expect_value = Some("--total-steps".into()),
            "--max-resolution" => expect_value = Some("--max-resolution".into()),
            _ => {}
        }
        if let Some(v) = line.strip_prefix("elapsed_ms:") {
            p.elapsed_ms = v.trim().parse().ok();
        }
    }
    p.produced_model = model_exists;
    // 有參數但沒有結束時間 → 視為正在訓練
    p.running = p.total_steps.is_some() && p.elapsed_ms.is_none();
    p
}

/// 讀取 COLMAP 最終模型中的已註冊影像數（`images.bin` 的前 8 位元組就是個數）。
///
/// 這比解析日誌精確得多：日誌反映的是「過程」（含失敗重試），
/// 模型檔反映的是「結果」。而且只讀 8 位元組，成本極低，適合每兩秒刷新一次。
pub fn count_model_images(project_dir: &Path) -> Option<usize> {
    for rel in ["work/colmap/sparse/0/images.bin", "work/colmap/images.bin", "sparse/0/images.bin"] {
        let p = project_dir.join(rel);
        if let Ok(bytes) = std::fs::read(&p) {
            if bytes.len() >= 8 {
                let n = u64::from_le_bytes(bytes[0..8].try_into().ok()?);
                // 合理性檢查：避免讀到非 COLMAP 檔
                if n > 0 && n < 10_000_000 {
                    return Some(n as usize);
                }
            }
        }
    }
    None
}

/// 讀取一個專案的目前狀態。
pub fn read_snapshot(project_dir: impl AsRef<Path>) -> MonitorSnapshot {
    let project_dir = project_dir.as_ref();
    let logs = project_dir.join("logs");
    let read = |name: &str| std::fs::read_to_string(logs.join(name)).unwrap_or_default();

    let frames_dir = ["work/frames", "frames", "source/images"]
        .iter()
        .map(|p| project_dir.join(p))
        .find(|p| p.is_dir());
    let mut frames: Vec<(std::time::SystemTime, String)> = Vec::new();
    if let Some(dir) = &frames_dir {
        if let Ok(entries) = std::fs::read_dir(dir) {
            for e in entries.flatten() {
                if e.path().is_file() {
                    let t = e
                        .metadata()
                        .and_then(|m| m.modified())
                        .unwrap_or(std::time::SystemTime::UNIX_EPOCH);
                    frames.push((t, e.file_name().to_string_lossy().to_string()));
                }
            }
        }
    }
    frames.sort_by(|a, b| b.0.cmp(&a.0));
    let newest_frames: Vec<String> = frames.iter().take(12).map(|(_, n)| n.clone()).collect();

    let ffmpeg = parse_ffmpeg_progress(&read("ffmpeg.log"));
    let mut colmap = parse_colmap_progress(&read("colmap.log"));
    // 模型檔是「結果」，日誌是「過程」：能讀到模型就改用精確值。
    colmap.registered_source = "log".into();
    if let Some(n) = count_model_images(project_dir) {
        colmap.registered_images = n;
        colmap.best_registered = colmap.best_registered.max(n);
        colmap.registered_source = "model".into();
    }
    let model_path = project_dir.join("work/brush/final.ply.tmp");
    let final_ply_path = project_dir.join("final.ply");
    let brush = parse_brush_progress(&read("brush.log"), model_path.is_file());

    // 中間快照數量（若重建時啟用了 --Mapper.snapshot_path）
    let snap_dir = project_dir.join("work/colmap/sparse/snapshots");
    colmap.snapshots = std::fs::read_dir(&snap_dir)
        .map(|d| d.flatten().filter(|e| e.path().is_dir()).count())
        .unwrap_or(0);

    let stage = if final_ply_path.is_file() {
        "完成（final.ply 已產出）".to_string()
    } else if brush.running || brush.total_steps.is_some() && !brush.elapsed_ms.is_some() {
        "Brush 訓練中（三維畫面請看 Brush 視窗）".to_string()
    } else if colmap.command.as_deref() == Some("mapper") && colmap.registered_images > 0 {
        format!(
            "COLMAP 建圖中（已註冊 {} 張{}、{} 個三維點）",
            colmap.registered_images,
            if colmap.registered_source == "model" { "" } else { "（日誌估算）" },
            colmap.points
        )
    } else if colmap.command.is_some() {
        format!("COLMAP {} 中", colmap.command.clone().unwrap_or_default())
    } else if ffmpeg.frames_reported > 0 || !frames.is_empty() {
        format!("FFmpeg 抽幀中（已抽出 {} 張）", frames.len())
    } else {
        "尚未開始".to_string()
    };

    MonitorSnapshot {
        project_name: project_dir
            .file_name()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_default(),
        stage,
        frames_on_disk: frames.len(),
        newest_frames,
        ffmpeg,
        colmap,
        brush,
        final_ply: std::fs::metadata(&final_ply_path).ok().map(|m| m.len()),
    }
}

/// 產生可自動刷新的展示頁（不依賴任何前端框架或伺服器）。
///
/// `frames_href` 為影格目錄相對於 HTML 檔案的連結（例如 `frames`）。
pub fn render_html(snap: &MonitorSnapshot, frames_href: &str, refresh_secs: u64, title: &str) -> String {
    let pct = |done: usize, total: usize| -> f64 {
        if total == 0 {
            0.0
        } else {
            (done as f64 / total as f64 * 100.0).min(100.0)
        }
    };
    let bar = |label: &str, done: usize, total: usize, unit: &str| -> String {
        let p = pct(done, total);
        format!(
            "<div class=\"row\"><div class=\"lab\">{label}</div>\
             <div class=\"track\"><div class=\"fill\" style=\"width:{p:.1}%\"></div></div>\
             <div class=\"num\">{done} / {total} {unit}</div></div>"
        )
    };
    let mut thumbs = String::new();
    for f in &snap.newest_frames {
        thumbs.push_str(&format!(
            "<figure><img loading=\"lazy\" src=\"{frames_href}/{f}\" alt=\"{f}\"><figcaption>{f}</figcaption></figure>"
        ));
    }
    let colmap_stage = snap.colmap.command.clone().unwrap_or_else(|| "（尚未開始）".into());
    let ffmpeg_speed = snap
        .ffmpeg
        .speed
        .map(|s| format!("{s:.1}x"))
        .unwrap_or_else(|| "—".into());
    let brush_steps = snap
        .brush
        .total_steps
        .map(|s| s.to_string())
        .unwrap_or_else(|| "—".into());
    let brush_res = snap
        .brush
        .max_resolution
        .map(|s| format!("{s}px"))
        .unwrap_or_else(|| "—".into());
    let brush_state = if snap.brush.produced_model {
        "已產出模型"
    } else if snap.brush.running {
        "訓練中"
    } else if snap.brush.elapsed_ms.is_some() {
        "已結束"
    } else {
        "尚未開始"
    };
    let final_ply = snap
        .final_ply
        .map(|b| format!("{:.1} MB", b as f64 / 1048576.0))
        .unwrap_or_else(|| "—".into());
    // 階段顯示以 final_ply 的存在為準（單一真相來源）：只要最終模型已產出，
    // 不論 snapshot 是何時建立的，畫面都必須顯示「完成」，避免顯示與資料不一致。
    let stage_display = match snap.final_ply {
        Some(bytes) => format!("完成（final.ply 已產出，{:.1} MB）", bytes as f64 / 1048576.0),
        None => snap.stage.clone(),
    };

    format!(
        r#"<!doctype html><html lang="zh-Hant"><head><meta charset="utf-8">
<meta http-equiv="refresh" content="{refresh}">
<title>{title}</title><style>
 body{{margin:0;background:#0b1622;color:#e8eef5;font-family:'Microsoft JhengHei',system-ui,sans-serif}}
 header{{padding:14px 22px;background:#10243a;border-bottom:3px solid #c8a24a}}
 h1{{margin:0;font-size:20px}} .stage{{margin-top:6px;font-size:15px;color:#c8a24a;font-weight:700}}
 .grid{{display:grid;grid-template-columns:1.15fr 1fr;gap:14px;padding:14px}}
 .card{{background:#16202c;border:1px solid #26374a;border-radius:10px;padding:14px}}
 .card h2{{margin:0 0 10px;font-size:15px;color:#9fb3c8;font-weight:600;letter-spacing:.04em}}
 .row{{display:flex;align-items:center;gap:10px;margin:8px 0}}
 .lab{{width:8.5rem;font-size:13px;color:#b9c9da}} .num{{width:9rem;font-size:13px;color:#8fa3b8;text-align:right}}
 .track{{flex:1;height:12px;background:#0d1620;border-radius:6px;overflow:hidden;border:1px solid #22303f}}
 .fill{{height:100%;background:linear-gradient(90deg,#2a7f8f,#c8a24a)}}
 .kpi{{display:flex;flex-wrap:wrap;gap:10px;margin-top:6px}}
 .kpi div{{background:#0f1a26;border:1px solid #22303f;border-radius:8px;padding:8px 12px;font-size:13px}}
 .kpi b{{color:#c8a24a;font-size:16px;display:block}}
 .wall{{display:grid;grid-template-columns:repeat(4,1fr);gap:8px}}
 figure{{margin:0}} .wall img{{width:100%;border-radius:6px;background:#000;display:block}}
 figcaption{{font-size:11px;color:#7f93a8;margin-top:3px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}}
 .note{{padding:0 22px 18px;color:#8fa3b8;font-size:12px;line-height:1.7}}
 code{{background:#16324d;padding:1px 5px;border-radius:4px}}
</style></head><body>
<header><h1>{title}</h1><div class="stage">▶ {stage_display}</div></header>
<div class="grid">
  <div class="card"><h2>階段進度</h2>
    {ff_bar}
    {match_bar}
    {map_bar}
    <div class="kpi">
      <div>抽幀影格<b>{frames}</b></div>
      <div>抽幀速度<b>{ff_speed}</b></div>
      <div>已註冊影格<b>{registered}</b><span style="color:#7f93a8">（{reg_src}）</span></div>
      <div>三維點<b>{points}</b></div>
      <div>中間快照<b>{snapshots}</b></div>
      <div>最終模型<b>{final_ply}</b></div>
    </div>
  </div>
  <div class="card"><h2>COLMAP / Brush 狀態</h2>
    <div class="row"><div class="lab">COLMAP 子命令</div><div class="num" style="width:auto">{colmap_stage}</div></div>
    <div class="row"><div class="lab">特徵抽取</div><div class="num" style="width:auto">{feature_images} 張（最後一張 {last_feat} 點）</div></div>
    <div class="row"><div class="lab">Brush 狀態</div><div class="num" style="width:auto">{brush_state}</div></div>
    <div class="row"><div class="lab">訓練步數設定</div><div class="num" style="width:auto">{brush_steps} 步 / 最大 {brush_res}</div></div>
    <div class="note" style="padding:10px 0 0">Brush 的三維即時畫面在它自己的視窗（<code>--with-viewer</code>）；
    本頁負責把它看不到的 FFmpeg 抽幀與 COLMAP 建圖進度集中呈現。</div>
  </div>
</div>
<div class="card" style="margin:0 14px 14px"><h2>最新抽出的影格</h2><div class="wall">{thumbs}</div></div>
<div class="note">此頁每 {refresh} 秒自動重新整理（由 <code>splatstudio monitor --watch</code> 重新產生）。
專案：{project}</div>
</body></html>"#,
        refresh = refresh_secs.max(1),
        title = title,
        stage_display = stage_display,
        ff_bar = bar("FFmpeg 抽幀", snap.ffmpeg.frames_reported.max(snap.frames_on_disk), snap.frames_on_disk.max(snap.ffmpeg.frames_reported), "張"),
        match_bar = bar("COLMAP 匹配", snap.colmap.matching_done, snap.colmap.matching_total, "張"),
        map_bar = bar("COLMAP 建圖（最佳重建）", snap.colmap.best_registered, snap.frames_on_disk.max(snap.colmap.best_registered), "張"),
        frames = snap.frames_on_disk,
        ff_speed = ffmpeg_speed,
        registered = snap.colmap.registered_images,
        reg_src = if snap.colmap.registered_source == "model" { "讀自模型" } else { "日誌估算" },
        points = snap.colmap.points,
        snapshots = snap.colmap.snapshots,
        final_ply = final_ply,
        colmap_stage = colmap_stage,
        feature_images = snap.colmap.feature_images,
        last_feat = snap.colmap.last_feature_count,
        brush_state = brush_state,
        brush_steps = brush_steps,
        brush_res = brush_res,
        thumbs = thumbs,
        project = snap.project_name,
    )
}

/// 監視輸出檔的預設路徑。
pub fn default_output(project_dir: impl AsRef<Path>) -> PathBuf {
    project_dir.as_ref().join("monitor.html")
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 以真實日誌的格式（取自實際專案）驗證 FFmpeg 進度解析。
    #[test]
    fn parses_ffmpeg_progress() {
        let text = "frame=120\nfps=9.0\nspeed=61.8x\nprogress=continue\nframe=273\nspeed=64.2x\nprogress=end\n";
        let p = parse_ffmpeg_progress(text);
        assert_eq!(p.frames_reported, 273);
        assert_eq!(p.speed, Some(64.2));
        assert!(p.finished);
    }

    /// 以真實日誌格式驗證 COLMAP 解析（含三個子命令的分段歸零）。
    #[test]
    fn parses_colmap_progress_across_stages() {
        let text = "\
executable: colmap.exe
arguments:
  feature_extractor
  --database_path
  db.db
Features:        2027 (SIFT)
Features:        2081 (SIFT)
exit_code: Some(0)
executable: colmap.exe
arguments:
  sequential_matcher
Processing image [1/273]
Processing image [2/273]
exit_code: Some(0)
executable: colmap.exe
arguments:
  mapper
Registering image #5 (num_reg_frames=1)
=> Image sees 1309 / 2397 points
Registering image #9 (num_reg_frames=2)
=> Image sees 1352 / 2450 points
Keeping successful reconstruction
Elapsed time: 2.710 [minutes]
";
        let p = parse_colmap_progress(text);
        assert_eq!(p.command.as_deref(), Some("mapper"), "應認得最後的階段是 mapper");
        assert_eq!(p.registered_images, 2, "應取最後的 num_reg_frames：{p:?}");
        assert_eq!(p.best_registered, 2, "{p:?}");
        assert_eq!(p.points, 2450, "應取最後一次的點數：{p:?}");
        assert_eq!(p.matching_total, 273, "{p:?}");
    }

    /// **關鍵回歸測試**：mapper 失敗重試會產生大量 `Registering image` 行，
    /// 但真正的註冊數是行內 `num_reg_frames` 的值。
    ///
    /// 實測案例：某專案日誌有 811 行 `Registering image`，實際只註冊 15 張。
    /// 若用行數當註冊數，畫面會顯示 811（超過影格總數），完全誤導觀眾。
    #[test]
    fn registration_count_uses_num_reg_frames_not_line_count() {
        let mut text = String::from("arguments:\n  mapper\n");
        // 模擬三次失敗的重建（各註冊 3、5 張）＋ 最後成功的一次（15 張）
        for target in [3usize, 5, 15] {
            for _ in 0..40 {
                for i in 1..=target {
                    text.push_str(&format!(
                        "I20260827 18:02:04.275585 31176 incremental_pipeline.cc:524] \
                         Registering image #{i} (num_reg_frames={i})\n"
                    ));
                }
            }
        }
        let p = parse_colmap_progress(&text);
        let line_count = text.matches("Registering image").count();
        assert!(line_count > 500, "測試資料應包含大量重複行（實際 {line_count} 行）");
        assert_eq!(p.registered_images, 15, "當前應為最後一次重建的 15 張：{p:?}");
        assert_eq!(p.best_registered, 15, "最佳應為最大的一次（15 張）：{p:?}");
        assert!(
            p.best_registered < line_count / 10,
            "註冊數（{}）必須遠小於行數（{line_count}），否則就是拿行數當註冊數了",
            p.best_registered
        );
    }

    /// 特徵抽取階段：應以進入該階段後重新計數。
    #[test]
    fn counts_features_only_within_its_stage() {
        let text = "\
arguments:
  feature_extractor
Features:        2027 (SIFT)
Features:        2081 (SIFT)
Features:        2113 (SIFT)
";
        let p = parse_colmap_progress(text);
        assert_eq!(p.command.as_deref(), Some("feature_extractor"));
        assert_eq!(p.feature_images, 3);
        assert_eq!(p.last_feature_count, 2113);
    }

    /// Brush 日誌只有參數與結束時間：正在訓練時 elapsed_ms 應為 None。
    #[test]
    fn parses_brush_progress_running_and_finished() {
        let running = "\
executable: brush_app.exe
arguments:
  --total-steps
  8000
  --max-resolution
  1200
";
        let p = parse_brush_progress(running, false);
        assert_eq!(p.total_steps, Some(8000));
        assert_eq!(p.max_resolution, Some(1200));
        assert!(p.running, "有參數但沒有結束時間應視為訓練中");
        assert!(!p.produced_model);

        let finished = format!("{running}exit_code: Some(0)\nelapsed_ms: 144583\n");
        let p = parse_brush_progress(&finished, true);
        assert_eq!(p.elapsed_ms, Some(144583));
        assert!(!p.running);
        assert!(p.produced_model);
    }

    /// 空日誌不應 panic，且狀態為「尚未開始」。
    #[test]
    fn empty_project_reports_not_started() {
        let dir = std::env::temp_dir().join(format!("macausplat-monitor-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let snap = read_snapshot(&dir);
        assert_eq!(snap.frames_on_disk, 0);
        assert_eq!(snap.stage, "尚未開始");
        assert!(snap.final_ply.is_none());
        let html = render_html(&snap, "frames", 2, "測試");
        assert!(html.contains("尚未開始"));
        assert!(html.contains("http-equiv=\"refresh\" content=\"2\""));
        let _ = std::fs::remove_dir_all(dir);
    }

    /// HTML 應包含三條進度條與最新影格牆，且重新整理秒數可設定。
    #[test]
    fn renders_expected_html_sections() {
        let mut snap = MonitorSnapshot {
            project_name: "demo".into(),
            stage: "COLMAP 建圖中（已註冊 2 張、2450 個三維點）".into(),
            frames_on_disk: 273,
            newest_frames: vec!["frame_0001.jpg".into(), "frame_0002.jpg".into()],
            ffmpeg: FfmpegProgress { frames_reported: 273, speed: Some(61.8), finished: true },
            colmap: ColmapProgress {
                command: Some("mapper".into()),
                feature_images: 273,
                last_feature_count: 2027,
                matching_done: 273,
                matching_total: 273,
                registered_images: 2,
                best_registered: 2,
                registered_source: "model".into(),
                points: 2450,
                best_points: 2450,
                snapshots: 3,
            },
            brush: BrushProgress { total_steps: Some(8000), max_resolution: Some(1200), running: false, elapsed_ms: None, produced_model: false },
            final_ply: None,
        };
        let html = render_html(&snap, "../work/frames", 3, "MacauSplat 即時監視");
        assert!(html.contains("MacauSplat 即時監視"));
        assert!(html.contains("content=\"3\""));
        assert!(html.contains("FFmpeg 抽幀"));
        assert!(html.contains("COLMAP 匹配"));
        assert!(html.contains("COLMAP 建圖"));
        assert!(html.contains("frame_0001.jpg"));
        assert!(html.contains("../work/frames/frame_0001.jpg"));
        assert!(html.contains("2450"), "應顯示三維點數");
        // 完成後階段文字應改變
        snap.final_ply = Some(1024 * 1024);
        let done = render_html(&snap, "frames", 2, "t");
        assert!(done.contains("完成"), "有 final.ply 時應顯示完成");
    }
}
