import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Crosshair, LoaderCircle, RotateCcw, X } from "lucide-react";
import { getSnapshotCloud } from "../../lib/backend";
import { useI18n } from "../../i18n";
import type { SnapshotCloud } from "../../types/pipeline";
import {
  boundsOf,
  cameraPosition,
  clampElevation,
  fitCamera,
  projectPoints,
  unionBounds,
  type OrbitCamera,
} from "./projection";

/** 輪詢間隔（毫秒）：建圖階段每註冊一張圖就換一份快照，1.5 秒足以看到增長。 */
const POLL_INTERVAL_MS = 1500;
/** 送到檢視器的點數上限（降低前端負擔，同時保留整體分布）。 */
const MAX_POINTS = 60_000;

type Props = {
  projectPath: string;
  /** 專案是否仍在執行（執行中才輪詢快照）。 */
  live: boolean;
  onExit: () => void;
};

/**
 * 建圖過程的點雲檢視器：把 COLMAP 的中間快照畫成三維點雲，
 * 並疊上已註冊影像的相機中心，讓「點雲增長 × 拍攝軌跡」同時可見。
 *
 * 這是既有工具都沒有的一塊：COLMAP 執行時不提供任何三維預覽，
 * Brush 的即時檢視器要等訓練階段才出現；本元件補上「特徵→匹配→建圖」這段空窗。
 */
export function PointCloudViewer({ projectPath, live, onExit }: Props) {
  const { t } = useI18n();
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const cameraRef = useRef<OrbitCamera | null>(null);
  const cloudRef = useRef<SnapshotCloud | null>(null);
  const dragRef = useRef<{ x: number; y: number } | null>(null);
  const [cloud, setCloud] = useState<SnapshotCloud | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [autoFit, setAutoFit] = useState(true);

  // ---------------------------------------------------------------- 取得點雲
  const refresh = useCallback(async () => {
    try {
      const next = await getSnapshotCloud(projectPath, MAX_POINTS);
      setCloud((prev) => {
        // 快照沒變就不重繪，避免執行中每 1.5 秒無謂重畫
        const same =
          prev &&
          prev.snapshot === next.snapshot &&
          prev.point_count === next.point_count &&
          prev.cameras.length === next.cameras.length;
        return same ? prev : next;
      });
      cloudRef.current = next;
      setError(null);
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : String(cause));
    } finally {
      setLoading(false);
    }
  }, [projectPath]);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  useEffect(() => {
    if (!live) return;
    const timer = window.setInterval(() => void refresh(), POLL_INTERVAL_MS);
    return () => window.clearInterval(timer);
  }, [live, refresh]);

  // ---------------------------------------------------------------- 自動取景
  const sceneBounds = useMemo(() => {
    if (!cloud || cloud.point_count === 0) return null;
    const cloudBounds = {
      min: [cloud.bounds[0], cloud.bounds[1], cloud.bounds[2]] as [number, number, number],
      max: [cloud.bounds[3], cloud.bounds[4], cloud.bounds[5]] as [number, number, number],
    };
    // 相機軌跡也納入取景，否則拍得比較遠的相機會落在畫面外
    const cameraBounds = boundsOf(Float32Array.from(cloud.cameras));
    return unionBounds(cloudBounds, cameraBounds);
  }, [cloud]);

  useEffect(() => {
    if (autoFit && sceneBounds) {
      cameraRef.current = fitCamera(sceneBounds);
      setAutoFit(false);
    }
  }, [autoFit, sceneBounds]);

  // 點雲第一次到達時先給一個預設相機
  useEffect(() => {
    if (!cameraRef.current && sceneBounds) cameraRef.current = fitCamera(sceneBounds);
  }, [sceneBounds]);

  // ---------------------------------------------------------------- 繪製
  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    const data = cloudRef.current;
    const camera = cameraRef.current;
    if (!canvas || !camera) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const { width, height } = canvas;
    ctx.fillStyle = "#0b0f14";
    ctx.fillRect(0, 0, width, height);
    if (!data || data.point_count === 0) return;

    const positions = Float32Array.from(data.positions);
    const { screen, depth } = projectPoints(positions, camera, width, height);

    // 用 ImageData 直接寫像素：6 萬點用 fillRect 會很慢，寫像素則在毫秒級
    const image = ctx.getImageData(0, 0, width, height);
    const px = image.data;
    const colors = data.colors;
    const maxDepth = camera.distance * 2;
    for (let i = 0; i < data.point_count; i += 1) {
      const z = depth[i];
      if (z < 0) continue;
      const sx = screen[i * 2] | 0;
      const sy = screen[i * 2 + 1] | 0;
      if (sx < 0 || sx >= width || sy < 0 || sy >= height) continue;
      // 深度越大越暗（模擬霧化），讓遠近有層次
      const fade = Math.max(0.25, Math.min(1, 1 - z / maxDepth));
      const r = (colors[i * 3] * fade) | 0;
      const g = (colors[i * 3 + 1] * fade) | 0;
      const b = (colors[i * 3 + 2] * fade) | 0;
      const offset = (sy * width + sx) * 4;
      px[offset] = r;
      px[offset + 1] = g;
      px[offset + 2] = b;
      px[offset + 3] = 255;
    }
    ctx.putImageData(image, 0, 0);

    // 相機中心：青色方點，畫在點雲之上
    if (data.cameras.length >= 3) {
      const camPositions = Float32Array.from(data.cameras);
      const { screen: camScreen, depth: camDepth } = projectPoints(camPositions, camera, width, height);
      ctx.fillStyle = "#38e8ff";
      for (let i = 0; i < camPositions.length / 3; i += 1) {
        if (camDepth[i] < 0) continue;
        const sx = camScreen[i * 2];
        const sy = camScreen[i * 2 + 1];
        ctx.fillRect(sx - 1.5, sy - 1.5, 3, 3);
      }
    }
  }, []);

  useEffect(() => {
    draw();
  }, [cloud, draw]);

  // 視窗大小改變時重設解析度並重繪
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const resize = () => {
      const rect = canvas.getBoundingClientRect();
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      canvas.width = Math.max(1, Math.floor(rect.width * dpr));
      canvas.height = Math.max(1, Math.floor(rect.height * dpr));
      draw();
    };
    resize();
    window.addEventListener("resize", resize);
    return () => window.removeEventListener("resize", resize);
  }, [draw]);

  // ---------------------------------------------------------------- 互動
  const onPointerDown = (event: React.PointerEvent<HTMLCanvasElement>) => {
    dragRef.current = { x: event.clientX, y: event.clientY };
    event.currentTarget.setPointerCapture(event.pointerId);
  };
  const onPointerMove = (event: React.PointerEvent<HTMLCanvasElement>) => {
    const drag = dragRef.current;
    const camera = cameraRef.current;
    if (!drag || !camera) return;
    const dx = event.clientX - drag.x;
    const dy = event.clientY - drag.y;
    dragRef.current = { x: event.clientX, y: event.clientY };
    // 依 drag 距離給一個合理的角度增量；仰角夾住避免翻轉
    const gain = camera.fov / 400;
    camera.azimuth -= dx * gain;
    camera.elevation = clampElevation(camera.elevation + dy * gain);
    draw();
  };
  const onPointerUp = () => {
    dragRef.current = null;
  };
  const onWheel = (event: React.WheelEvent<HTMLCanvasElement>) => {
    const camera = cameraRef.current;
    if (!camera) return;
    camera.distance = Math.max(0.05, camera.distance * (event.deltaY > 0 ? 1.12 : 1 / 1.12));
    draw();
  };

  const resetView = () => {
    if (sceneBounds) cameraRef.current = fitCamera(sceneBounds);
    draw();
  };

  const camera = cameraRef.current;

  return (
    <section className="pointcloud-viewer">
      <header className="pointcloud-topbar">
        <div className="pointcloud-title">
          <Crosshair size={16} />
          <strong>{t("pointcloud.title")}</strong>
          {live && <span className="pointcloud-live"><span className="status-light" />{t("pointcloud.live")}</span>}
        </div>
        <div className="pointcloud-actions">
          <button type="button" className="settings-action" onClick={resetView}><RotateCcw size={15} />{t("pointcloud.resetView")}</button>
          <button type="button" className="settings-action" onClick={onExit}><X size={15} />{t("pointcloud.exit")}</button>
        </div>
      </header>

      <div className="pointcloud-stage">
        <canvas
          ref={canvasRef}
          onPointerDown={onPointerDown}
          onPointerMove={onPointerMove}
          onPointerUp={onPointerUp}
          onPointerCancel={onPointerUp}
          onWheel={onWheel}
        />
        {loading && (
          <div className="pointcloud-overlay"><LoaderCircle className="spin" size={22} /><span>{t("pointcloud.loading")}</span></div>
        )}
        {!loading && cloud && cloud.point_count === 0 && (
          <div className="pointcloud-overlay"><span>{t("pointcloud.empty")}</span></div>
        )}
        {error && <div className="pointcloud-overlay error"><span>{error}</span></div>}
      </div>

      <footer className="pointcloud-stats">
        <div><dt>{t("pointcloud.points")}</dt><dd>{cloud ? cloud.point_count.toLocaleString() : "—"}</dd></div>
        <div><dt>{t("pointcloud.totalPoints")}</dt><dd>{cloud ? cloud.total_points.toLocaleString() : "—"}</dd></div>
        <div><dt>{t("pointcloud.cameras")}</dt><dd>{cloud ? (cloud.cameras.length / 3).toLocaleString() : "—"}</dd></div>
        <div><dt>{t("pointcloud.snapshot")}</dt><dd>{cloud?.snapshot ?? "—"}{cloud?.is_final ? `（${t("pointcloud.finalModel")}）` : ""}</dd></div>
        <div><dt>{t("pointcloud.snapshots")}</dt><dd>{cloud ? cloud.snapshot_count : "—"}</dd></div>
        <div><dt>{t("pointcloud.distance")}</dt><dd>{camera ? camera.distance.toFixed(2) : "—"}</dd></div>
      </footer>
    </section>
  );
}
