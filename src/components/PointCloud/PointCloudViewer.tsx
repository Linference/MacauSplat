import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Crosshair, LoaderCircle, RotateCcw, X } from "lucide-react";
import { getSnapshotCloud } from "../../lib/backend";
import { localizePipelineMessage, useI18n } from "../../i18n";
import type { SnapshotCloud } from "../../types/pipeline";
import {
  boundsOf,
  clampElevation,
  elementBounds,
  fitCamera,
  projectPoints,
  unionBounds,
  visibleIndex,
  type OrbitCamera,
} from "./projection";

/** 輪詢間隔（毫秒）：建圖階段每註冊一張圖就換一份快照，1.5 秒足以看到增長。 */
const POLL_INTERVAL_MS = 1500;
/** 連續失敗後的退避間隔；用完就停止輪詢，改由使用者按「重試」恢復。 */
const BACKOFF_INTERVAL_MS = [5000, 15_000];
/** 送到檢視器的點數上限（降低前端負擔，同時保留整體分布）。 */
const MAX_POINTS = 60_000;
/** 閒置時允許的最高裝置像素比；點雲不需要視網膜解析度。 */
const MAX_IDLE_DPR = 2;
/** 拖曳／縮放期間固定用 1，放開後再以高解析度重畫一次。 */
const INTERACTIVE_DPR = 1;
/** 畫布底色（與 .pointcloud-viewer 一致）。 */
const BACKGROUND = "#0b0f14";

const POINT_FADE = 0.35;
const MIN_BRIGHTNESS = 0.3;
const TRAJECTORY_COLOR = "#38e8ff";
const TRAJECTORY_HEAD_COLOR = "#eafcff";
/** 相機標記的半寬，決定髒矩形要外擴多少。 */
const CAMERA_MARKER_MARGIN = 6;
/** 游標樣式的兩個狀態（避免每次 render 重建物件）。 */
const CURSOR_GRABBING = { cursor: "grabbing" } as const;
const CURSOR_GRAB = { cursor: "grab" } as const;
/** 快照形狀不正確時的錯誤碼（由 i18n 轉成可讀訊息）。 */
const INVALID_SHAPE = "pointcloud.invalidShape";

type CachedScene = {
  /** 快取鍵：refresh() 既有的去重鍵（snapshot + 點數 + 相機數）。 */
  key: string;
  positions: Float32Array;
  colors: Float32Array;
  cameras: Float32Array;
  count: number;
  bounds: ReturnType<typeof unionBounds>;
};

type Metrics = {
  points: number;
  cameras: number;
  totalPoints: number;
  snapshot: string | null;
  isFinal: boolean;
};

type DirtyBox = { x0: number; y0: number; x1: number; y1: number };

type Props = {
  projectPath: string;
  /** 專案是否仍在執行（執行中才輪詢快照）。 */
  live: boolean;
  onExit: () => void;
};

const sceneKeyOf = (cloud: SnapshotCloud) =>
  `${cloud.snapshot ?? ""}|${cloud.point_count}|${cloud.cameras.length}`;

/**
 * 檢查後端回傳的快照形狀是否自洽。
 *
 * 為什麼一定要做：`positions` / `colors` / `cameras` 是裸陣列，一旦長度與
 * `point_count` 不一致，繪製迴圈就會建出 NaN 座標或直接拋錯；
 * 而這個元件在 App 裡是 lazy 邊界，拋錯會讓整個介面空白。
 * 形狀不合格時當成一次失敗處理，並保留上一份好資料。
 */
function shapeErrorOf(cloud: SnapshotCloud): string | null {
  const { point_count: count, positions, colors, cameras, bounds } = cloud;
  if (!Number.isInteger(count) || count < 0 || count > MAX_POINTS) return INVALID_SHAPE;
  if (!Array.isArray(positions) || positions.length !== count * 3) return INVALID_SHAPE;
  if (!Array.isArray(colors) || colors.length < count * 3) return INVALID_SHAPE;
  if (!Array.isArray(cameras) || cameras.length % 3 !== 0) return INVALID_SHAPE;
  if (!Array.isArray(bounds) || bounds.length !== 6) return INVALID_SHAPE;
  return null;
}

/**
 * 建圖過程的點雲檢視器：把 COLMAP 的中間快照畫成三維點雲，
 * 並疊上已註冊影像的相機中心與拍攝軌跡，讓「點雲增長 × 拍攝順序」同時可見。
 *
 * 這是既有工具都沒有的一塊：COLMAP 執行時不提供任何三維預覽，
 * Brush 的即時檢視器要等訓練階段才出現；本元件補上「特徵→匹配→建圖」這段空窗。
 *
 * 效能取捨：六萬個點用 `fillRect` 太慢，所以直接寫像素。`ImageData` 只在
 * 畫布尺寸變動時重建，且每幀只把「真的有畫到」的髒矩形回寫回畫布——
 * 4K 視窗下整張畫布的 `getImageData` + `putImageData` 每幀要搬上百 MB。
 */
export function PointCloudViewer({ projectPath, live, onExit }: Props) {
  const { locale, t, formatNumber } = useI18n();
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const cameraRef = useRef<OrbitCamera | null>(null);
  const cloudRef = useRef<SnapshotCloud | null>(null);
  const sceneRef = useRef<CachedScene | null>(null);
  const dragRef = useRef<{ x: number; y: number } | null>(null);
  const frameRef = useRef<number | null>(null);
  const wheelTimerRef = useRef<number | null>(null);
  const fullRedrawRef = useRef(true);
  const sizeRef = useRef({ width: 0, height: 0 });
  const imageRef = useRef<ImageData | null>(null);
  const baseDprRef = useRef(INTERACTIVE_DPR);
  const lastSizeRef = useRef({ width: 0, height: 0 });
  const dprValueRef = useRef(INTERACTIVE_DPR);
  const generationRef = useRef(0);
  const inFlightRef = useRef(false);
  const failuresRef = useRef(0);
  /** 目前這輪 refresh 的失敗次數；成功才算進 failuresRef，避免卸載後才結算。 */
  const attemptRef = useRef(0);
  /** 統計列的數字放在 ref 並由 rAF 帶動版本號，避免重畫路徑上的 setState。 */
  const statsRef = useRef<Metrics | null>(null);

  const [cloud, setCloud] = useState<SnapshotCloud | null>(null);
  const [failure, setFailure] = useState<string | null>(null);
  /**
   * 是否還有自動重試。
   * live 專案會持續輪詢，失敗時退避重試；非 live（或退避用完）就停下來，
   * 改由使用者按「重試」——那時錯誤要顯示出來，而不是一直寫「正在重試」。
   */
  const [polling, setPolling] = useState(live);
  const [loading, setLoading] = useState(true);
  const [autoFit, setAutoFit] = useState(true);
  const [dpr, setDpr] = useState(INTERACTIVE_DPR);
  const [dragging, setDragging] = useState(false);
  /**
   * 相機放在 state 而不是 ref：繪製與互動都經過 `commitCamera`，
   * 讓重畫被 rAF 節流，而「視距」這類要顯示在統計列的欄位
   * 不必依賴 ref 改動後的額外一次 render。
   */
  const [camera, setCamera] = useState<OrbitCamera | null>(null);

  /**
   * 換專案或卸載時讓還在飛的請求作廢。
   *
   * 用「世代」而不是單純的 mounted 旗標：`refresh()` 是 async，`await` 之後
   * 才寫 state；慢速磁碟會讓舊專案的回應晚於新專案到達，previous 專案的回應
   * 不只要能分辨「已卸載」，還要能分辨「換專案了」。
   * 同時放掉 in-flight 旗標，否則新專案第一次抓取會被舊請求的旗標擋掉。
   */
  useEffect(() => () => {
    generationRef.current += 1;
    inFlightRef.current = false;
  }, [projectPath]);

  /**
   * 相機放在 state 而不是只放 ref：繪製與互動都經過這裡，
   * 重畫仍由 rAF 節流，但「視距」這類要顯示在統計列的欄位
   * 不會因為 ref 改動時序而落後一帧。
   */
  const commitCamera = useCallback((next: OrbitCamera) => {
    cameraRef.current = next;
    setCamera(next);
  }, []);

  const messageOf = useCallback(
    (value: unknown) => (value instanceof Error ? value.message : String(value)),
    [],
  );
  /** 形狀錯誤走 i18n；後端回傳的原始字串只做等效的訊息處理（不可當成 key）。 */
  const renderedFailure = failure == null
    ? null
    : failure === INVALID_SHAPE
      ? t(INVALID_SHAPE)
      : localizePipelineMessage(locale, failure);
  const stats = statsRef.current;

  // ---------------------------------------------------------------- 繪製
  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    const ctx = canvas?.getContext("2d");
    if (!canvas || !ctx) return;

    const width = sizeRef.current.width || canvas.width;
    const height = sizeRef.current.height || canvas.height;
    if (width <= 0 || height <= 0) return;

    // 畫布尺寸改變時內容會被清空，且快取的 ImageData 也必須重建
    if (canvas.width !== width || canvas.height !== height) {
      canvas.width = width;
      canvas.height = height;
      imageRef.current = null;
      fullRedrawRef.current = true;
    }

    const camera = cameraRef.current;
    const data = cloudRef.current;
    const scene = sceneRef.current;

    // 還沒有可畫的場景（空點雲或尚未取景）時只清一次底色：
    // 不必為幾百萬像素的空畫布配置 ImageData。
    if (!camera || !data || !scene || scene.count <= 0) {
      if (fullRedrawRef.current) {
        fullRedrawRef.current = false;
        ctx.setTransform(1, 0, 0, 1, 0, 0);
        ctx.fillStyle = BACKGROUND;
        ctx.fillRect(0, 0, width, height);
      }
      return;
    }

    let image = imageRef.current;
    if (!image || image.width !== width || image.height !== height) {
      image = ctx.createImageData(width, height);
      imageRef.current = image;
    }

    const full = fullRedrawRef.current;
    fullRedrawRef.current = false;
    let dirty: DirtyBox | null = null;
    if (full) {
      ctx.setTransform(1, 0, 0, 1, 0, 0);
      ctx.fillStyle = BACKGROUND;
      ctx.fillRect(0, 0, width, height);
      dirty = { x0: 0, y0: 0, x1: width - 1, y1: height - 1 };
    }

    const projected = projectPoints(scene.positions, camera, width, height);
    const limit = Math.min(scene.count, MAX_POINTS, projected.depth.length);
    const box = elementBounds(projected.screen, projected.depth, width, height, 1);
    if (box) {
      const px = image.data;
      const colors = scene.colors;
      const maxDepth = camera.distance * 2;
      for (let i = 0; i < limit; i += 1) {
        const z = projected.depth[i];
        if (z < 0) continue;
        const sx = projected.screen[i * 2] | 0;
        const sy = projected.screen[i * 2 + 1] | 0;
        if (sx < 0 || sx >= width || sy < 0 || sy >= height) continue;
        // 深度越大越暗（模擬霧化），讓遠近有層次
        const fade = Math.max(MIN_BRIGHTNESS, Math.min(1, 1 - (z / maxDepth) * POINT_FADE));
        const offset = (sy * width + sx) * 4;
        px[offset] = colors[i * 3] * fade;
        px[offset + 1] = colors[i * 3 + 1] * fade;
        px[offset + 2] = colors[i * 3 + 2] * fade;
        px[offset + 3] = 255;
      }
      dirty = dirty ? unionBox(dirty, box) : box;
    }

    if (dirty) ctx.putImageData(image, 0, 0, dirty.x0, dirty.y0, dirty.x1 - dirty.x0 + 1, dirty.y1 - dirty.y0 + 1);
    const track = drawCameraTrack(ctx, camera, width, height, scene.cameras);
    // 軌跡是直接畫在畫布上的（不進 ImageData），所以它佔的區域也要一起回寫，
    // 否則上一幀的軌跡會殘留成鬼影。
    if (track) {
      const trackBox = intersectBox({
        x0: Math.floor(track.x0 - CAMERA_MARKER_MARGIN),
        y0: Math.floor(track.y0 - CAMERA_MARKER_MARGIN),
        x1: Math.ceil(track.x1 + CAMERA_MARKER_MARGIN),
        y1: Math.ceil(track.y1 + CAMERA_MARKER_MARGIN),
      }, width, height);
      if (trackBox) {
        ctx.putImageData(image, 0, 0, trackBox.x0, trackBox.y0, trackBox.x1 - trackBox.x0 + 1, trackBox.y1 - trackBox.y0 + 1);
      }
    }
  }, []);

  /** 把同一輪的多個事件合併成一次重畫（pointermove 不再同步重畫）。 */
  const scheduleDraw = useCallback((full = false) => {
    if (full) fullRedrawRef.current = true;
    if (frameRef.current != null) return;
    frameRef.current = window.requestAnimationFrame(() => {
      frameRef.current = null;
      draw();
    });
  }, [draw]);

  // ---------------------------------------------------------------- 解析度
  const resize = useCallback((width: number, height: number, dprValue: number) => {
    const nextWidth = Math.max(1, Math.floor(width * dprValue));
    const nextHeight = Math.max(1, Math.floor(height * dprValue));
    if (lastSizeRef.current.width === nextWidth && lastSizeRef.current.height === nextHeight) return;
    lastSizeRef.current = { width: nextWidth, height: nextHeight };
    dprValueRef.current = dprValue;
    sizeRef.current = { width: nextWidth, height: nextHeight };
    // 畫布內容會隨尺寸設定被清空，所以下一幀一定要整張重畫
    scheduleDraw(true);
  }, [scheduleDraw]);

  /** 拖曳／滾輪期間降到 1 倍解析度，結束後恢復並整張重畫。 */
  const setInteractive = useCallback((active: boolean) => {
    const target = active ? INTERACTIVE_DPR : baseDprRef.current;
    if (dprValueRef.current === target) {
      scheduleDraw();
      return;
    }
    setDpr(target);
    const rect = canvasRef.current?.getBoundingClientRect();
    if (rect && rect.width > 0 && rect.height > 0) resize(rect.width, rect.height, target);
    else scheduleDraw(true);
  }, [resize, scheduleDraw]);

  // ---------------------------------------------------------------- 取得點雲
  const refresh = useCallback(async () => {
    const generation = generationRef.current;
    // in-flight 去重：慢速磁碟不會讓同一份快照的請求堆疊
    if (inFlightRef.current) return;
    inFlightRef.current = true;
    try {
      const next = await getSnapshotCloud(projectPath, MAX_POINTS);
      // await 之後才寫 state，必須先確認元件還在、專案還沒換
      if (generationRef.current !== generation) return;
      const invalid = shapeErrorOf(next);
      if (invalid) {
        attemptRef.current += 1;
        failuresRef.current = attemptRef.current;
        setFailure(invalid);
        return;
      }
      attemptRef.current = 0;
      failuresRef.current = 0;
      cloudRef.current = next;
      sceneRef.current = null;
      setCloud((previous) => {
        // 快照沒變就不重繪，避免執行中每 1.5 秒無謂重畫
        const same = previous != null && sceneKeyOf(previous) === sceneKeyOf(next);
        return same ? previous : next;
      });
      setFailure(null);
    } catch (cause) {
      if (generationRef.current !== generation) return;
      attemptRef.current += 1;
      failuresRef.current = attemptRef.current;
      setFailure(messageOf(cause));
    } finally {
      // 世代若已改變，這輪的旗標屬於上一個專案，不能據此放行新專案的請求
      if (generationRef.current === generation) {
        inFlightRef.current = false;
        setLoading(false);
      }
    }
  }, [messageOf, projectPath]);

  // 首次抓取與輪詢收在同一個 effect：`live` 為 false 時完全不建立計時器
  useEffect(() => {
    if (!live) {
      void refresh();
      return;
    }
    const generation = generationRef.current;
    let timer: number | null = null;
    const tick = () => {
      timer = null;
      void refresh().then(() => {
        if (generationRef.current !== generation) return;
        // 連續失敗時退避，退避用完就停止輪詢（保留最後一次成功的畫面）
        const count = failuresRef.current;
        if (count === 0) {
          timer = window.setTimeout(tick, POLL_INTERVAL_MS);
          return;
        }
        const delay = BACKOFF_INTERVAL_MS[count - 1];
        if (delay == null) {
          setPolling(false);
          return;
        }
        timer = window.setTimeout(tick, delay);
      });
    };
    void refresh().then(() => {
      if (generationRef.current === generation) timer = window.setTimeout(tick, POLL_INTERVAL_MS);
    });
    return () => {
      if (timer != null) window.clearTimeout(timer);
    };
  }, [live, refresh]);

  const retry = useCallback(() => {
    attemptRef.current = 0;
    failuresRef.current = 0;
    setPolling(live);
    setLoading(true);
    void refresh();
  }, [live, refresh]);

  // ---------------------------------------------------------------- 場景資料
  // 依「來源身分」快取轉換結果：同一次 refresh 之後的每一幀都不再重建
  // Float32Array，也不再為相機軌跡重算一次包圍盒。
  const sceneBounds = useMemo(() => {
    if (!cloud || cloud.point_count === 0) return null;
    const key = sceneKeyOf(cloud);
    const cached = sceneRef.current;
    if (cached?.key === key) return cached.bounds;
    const count = Math.min(cloud.point_count, MAX_POINTS);
    // boundsOf 接受 ArrayLike<number>，所以不必先複製成 Float32Array 再算包圍盒
    const positions = Float32Array.from(cloud.positions);
    const colors = Float32Array.from(cloud.colors);
    const cameras = Float32Array.from(cloud.cameras);
    const cloudBounds = {
      min: [cloud.bounds[0], cloud.bounds[1], cloud.bounds[2]] as [number, number, number],
      max: [cloud.bounds[3], cloud.bounds[4], cloud.bounds[5]] as [number, number, number],
    };
    // 相機軌跡也納入取景，否則拍得比較遠的相機會落在畫面外
    const cameraBounds = boundsOf(cameras);
    const scene: CachedScene = {
      key,
      positions,
      colors,
      cameras,
      count,
      bounds: unionBounds(cloudBounds, cameraBounds),
    };
    sceneRef.current = scene;
    statsRef.current = {
      points: count,
      cameras: cloud.cameras.length / 3,
      totalPoints: cloud.total_points,
      snapshot: cloud.snapshot,
      isFinal: cloud.is_final,
    };
    return scene.bounds;
  }, [cloud]);

  /** 目前畫布的長寬比（fitCamera 需要它才不會在直向視窗裁掉左右兩側）。 */
  const aspectRatio = useCallback(() => {
    const { width, height } = sizeRef.current;
    return height > 0 ? width / height : 1;
  }, []);

  const applyFit = useCallback(() => {
    const bounds = sceneRef.current?.bounds;
    if (!bounds) return;
    commitCamera(fitCamera(bounds, undefined, aspectRatio()));
    scheduleDraw(true);
  }, [aspectRatio, commitCamera, scheduleDraw]);

  useEffect(() => {
    if (!sceneBounds) return;
    // 自動取景只在第一次拿到場景與使用者重設時發生，之後不搶使用者的視角
    if (!cameraRef.current || autoFit) {
      applyFit();
      if (autoFit) setAutoFit(false);
    }
  }, [applyFit, autoFit, sceneBounds]);

  useEffect(() => {
    if (!cloud) statsRef.current = null;
    scheduleDraw(true);
  }, [cloud, scheduleDraw]);

  // 畫布大小：ResizeObserver 觀察 .pointcloud-stage（底部統計列在窄視窗
  // 會 flex-wrap 變高，stage 高度跟著變，但不會觸發 window resize），
  // 同一幀的多個通知用 rAF 合併。window resize 保留作保險。
  //
  // 這個 effect 必須排在「讀取裝置像素比」的 effect 之後：第一次量測要用
  // 真正的 DPR，否則畫布會先用 1 倍建立，得等到下一次通知才會修正。
  useEffect(() => {
    baseDprRef.current = Math.min(window.devicePixelRatio || 1, MAX_IDLE_DPR);
    // 拖曳中不改解析度，免得起點與後續量測互相打架
    if (!dragRef.current) {
      dprValueRef.current = baseDprRef.current;
      setDpr(baseDprRef.current);
    }
  }, []);

  useEffect(() => {
    const canvas = canvasRef.current;
    const stage = canvas?.parentElement;
    if (!canvas || !stage) return;
    const apply = () => {
      const rect = canvas.getBoundingClientRect();
      if (rect.width <= 0 || rect.height <= 0) return;
      resize(rect.width, rect.height, dprValueRef.current);
    };
    apply();
    let pending: number | null = null;
    const schedule = () => {
      if (pending != null) return;
      pending = window.requestAnimationFrame(() => {
        pending = null;
        apply();
      });
    };
    const observer = typeof ResizeObserver === "undefined" ? null : new ResizeObserver(schedule);
    observer?.observe(stage);
    window.addEventListener("resize", schedule);
    return () => {
      observer?.disconnect();
      window.removeEventListener("resize", schedule);
      if (pending != null) window.cancelAnimationFrame(pending);
    };
  }, [resize]);

  useEffect(() => () => {
    if (frameRef.current != null) window.cancelAnimationFrame(frameRef.current);
    frameRef.current = null;
  }, []);

  // ------------------------------------------------------------------ 互動
  const onPointerDown = (event: React.PointerEvent<HTMLCanvasElement>) => {
    if (event.button !== 0) return;
    dragRef.current = { x: event.clientX, y: event.clientY };
    event.currentTarget.setPointerCapture?.(event.pointerId);
    setDragging(true);
    setInteractive(true);
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
    commitCamera({
      ...camera,
      azimuth: camera.azimuth - dx * gain,
      elevation: clampElevation(camera.elevation + dy * gain),
    });
    scheduleDraw();
  };
  const onPointerUp = () => {
    dragRef.current = null;
    setDragging(false);
    setInteractive(false);
  };

  // React 19 的 onWheel 是被動監聽，無法 preventDefault，所以改在 effect 內
  // 用原生監聽器（passive: false），否則滾輪縮放會捲動底層的 .preview-pane。
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const onWheel = (event: WheelEvent) => {
      event.preventDefault();
      const camera = cameraRef.current;
      if (!camera) return;
      commitCamera({
        ...camera,
        distance: Math.max(0.05, camera.distance * (event.deltaY > 0 ? 1.12 : 1 / 1.12)),
      });
      if (dprValueRef.current !== INTERACTIVE_DPR) setInteractive(true);
      // 連續滾輪期間維持 1 倍解析度，停下來 160ms 後才恢復高解析度
      if (wheelTimerRef.current != null) window.clearTimeout(wheelTimerRef.current);
      wheelTimerRef.current = window.setTimeout(() => {
        wheelTimerRef.current = null;
        setInteractive(false);
      }, 160);
      scheduleDraw();
    };
    canvas.addEventListener("wheel", onWheel, { passive: false });
    return () => {
      canvas.removeEventListener("wheel", onWheel);
      if (wheelTimerRef.current != null) {
        window.clearTimeout(wheelTimerRef.current);
        wheelTimerRef.current = null;
      }
    };
  }, [commitCamera, scheduleDraw, setInteractive]);

  const resetView = () => {
    setAutoFit(false);
    applyFit();
  };

  /** 鍵盤操作沿用 App 既有寫法（方向鍵旋轉、+/- 縮放、Home 重設）。 */
  const onKeyDown = (event: React.KeyboardEvent<HTMLCanvasElement>) => {
    const camera = cameraRef.current;
    if (!camera) return;
    const gain = camera.fov / 12;
    const next: OrbitCamera = { ...camera };
    let handled = true;
    switch (event.key) {
      case "ArrowLeft": next.azimuth -= gain; break;
      case "ArrowRight": next.azimuth += gain; break;
      case "ArrowUp": next.elevation = clampElevation(next.elevation + gain); break;
      case "ArrowDown": next.elevation = clampElevation(next.elevation - gain); break;
      case "+":
      case "=": next.distance = Math.max(0.05, next.distance / 1.15); break;
      case "-":
      case "_": next.distance = Math.min(1e6, next.distance * 1.15); break;
      case "Home": resetView(); return;
      default: handled = false;
    }
    if (!handled) return;
    event.preventDefault();
    commitCamera(next);
    scheduleDraw();
  };

  return (
    <section className="pointcloud-viewer">
      <header className="pointcloud-topbar">
        <div className="pointcloud-title">
          <Crosshair size={16} />
          <strong>{t("pointcloud.title")}</strong>
          {live && <span className="pointcloud-live"><span className="status-light" />{t("pointcloud.live")}</span>}
          {failure && polling && <span className="pointcloud-badge" role="status">{t("pointcloud.reconnecting")}</span>}
          {failure && !polling && (
            <span className="pointcloud-badge error" role="status">
              <span>{renderedFailure}</span>
              <button type="button" className="settings-action" onClick={retry}>{t("common.retry")}</button>
            </span>
          )}
        </div>
        <div className="pointcloud-actions">
          <button type="button" className="settings-action" onClick={resetView}><RotateCcw size={15} />{t("pointcloud.resetView")}</button>
          <button type="button" className="settings-action" onClick={onExit}><X size={15} />{t("pointcloud.exit")}</button>
        </div>
      </header>

      <div className="pointcloud-stage">
        <canvas
          ref={canvasRef}
          role="img"
          aria-label={t("pointcloud.canvasAria")}
          tabIndex={0}
          style={dragging ? CURSOR_GRABBING : CURSOR_GRAB}
          onPointerDown={onPointerDown}
          onPointerMove={onPointerMove}
          onPointerUp={onPointerUp}
          onPointerCancel={onPointerUp}
          onKeyDown={onKeyDown}
        />
        {loading && (
          <div className="pointcloud-overlay"><LoaderCircle className="spin" size={22} /><span>{t("pointcloud.loading")}</span></div>
        )}
        {!loading && cloud && cloud.point_count === 0 && (
          <div className="pointcloud-overlay"><span>{t("pointcloud.empty")}</span></div>
        )}
      </div>

      <footer className="pointcloud-stats" aria-live="polite">
        <div><dt>{t("pointcloud.points")}</dt><dd>{stats ? formatNumber(stats.points) : "—"}</dd></div>
        <div><dt>{t("pointcloud.totalPoints")}</dt><dd>{stats ? formatNumber(stats.totalPoints) : "—"}</dd></div>
        <div><dt>{t("pointcloud.cameras")}</dt><dd>{stats ? formatNumber(stats.cameras) : "—"}</dd></div>
        <div><dt>{t("pointcloud.snapshot")}</dt><dd>{stats?.snapshot ?? "—"}{stats?.isFinal ? `（${t("pointcloud.finalModel")}）` : ""}</dd></div>
        <div><dt>{t("pointcloud.snapshots")}</dt><dd>{cloud ? cloud.snapshot_count : "—"}</dd></div>
        <div><dt>{t("pointcloud.distance")}</dt><dd>{camera ? camera.distance.toFixed(2) : "—"}</dd></div>
      </footer>
    </section>
  );
}

/** 兩兩合併髒矩形。 */
function unionBox(a: DirtyBox, b: DirtyBox): DirtyBox {
  return {
    x0: Math.min(a.x0, b.x0),
    y0: Math.min(a.y0, b.y0),
    x1: Math.max(a.x1, b.x1),
    y1: Math.max(a.y1, b.y1),
  };
}

/** 把髒矩形夾進畫布；完全落在畫布外時回傳 null。 */
function intersectBox(box: DirtyBox, width: number, height: number): DirtyBox | null {
  const x0 = Math.max(0, box.x0);
  const y0 = Math.max(0, box.y0);
  const x1 = Math.min(width - 1, box.x1);
  const y1 = Math.min(height - 1, box.y1);
  return x1 < x0 || y1 < y0 ? null : { x0, y0, x1, y1 };
}

/**
 * 畫出相機軌跡：依 `cameras` 的順序連成折線，並用透明度漸層與首尾標記
 * 表達拍攝方向（原本只畫 3×3 的方點，看不出順序與方向）。
 *
 * 回傳畫到的範圍（螢幕座標），讓呼叫端能把這塊區域一起回寫。
 */
function drawCameraTrack(
  ctx: CanvasRenderingContext2D,
  camera: OrbitCamera,
  width: number,
  height: number,
  cameras: Float32Array | null,
): { x0: number; y0: number; x1: number; y1: number } | null {
  const count = cameras ? Math.floor(cameras.length / 3) : 0;
  if (!cameras || count < 2) return null;
  const projected = projectPoints(cameras, camera, width, height);
  // 投影到相機後方的中心標成 NaN，讓折線在該處斷開而不是橫跨整個畫面
  for (let i = 0; i < count; i += 1) {
    if (projected.depth[i] < 0) {
      projected.screen[i * 2] = Number.NaN;
      projected.screen[i * 2 + 1] = Number.NaN;
    }
  }
  const first = visibleIndex(projected.screen, 0, 1);
  const last = visibleIndex(projected.screen, count - 1, -1);
  if (first < 0 || last < 0) return null;
  const start = Math.min(first, last);
  const end = Math.max(first, last);
  const startX = projected.screen[start * 2];
  const startY = projected.screen[start * 2 + 1];
  const endX = projected.screen[end * 2];
  const endY = projected.screen[end * 2 + 1];

  ctx.save();
  // 由淡到亮 = 拍攝順序（軌跡反轉時顏色也跟著反轉）
  const gradient = ctx.createLinearGradient(startX, startY, endX, endY);
  const faded = "rgba(56, 232, 255, 0.22)";
  const bright = "rgba(56, 232, 255, 0.95)";
  gradient.addColorStop(0, first <= last ? faded : bright);
  gradient.addColorStop(1, first <= last ? bright : faded);
  ctx.strokeStyle = gradient;
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  let minX = Infinity;
  let minY = Infinity;
  let maxX = -Infinity;
  let maxY = -Infinity;
  for (let i = start; i <= end; i += 1) {
    const sx = projected.screen[i * 2];
    const sy = projected.screen[i * 2 + 1];
    if (!Number.isFinite(sx) || !Number.isFinite(sy)) continue;
    if (i === start) ctx.moveTo(sx, sy);
    else ctx.lineTo(sx, sy);
    if (sx < minX) minX = sx;
    if (sx > maxX) maxX = sx;
    if (sy < minY) minY = sy;
    if (sy > maxY) maxY = sy;
  }
  ctx.stroke();

  // 起點是空心圓，終點是較亮的實心圓
  ctx.lineWidth = 1.2;
  ctx.globalAlpha = 0.6;
  ctx.strokeStyle = TRAJECTORY_COLOR;
  ctx.beginPath();
  ctx.arc(startX, startY, 2.5, 0, Math.PI * 2);
  ctx.stroke();
  ctx.globalAlpha = 1;
  ctx.fillStyle = TRAJECTORY_HEAD_COLOR;
  ctx.beginPath();
  ctx.arc(endX, endY, 3, 0, Math.PI * 2);
  ctx.fill();
  ctx.restore();
  for (const [x, y] of [[startX, startY], [endX, endY]] as const) {
    if (x < minX) minX = x;
    if (x > maxX) maxX = x;
    if (y < minY) minY = y;
    if (y > maxY) maxY = y;
  }
  return Number.isFinite(minX) ? { x0: minX, y0: minY, x1: maxX, y1: maxY } : null;
}
