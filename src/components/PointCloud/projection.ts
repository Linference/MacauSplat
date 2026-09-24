/**
 * 點雲的透視投影與自動取景（純函式，可單元測試）。
 *
 * 為什麼自己寫投影而不是直接引入三維引擎：
 * 應用內已經有 PlayCanvas（GaussianViewer 在用），但它綁定高斯潑濺的渲染管線；
 * 點雲只需要「把幾萬個點畫成有深度的彩色像素」，用 Canvas 2D + 自寫投影
 * 反而更小、更可測、也不會與既有渲染器互相搶 GPU 資源。
 * 若日後要升級成 PlayCanvas 點渲染，只要替換 draw()，投影與取景邏輯可原樣沿用。
 */

export type Vec3 = [number, number, number];

export interface Bounds {
  min: Vec3;
  max: Vec3;
}

/** 軌道相機：以目標點為中心，用方位角／仰角／距離描述。 */
export interface OrbitCamera {
  /** 方位角（弧度）。 */
  azimuth: number;
  /** 仰角（弧度），限制在 ±85° 以免萬向鎖。 */
  elevation: number;
  /** 與目標點的距離（世界單位）。 */
  distance: number;
  /** 注視點。 */
  target: Vec3;
  /** 垂直視野（弧度）。 */
  fov: number;
}

export const clampElevation = (elevation: number): number => {
  const limit = (85 * Math.PI) / 180;
  return Math.max(-limit, Math.min(limit, elevation));
};

/**
 * 由包圍盒取得建議的相機（自動取景：讓整個點雲落在視野內）。
 *
 * `fov` 是**垂直**視野，而 `projectPoints` 的水平視野由 aspect 決定
 * （水平半角 = atan(tan(fov/2) × width/height)）。
 * 直向視窗（aspect < 1）的水平視野比垂直視野窄，若只用垂直視野推距離，
 * 左右兩側會被裁掉，因此取兩者中較窄的視野半角來推距離。
 */
export function fitCamera(
  bounds: Bounds,
  fov = (50 * Math.PI) / 180,
  aspect = 1,
): OrbitCamera {
  const center: Vec3 = [
    (bounds.min[0] + bounds.max[0]) / 2,
    (bounds.min[1] + bounds.max[1]) / 2,
    (bounds.min[2] + bounds.max[2]) / 2,
  ];
  const size = Math.max(
    bounds.max[0] - bounds.min[0],
    bounds.max[1] - bounds.min[1],
    bounds.max[2] - bounds.min[2],
    1e-3,
  );
  // 球體半徑 → 需要的距離（留 1.35 倍餘裕，避免貼邊）
  const radius = size * 0.866;
  const safeAspect = Number.isFinite(aspect) && aspect > 0 ? aspect : 1;
  const halfAngle = Math.atan(Math.tan(fov / 2) * Math.min(1, safeAspect));
  const distance = (radius / Math.sin(halfAngle)) * 1.35;
  return { azimuth: Math.PI * 0.25, elevation: Math.PI * 0.18, distance, target: center, fov };
}

/** 相機在世界座標的位置。 */
export function cameraPosition(camera: OrbitCamera): Vec3 {
  const { azimuth, elevation, distance, target } = camera;
  return [
    target[0] + distance * Math.cos(elevation) * Math.cos(azimuth),
    target[1] + distance * Math.sin(elevation),
    target[2] + distance * Math.cos(elevation) * Math.sin(azimuth),
  ];
}

export interface ProjectedPoints {
  /**
   * 螢幕座標（交錯 xy；長度 = 2 × 點數）。
   * 深度不足（在相機後方或太靠近近平面）的點為 `[-1, -1]`；
   * 其餘點可能是畫面外的座標，呼叫端必須自行做範圍裁切。
   */
  screen: Float32Array;
  /** 每點的相機深度（用於排序與大小衰減）；`-1` 表示不可見。 */
  depth: Float32Array;
}

/**
 * 把世界座標的點投影到螢幕。
 *
 * 以「相機位於球面上、看向 target」建立視座標系，再做透視除法。
 * 深度 ≤ 近平面（0.01 倍距離）的點會被標記為不可見（depth = -1、screen = [-1, -1]）。
 */
export function projectPoints(
  positions: ArrayLike<number>,
  camera: OrbitCamera,
  width: number,
  height: number,
): ProjectedPoints {
  const count = Math.floor(positions.length / 3);
  const screen = new Float32Array(count * 2);
  const depth = new Float32Array(count);
  const eye = cameraPosition(camera);

  // 前向量與右／上向量（世界座標）
  const forward = normalize([
    camera.target[0] - eye[0],
    camera.target[1] - eye[1],
    camera.target[2] - eye[2],
  ]);
  // 以世界 Y 軸為「上」的參考（建築與室內場景通常是 Y 向上；此處僅影響畫面朝向）
  let right = normalize(cross(forward, [0, 1, 0]));
  if (!Number.isFinite(right[0]) || Math.hypot(right[0], right[1], right[2]) < 1e-6) {
    right = [1, 0, 0];
  }
  const up = cross(right, forward);

  const focal = height / (2 * Math.tan(camera.fov / 2));
  const near = camera.distance * 0.01;

  for (let i = 0; i < count; i += 1) {
    const dx = positions[i * 3] - eye[0];
    const dy = positions[i * 3 + 1] - eye[1];
    const dz = positions[i * 3 + 2] - eye[2];
    const z = dx * forward[0] + dy * forward[1] + dz * forward[2]; // 深度
    if (z <= near) {
      depth[i] = -1;
      screen[i * 2] = -1;
      screen[i * 2 + 1] = -1;
      continue;
    }
    const x = dx * right[0] + dy * right[1] + dz * right[2];
    const y = dx * up[0] + dy * up[1] + dz * up[2];
    screen[i * 2] = width / 2 + (x * focal) / z;
    screen[i * 2 + 1] = height / 2 - (y * focal) / z;
    depth[i] = z;
  }
  return { screen, depth };
}

/**
 * 由相機中心點計算包圍盒（供相機軌跡顯示用）。
 * 接受任何 `ArrayLike<number>`，呼叫端不必先複製成 `Float32Array`。
 */
export function boundsOf(positions: ArrayLike<number>): Bounds {
  if (positions.length === 0) {
    return { min: [0, 0, 0], max: [0, 0, 0] };
  }
  const min: Vec3 = [Infinity, Infinity, Infinity];
  const max: Vec3 = [-Infinity, -Infinity, -Infinity];
  for (let i = 0; i < positions.length; i += 3) {
    for (let k = 0; k < 3; k += 1) {
      const v = positions[i + k];
      if (v < min[k]) min[k] = v;
      if (v > max[k]) max[k] = v;
    }
  }
  return { min, max };
}

/**
 * 由一連串二維點算出像素髒矩形（含 `margin` 邊距，涵蓋相機方點的半寬）。
 *
 * 只回寫「真的有畫到」的區域，避免每幀對整個畫布 `putImageData`。
 * 沒有任何有限值時回傳 `null`（呼叫端據此跳過回寫）。
 */
export function elementBounds(
  screen: Float32Array,
  depth: Float32Array,
  width: number,
  height: number,
  margin = 0,
): { x0: number; y0: number; x1: number; y1: number } | null {
  let x0 = Infinity;
  let y0 = Infinity;
  let x1 = -Infinity;
  let y1 = -Infinity;
  const count = Math.min(Math.floor(screen.length / 2), depth.length);
  for (let i = 0; i < count; i += 1) {
    if (depth[i] < 0) continue;
    const sx = screen[i * 2];
    const sy = screen[i * 2 + 1];
    if (!Number.isFinite(sx) || !Number.isFinite(sy)) continue;
    if (sx < x0) x0 = sx;
    if (sx > x1) x1 = sx;
    if (sy < y0) y0 = sy;
    if (sy > y1) y1 = sy;
  }
  if (x0 > x1 || y0 > y1) return null;
  // 夾在畫布內：超出畫布的部分不需要回寫
  const clampedX0 = Math.max(0, Math.floor(x0 - margin));
  const clampedY0 = Math.max(0, Math.floor(y0 - margin));
  const clampedX1 = Math.min(width - 1, Math.ceil(x1 + margin));
  const clampedY1 = Math.min(height - 1, Math.ceil(y1 + margin));
  if (clampedX1 < clampedX0 || clampedY1 < clampedY0) return null;
  return { x0: clampedX0, y0: clampedY0, x1: clampedX1, y1: clampedY1 };
}

/**
 * 從 `start` 往 `step` 方向找出第一個投影有效的索引（用於相機軌跡的首尾標記）。
 *
 * 相機在畫面外的座標仍然有效（只是被裁掉），指標必須停在序列兩端；
 * 只有投影到相機後方（`NaN`）的項目才需要跳過。
 */
export function visibleIndex(screen: Float32Array, start: number, step: number): number {
  const count = Math.floor(screen.length / 2);
  for (let i = start; i >= 0 && i < count; i += step) {
    if (Number.isFinite(screen[i * 2]) && Number.isFinite(screen[i * 2 + 1])) return i;
  }
  return -1;
}

/** 合併兩個包圍盒（點雲 + 相機軌跡一起取景）。 */
export function unionBounds(a: Bounds, b: Bounds): Bounds {
  return {
    min: [Math.min(a.min[0], b.min[0]), Math.min(a.min[1], b.min[1]), Math.min(a.min[2], b.min[2])],
    max: [Math.max(a.max[0], b.max[0]), Math.max(a.max[1], b.max[1]), Math.max(a.max[2], b.max[2])],
  };
}

function cross(a: Vec3, b: Vec3): Vec3 {
  return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
}

function normalize(v: Vec3): Vec3 {
  const n = Math.hypot(v[0], v[1], v[2]);
  return n < 1e-9 ? [0, 0, 0] : [v[0] / n, v[1] / n, v[2] / n];
}
