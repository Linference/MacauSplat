import { describe, expect, it } from "vitest";
import {
  boundsOf,
  cameraPosition,
  clampElevation,
  elementBounds,
  fitCamera,
  projectPoints,
  unionBounds,
  type Bounds,
  type OrbitCamera,
} from "./projection";

const bounds = (min: [number, number, number], max: [number, number, number]): Bounds => ({ min, max });

/** 包圍盒的八個角（交錯 xyz），用來檢查自動取景是否真的把整個模型收進畫面。 */
function cornersOf(box: Bounds): Float32Array {
  const corners = new Float32Array(24);
  let i = 0;
  for (const x of [box.min[0], box.max[0]]) {
    for (const y of [box.min[1], box.max[1]]) {
      for (const z of [box.min[2], box.max[2]]) {
        corners[i++] = x;
        corners[i++] = y;
        corners[i++] = z;
      }
    }
  }
  return corners;
}

/** 「可見」＝深度有效且螢幕座標落在畫面內；改用這個取代已移除的 ProjectedPoints.visible。 */
function visiblePoints(
  positions: Float32Array,
  camera: OrbitCamera,
  width: number,
  height: number,
): number {
  const { screen, depth } = projectPoints(positions, camera, width, height);
  let visible = 0;
  for (let i = 0; i < depth.length; i += 1) {
    if (depth[i] < 0) continue;
    if (screen[i * 2] >= 0 && screen[i * 2] < width && screen[i * 2 + 1] >= 0 && screen[i * 2 + 1] < height) {
      visible += 1;
    }
  }
  return visible;
}

describe("點雲投影", () => {
  it("注視點應投影到畫面正中央", () => {
    const camera = fitCamera(bounds([-1, -1, -1], [1, 1, 1]));
    // 把一個點放在注視點上
    const positions = new Float32Array(camera.target);
    const { screen, depth } = projectPoints(positions, camera, 800, 600);
    expect(screen[0]).toBeCloseTo(400, 4);
    expect(screen[1]).toBeCloseTo(300, 4);
    expect(depth[0]).toBeGreaterThan(0);
    expect(visiblePoints(positions, camera, 800, 600)).toBe(1);
  });

  it("相機後方的點應標記為不可見", () => {
    const camera: OrbitCamera = {
      azimuth: 0,
      elevation: 0,
      distance: 10,
      target: [0, 0, 0],
      fov: Math.PI / 3,
    };
    // 相機位置為 (10,0,0)、看向原點；把點放在相機背後 (20,0,0)
    const positions = new Float32Array([20, 0, 0]);
    const { depth, screen } = projectPoints(positions, camera, 400, 400);
    expect(depth[0]).toBe(-1);
    expect(screen[0]).toBe(-1);
    expect(screen[1]).toBe(-1);
    expect(visiblePoints(positions, camera, 400, 400)).toBe(0);
  });

  it("距離越遠的點投影越靠中央（透視正確）", () => {
    const camera: OrbitCamera = {
      azimuth: 0,
      elevation: 0,
      distance: 5,
      target: [0, 0, 0],
      fov: Math.PI / 3,
    };
    // 相機在 (5,0,0) 且看向 −X，因此「深度」沿世界 X 軸變化。
    // 兩個點的側向偏移相同（y = 1），但深度不同（5 與 7）→ 較遠者的畫面偏移必須較小。
    const positions = new Float32Array([0, 1, 0, -2, 1, 0]);
    const { screen, depth } = projectPoints(positions, camera, 400, 400);
    expect(depth[0]).toBeCloseTo(5, 5);
    expect(depth[1]).toBeCloseTo(7, 5);
    const offsetNear = Math.abs(screen[1] - 200);
    const offsetFar = Math.abs(screen[3] - 200);
    expect(offsetFar).toBeLessThan(offsetNear);
    // 透視關係應精確成立：偏移 ∝ 1/深度
    expect(offsetNear / offsetFar).toBeCloseTo(7 / 5, 3);
  });

  it("自動取景讓點雲落在畫面內", () => {
    const box = bounds([-10, -5, -8], [10, 5, 8]);
    const camera = fitCamera(box, undefined, 800 / 600);
    expect(camera.distance).toBeGreaterThan(10);
    // 八個角都應投影在畫面內
    expect(visiblePoints(cornersOf(box), camera, 800, 600)).toBe(8);
  });

  it("直向視窗的自動取景會依 aspect 拉遠（回歸：fitCamera 需考慮 aspect）", () => {
    // 800×600 是橫向，垂直視野是最嚴格的方向；600×800 是直向，水平視野反而更窄
    // （水平半角 = atan(tan(fov/2) × width/height)），必須站得更遠才不會裁掉左右兩側。
    const box = bounds([-150, -4, -4], [150, 4, 4]);
    const halfAngleFor = (aspect: number) =>
      Math.atan(Math.tan((50 * Math.PI) / 180 / 2) * Math.min(1, aspect));
    const requiredDistance = (aspect: number) =>
      ((300 * 0.866) / Math.sin(halfAngleFor(aspect))) * 1.35;

    const landscape = fitCamera(box, undefined, 800 / 600);
    const portrait = fitCamera(box, undefined, 600 / 800);

    // 橫向：水平視野比垂直寬，垂直視野才是瓶頸
    expect(landscape.distance).toBeCloseTo(requiredDistance(800 / 600), 6);
    // 直向：必須改用較窄的水平視野推距離
    expect(portrait.distance).toBeCloseTo(requiredDistance(600 / 800), 6);
    expect(portrait.distance).toBeGreaterThan(landscape.distance);
    // 舊行為（忽略 aspect）在這個模型上會少站約 1/3 的距離
    expect(landscape.distance / portrait.distance).toBeLessThan(0.8);

    // 兩種視窗下八個角都必須落在畫面內
    expect(visiblePoints(cornersOf(box), landscape, 800, 600)).toBe(8);
    expect(visiblePoints(cornersOf(box), portrait, 600, 800)).toBe(8);

    // 正方形視窗與不合法的 aspect 都退回垂直視野
    const square = bounds([-8, -12, -8], [8, 12, 8]);
    expect(fitCamera(square, undefined, 1).distance).toBeCloseTo(fitCamera(square).distance, 6);
    expect(fitCamera(square, undefined, Number.NaN).distance).toBeCloseTo(fitCamera(square).distance, 6);
    expect(fitCamera(square, undefined, 0).distance).toBeCloseTo(fitCamera(square).distance, 6);
  });

  it("仰角會被夾在 ±85° 內（避免萬向鎖）", () => {
    expect(clampElevation(Math.PI)).toBeCloseTo((85 * Math.PI) / 180, 6);
    expect(clampElevation(-Math.PI)).toBeCloseTo((-85 * Math.PI) / 180, 6);
    expect(clampElevation(0)).toBe(0);
  });

  it("相機位置與距離一致", () => {
    const camera = fitCamera(bounds([0, 0, 0], [2, 2, 2]));
    const eye = cameraPosition(camera);
    const d = Math.hypot(eye[0] - camera.target[0], eye[1] - camera.target[1], eye[2] - camera.target[2]);
    expect(d).toBeCloseTo(camera.distance, 4);
  });

  it("包圍盒：空輸入安全、合併正確", () => {
    expect(boundsOf(new Float32Array([]))).toEqual({ min: [0, 0, 0], max: [0, 0, 0] });
    const b = boundsOf(new Float32Array([1, 2, 3, -1, -2, -3]));
    expect(b.min).toEqual([-1, -2, -3]);
    expect(b.max).toEqual([1, 2, 3]);
    const u = unionBounds(bounds([0, 0, 0], [1, 1, 1]), bounds([-2, 0, 0], [0, 5, 0]));
    expect(u.min).toEqual([-2, 0, 0]);
    expect(u.max).toEqual([1, 5, 1]);
  });

  it("包圍盒直接接受 number[]（呼叫端不必先複製成 Float32Array）", () => {
    const b = boundsOf([4, -5, 6, -1, 2, 3]);
    expect(b.min).toEqual([-1, -5, 3]);
    expect(b.max).toEqual([4, 2, 6]);
    expect(boundsOf([])).toEqual({ min: [0, 0, 0], max: [0, 0, 0] });
  });

  it("髒矩形只涵蓋有畫到的區域並夾在畫布內", () => {
    const camera = fitCamera(bounds([-1, -1, -1], [1, 1, 1]));
    // 一個在畫面正中央的點 → 髒矩形應遠小於整張畫布
    const centered = projectPoints(new Float32Array(camera.target), camera, 400, 300);
    const box = elementBounds(centered.screen, centered.depth, 400, 300, 2);
    expect(box).not.toBeNull();
    expect(box!.x0).toBeGreaterThan(0);
    expect(box!.x1).toBeLessThan(399);
    expect(box!.y0).toBeGreaterThan(0);
    expect(box!.y1).toBeLessThan(299);
    // 邊距要涵蓋相機方點的半寬
    expect(box!.x1 - box!.x0).toBeLessThanOrEqual(5);

    // 畫面外的座標要被夾回畫布內，超出畫布的部分不需要回寫
    const outside = elementBounds(new Float32Array([-500, -500, 9_000, 9_000]), new Float32Array([1, 1]), 400, 300);
    expect(outside).toEqual({ x0: 0, y0: 0, x1: 399, y1: 299 });

    // 完全沒有可見點時不應要求任何回寫
    const hidden = elementBounds(new Float32Array([-1, -1]), new Float32Array([-1]), 400, 300);
    expect(hidden).toBeNull();
    expect(elementBounds(new Float32Array([]), new Float32Array([]), 400, 300)).toBeNull();
  });
});
