import { describe, expect, it } from "vitest";
import {
  boundsOf,
  cameraPosition,
  clampElevation,
  fitCamera,
  projectPoints,
  unionBounds,
  type OrbitCamera,
} from "./projection";

describe("點雲投影", () => {
  it("注視點應投影到畫面正中央", () => {
    const camera = fitCamera({ min: [-1, -1, -1], max: [1, 1, 1] });
    // 把一個點放在注視點上
    const positions = new Float32Array(camera.target);
    const { screen, depth, visible } = projectPoints(positions, camera, 800, 600);
    expect(screen[0]).toBeCloseTo(400, 4);
    expect(screen[1]).toBeCloseTo(300, 4);
    expect(depth[0]).toBeGreaterThan(0);
    expect(visible).toBe(1);
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
    const { depth, visible } = projectPoints(positions, camera, 400, 400);
    expect(depth[0]).toBe(-1);
    expect(visible).toBe(0);
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
    const bounds = { min: [-10, -5, -8] as [number, number, number], max: [10, 5, 8] as [number, number, number] };
    const camera = fitCamera(bounds);
    expect(camera.distance).toBeGreaterThan(10);
    // 八個角都應投影在畫面內
    const corners = new Float32Array(24);
    let i = 0;
    for (const x of [bounds.min[0], bounds.max[0]]) {
      for (const y of [bounds.min[1], bounds.max[1]]) {
        for (const z of [bounds.min[2], bounds.max[2]]) {
          corners[i++] = x;
          corners[i++] = y;
          corners[i++] = z;
        }
      }
    }
    const { visible } = projectPoints(corners, camera, 800, 600);
    expect(visible).toBe(8);
  });

  it("仰角會被夾在 ±85° 內（避免萬向鎖）", () => {
    expect(clampElevation(Math.PI)).toBeCloseTo((85 * Math.PI) / 180, 6);
    expect(clampElevation(-Math.PI)).toBeCloseTo((-85 * Math.PI) / 180, 6);
    expect(clampElevation(0)).toBe(0);
  });

  it("相機位置與距離一致", () => {
    const camera = fitCamera({ min: [0, 0, 0], max: [2, 2, 2] });
    const eye = cameraPosition(camera);
    const d = Math.hypot(eye[0] - camera.target[0], eye[1] - camera.target[1], eye[2] - camera.target[2]);
    expect(d).toBeCloseTo(camera.distance, 4);
  });

  it("包圍盒：空輸入安全、合併正確", () => {
    expect(boundsOf(new Float32Array([]))).toEqual({ min: [0, 0, 0], max: [0, 0, 0] });
    const b = boundsOf(new Float32Array([1, 2, 3, -1, -2, -3]));
    expect(b.min).toEqual([-1, -2, -3]);
    expect(b.max).toEqual([1, 2, 3]);
    const u = unionBounds({ min: [0, 0, 0], max: [1, 1, 1] }, { min: [-2, 0, 0], max: [0, 5, 0] });
    expect(u.min).toEqual([-2, 0, 0]);
    expect(u.max).toEqual([1, 5, 1]);
  });
});
