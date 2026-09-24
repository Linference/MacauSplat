// @vitest-environment jsdom

import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi, type MockInstance } from "vitest";
import { LanguageProvider } from "../../i18n";
import type { SnapshotCloud } from "../../types/pipeline";
import { PointCloudViewer } from "./PointCloudViewer";

const mocks = vi.hoisted(() => ({
  getSnapshotCloud: vi.fn(),
}));

vi.mock("../../lib/backend", () => ({
  getSnapshotCloud: mocks.getSnapshotCloud,
}));

/**
 * 假 canvas context：只記錄呼叫。
 *
 * jsdom 沒有 canvas 實作，`getContext("2d")` 會回 null 並印出「未實作」警告；
 * 這裡 stub 掉它，讓繪製路徑真的被執行到（而不是提早 return）。
 */
function createFakeContext() {
  return {
    canvas: null as HTMLCanvasElement | null,
    fillStyle: "",
    strokeStyle: "",
    lineWidth: 1,
    globalAlpha: 1,
    fillRect: vi.fn(),
    putImageData: vi.fn(),
    getImageData: vi.fn(),
    createImageData: vi.fn((width: number, height: number) => ({
      width,
      height,
      data: new Uint8ClampedArray(width * height * 4),
      colorSpace: "srgb" as const,
    })),
    beginPath: vi.fn(),
    moveTo: vi.fn(),
    lineTo: vi.fn(),
    stroke: vi.fn(),
    fill: vi.fn(),
    arc: vi.fn(),
    setTransform: vi.fn(),
    save: vi.fn(),
    restore: vi.fn(),
    createLinearGradient: vi.fn(() => ({ addColorStop: vi.fn() })),
  };
}

let context: ReturnType<typeof createFakeContext>;
let frameQueue: FrameRequestCallback[];
let observerCallback: (() => void) | null;
let intervalSpy: MockInstance<typeof window.setInterval>;
let originalResizeObserver: typeof globalThis.ResizeObserver | undefined;
let originalRaf: typeof window.requestAnimationFrame;
let originalCancelRaf: typeof window.cancelAnimationFrame;

/** 三點（一個垂直三角形）＋兩台相機，數值自洽，能通過形狀檢查。 */
function cloudOf(overrides: Partial<SnapshotCloud> = {}): SnapshotCloud {
  return {
    positions: [0, 1, 0, -1, -1, 0, 1, -1, 0],
    colors: [255, 0, 0, 0, 255, 0, 0, 0, 255],
    point_count: 3,
    total_points: 3,
    cameras: [0, 0, -4, 0, 0.5, -3.5],
    bounds: [-1, -2, -1, 1, 2, 1],
    snapshot: "1",
    snapshot_count: 1,
    is_final: false,
    ...overrides,
  };
}

/**
 * 執行所有排隊中的 rAF（也就是讓被節流的重畫真的發生）。
 *
 * 會反覆清空佇列：一輪 resize 先量測，再由下一次 rAF 真的改動畫布尺寸。
 */
async function flushFrames() {
  await act(async () => {
    for (let pass = 0; frameQueue.length > 0 && pass < 8; pass += 1) {
      const queued = frameQueue;
      frameQueue = [];
      for (const callback of queued) callback(performance.now());
      await Promise.resolve();
    }
  });
}

/** 建立 ResizeObserver 的假實作，讓測試可以手動觸發「stage 尺寸變了」。 */
class FakeResizeObserver {
  constructor(callback: () => void) {
    observerCallback = callback;
  }
  observe() { /* 測試不需要註冊目標 */ }
  unobserve() { /* noop */ }
  disconnect() { observerCallback = null; }
}

describe("PointCloudViewer", () => {
  let container: HTMLDivElement;
  let root: Root;
  let canvas: HTMLCanvasElement;
  let rect: { width: number; height: number };

  /** 讓測試改動「量到的」CSS 尺寸（jsdom 不做排版，一定要自己 stub）。 */
  const setRectSize = (width: number, height: number) => { rect = { width, height }; };

  const mount = async (props: Partial<{ projectPath: string; live: boolean; onExit: () => void }> = {}) => {
    await act(async () => {
      root.render(
        <LanguageProvider>
          <PointCloudViewer projectPath="E:\\Projects\\示例项目" live={false} onExit={() => undefined} {...props} />
        </LanguageProvider>,
      );
    });
    canvas = container.querySelector<HTMLCanvasElement>("canvas")!;
    // 讓量測與第一次繪製真的發生，mount 之後 DOM 就是穩定狀態
    await flushFrames();
    return canvas;
  };

  beforeEach(() => {
    (globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;
    window.localStorage.setItem("ooo-splat-language", "zh-CN");
    rect = { width: 300, height: 200 };
    context = createFakeContext();
    Object.defineProperty(HTMLCanvasElement.prototype, "getContext", {
      configurable: true,
      value: () => context,
    });
    // jsdom 沒有實作 width/height 的反射，補上瀏覽器行為
    Object.defineProperty(HTMLCanvasElement.prototype, "width", {
      configurable: true,
      get(this: HTMLCanvasElement & { __w?: number }) { return this.__w ?? 300; },
      set(this: HTMLCanvasElement & { __w?: number }, value: number) { this.__w = value; },
    });
    Object.defineProperty(HTMLCanvasElement.prototype, "height", {
      configurable: true,
      get(this: HTMLCanvasElement & { __h?: number }) { return this.__h ?? 150; },
      set(this: HTMLCanvasElement & { __h?: number }, value: number) { this.__h = value; },
    });
    Object.defineProperty(Element.prototype, "setPointerCapture", {
      configurable: true,
      value: () => undefined,
    });
    // 元件在 effect 內量測畫布，stub 必須在 mount 之前就位
    Object.defineProperty(HTMLCanvasElement.prototype, "getBoundingClientRect", {
      configurable: true,
      value() {
        return {
          width: rect.width, height: rect.height, top: 0, left: 0,
          right: rect.width, bottom: rect.height, x: 0, y: 0, toJSON: () => ({}),
        };
      },
    });
    frameQueue = [];
    originalRaf = window.requestAnimationFrame;
    originalCancelRaf = window.cancelAnimationFrame;
    window.requestAnimationFrame = ((callback: FrameRequestCallback) => {
      frameQueue.push(callback);
      return frameQueue.length;
    }) as typeof window.requestAnimationFrame;
    window.cancelAnimationFrame = (() => undefined) as typeof window.cancelAnimationFrame;
    observerCallback = null;
    originalResizeObserver = globalThis.ResizeObserver;
    globalThis.ResizeObserver = FakeResizeObserver as unknown as typeof globalThis.ResizeObserver;
    intervalSpy = vi.spyOn(window, "setInterval");
    mocks.getSnapshotCloud.mockReset().mockResolvedValue(cloudOf());
    container = document.createElement("div");
    document.body.appendChild(container);
    root = createRoot(container);
  });

  afterEach(async () => {
    await act(async () => { root.unmount(); });
    container.remove();
    window.requestAnimationFrame = originalRaf;
    window.cancelAnimationFrame = originalCancelRaf;
    if (originalResizeObserver) globalThis.ResizeObserver = originalResizeObserver;
    else Reflect.deleteProperty(globalThis, "ResizeObserver");
    intervalSpy.mockRestore();
    window.localStorage.clear();
  });

  it("顯示空點雲提示，且不會嘗試分配 ImageData", async () => {
    mocks.getSnapshotCloud.mockResolvedValue(cloudOf({ positions: [], colors: [], cameras: [], point_count: 0, total_points: 0 }));
    await mount();

    expect(container.textContent).toContain("还没有可显示的点云（建图开始后会自动出现）");
    // 空點雲不需要為整張畫布配置像素緩衝，也不該有任何回寫
    expect(context.createImageData).not.toHaveBeenCalled();
    expect(context.putImageData).not.toHaveBeenCalled();
    // 空點雲是合法狀態，不應顯示錯誤
    expect(container.querySelector(".pointcloud-badge")).toBeNull();
  });

  it("命令失敗時保留畫布並以非阻斷徽章顯示錯誤，重試後恢復", async () => {
    mocks.getSnapshotCloud.mockRejectedValueOnce(new Error("读取快照失败：磁盘暂时不可用"));
    await mount();

    const badge = container.querySelector(".pointcloud-badge");
    expect(badge?.textContent).toContain("读取快照失败");
    // 不再有覆蓋整個畫布的 .pointcloud-overlay
    expect(container.querySelector(".pointcloud-overlay")).toBeNull();
    // live=false 不會輪詢，所以停在「重試」
    expect(badge?.querySelector("button")?.textContent).toBe("重试");
    expect(container.textContent).toContain("已显示点数");

    mocks.getSnapshotCloud.mockResolvedValue(cloudOf());
    await act(async () => badge?.querySelector<HTMLButtonElement>("button")?.click());
    await flushFrames();
    expect(container.querySelector(".pointcloud-badge")).toBeNull();
    expect(container.querySelectorAll(".pointcloud-stats dd")[3]?.textContent).toBe("1");
  });

  it("形狀不合格的快照當成錯誤處理，且不覆蓋上一份好資料", async () => {
    vi.useFakeTimers();
    try {
      mocks.getSnapshotCloud.mockResolvedValueOnce(cloudOf({ snapshot: "1" }));
      await mount({ live: true });
      expect(container.querySelectorAll(".pointcloud-stats dd")[0]?.textContent).toBe("3");
      expect(container.querySelectorAll(".pointcloud-stats dd")[3]?.textContent).toBe("1");

      // 第二份快照的 point_count 與 positions 長度不一致
      mocks.getSnapshotCloud.mockResolvedValue(cloudOf({ snapshot: "2", point_count: 9, positions: [0, 0, 0] }));
      await act(async () => { await vi.advanceTimersByTimeAsync(1600); });

      const badge = container.querySelector(".pointcloud-badge");
      expect(badge).not.toBeNull();
      // live 輪詢中：顯示非阻斷的連線狀態（不會蓋住畫布）
      expect(badge?.textContent).toContain("正在重试");

      // 退避用完、停止輪詢後，才顯示形狀錯誤的 i18n 訊息與「重試」
      // （時間軸：1.5s 第一次失敗 → 5s 第二次 → 15s 第三次 → 停止）
      await act(async () => { await vi.advanceTimersByTimeAsync(1500); });
      await act(async () => { await vi.advanceTimersByTimeAsync(5000); });
      await act(async () => { await vi.advanceTimersByTimeAsync(15_000); });
      expect(container.querySelector(".pointcloud-badge")?.textContent).toContain("快照数据不完整");
      expect(container.querySelector(".pointcloud-badge button")?.textContent).toBe("重试");
      // 上一份好資料（snapshot 1、3 個點）仍然顯示
      expect(container.querySelectorAll(".pointcloud-stats dd")[0]?.textContent).toBe("3");
      expect(container.querySelectorAll(".pointcloud-stats dd")[3]?.textContent).toBe("1");
      expect(container.querySelector(".pointcloud-overlay")).toBeNull();
    } finally {
      vi.useRealTimers();
    }
  });

  it("live=false 時不建立任何 interval，live=true 才輪詢", async () => {
    await mount({ live: false });
    expect(mocks.getSnapshotCloud).toHaveBeenCalledTimes(1);
    expect(intervalSpy).not.toHaveBeenCalled();

    vi.useFakeTimers();
    try {
      await act(async () => { root.unmount(); });
      root = createRoot(container);
      await act(async () => {
        root.render(<LanguageProvider><PointCloudViewer projectPath="E:\\Projects\\示例项目" live onExit={() => undefined} /></LanguageProvider>);
      });
      expect(mocks.getSnapshotCloud).toHaveBeenCalledTimes(2);

      // 首次抓取完成後才排下一次；之後每 1.5 秒一次
      await act(async () => { await vi.advanceTimersByTimeAsync(0); });
      await act(async () => { await vi.advanceTimersByTimeAsync(1500); });
      expect(mocks.getSnapshotCloud).toHaveBeenCalledTimes(3);
      await act(async () => { await vi.advanceTimersByTimeAsync(3000); });
      expect(mocks.getSnapshotCloud).toHaveBeenCalledTimes(5);
    } finally {
      vi.useRealTimers();
    }
  });

  it("連續失敗時退避，用完後停止輪詢並保留最後一次成功的畫面", async () => {
    vi.useFakeTimers();
    try {
      mocks.getSnapshotCloud.mockResolvedValue(cloudOf());
      await mount({ live: true });
      await act(async () => { await vi.advanceTimersByTimeAsync(0); });
      const afterFirstPoll = mocks.getSnapshotCloud.mock.calls.length;

      mocks.getSnapshotCloud.mockRejectedValue(new Error("读取快照失败：磁盘暂时不可用"));
      // 第一次失敗，退避 5 秒
      await act(async () => { await vi.advanceTimersByTimeAsync(1500); });
      const afterFirstFailure = mocks.getSnapshotCloud.mock.calls.length;
      expect(afterFirstFailure).toBeGreaterThan(afterFirstPoll);
      expect(container.querySelector(".pointcloud-badge")?.textContent).toContain("正在重试");

      // 5 秒前不該再打，5 秒後才打第二次
      await act(async () => { await vi.advanceTimersByTimeAsync(4000); });
      expect(mocks.getSnapshotCloud.mock.calls.length).toBe(afterFirstFailure);
      await act(async () => { await vi.advanceTimersByTimeAsync(1100); });
      const afterSecondFailure = mocks.getSnapshotCloud.mock.calls.length;
      expect(afterSecondFailure).toBeGreaterThan(afterFirstFailure);

      // 第三次失敗，退避用完，停止輪詢並改成「重試」按鈕
      await act(async () => { await vi.advanceTimersByTimeAsync(15_100); });
      const afterStop = mocks.getSnapshotCloud.mock.calls.length;
      expect(container.querySelector(".pointcloud-badge button")?.textContent).toBe("重试");
      await act(async () => { await vi.advanceTimersByTimeAsync(60_000); });
      expect(mocks.getSnapshotCloud.mock.calls.length).toBe(afterStop);

      // 最後一次成功的畫面仍在（1 個快照、3 個點）
      expect(container.querySelectorAll(".pointcloud-stats dd")[0]?.textContent).toBe("3");
    } finally {
      vi.useRealTimers();
    }
  });

  it("卸載後不再觸碰 canvas，也不讓未完成的請求寫入 state", async () => {
    let resolveCloud: ((value: SnapshotCloud) => void) | undefined;
    mocks.getSnapshotCloud.mockImplementationOnce(() => new Promise<SnapshotCloud>((resolve) => { resolveCloud = resolve; }));
    await mount({ live: true });
    await flushFrames();

    const before = context.putImageData.mock.calls.length;
    await act(async () => { root.unmount(); });
    await flushFrames();

    await act(async () => {
      resolveCloud?.(cloudOf());
      await Promise.resolve();
      await Promise.resolve();
    });
    await flushFrames();

    expect(context.putImageData.mock.calls.length).toBe(before);
  });

  it("切換專案後忽略舊專案的回應", async () => {
    let resolveFirst: ((value: SnapshotCloud) => void) | undefined;
    mocks.getSnapshotCloud.mockImplementationOnce(() => new Promise<SnapshotCloud>((resolve) => { resolveFirst = resolve; }));
    await act(async () => {
      root.render(<LanguageProvider><PointCloudViewer projectPath="E:\\Projects\\甲" live={false} onExit={() => undefined} /></LanguageProvider>);
    });
    canvas = container.querySelector<HTMLCanvasElement>("canvas")!;
    await flushFrames();

    // 還沒回應就換到另一個專案
    mocks.getSnapshotCloud.mockResolvedValue(cloudOf({ snapshot: "乙", positions: [], colors: [], cameras: [], point_count: 0, total_points: 0, snapshot_count: 0 }));
    await act(async () => {
      root.render(<LanguageProvider><PointCloudViewer projectPath="E:\\Projects\\乙" live={false} onExit={() => undefined} /></LanguageProvider>);
    });
    await flushFrames();

    // 舊專案的回應此時才到，必須被丟棄
    await act(async () => {
      resolveFirst?.(cloudOf({ snapshot: "甲", snapshot_count: 9 }));
      await Promise.resolve();
      await Promise.resolve();
    });
    await flushFrames();

    expect(mocks.getSnapshotCloud).toHaveBeenCalledTimes(2);
    expect(container.querySelectorAll(".pointcloud-stats dd")[4]?.textContent).toBe("0");
  });

  it("resize 後依 DPR 更新 canvas 尺寸，且每種尺寸只配置一次 ImageData", async () => {
    window.devicePixelRatio = 2;
    await mount();

    // 初次量測（effect 內的 apply）在 mount 時就完成
    expect(canvas.width).toBe(600);
    expect(canvas.height).toBe(400);

    // 底部統計列換行讓 stage 變矮：不會觸發 window resize，只有 ResizeObserver 看得到
    setRectSize(300, 150);
    await act(async () => {
      observerCallback?.();
      await Promise.resolve();
    });
    await flushFrames();
    expect(canvas.width).toBe(600);
    expect(canvas.height).toBe(300);

    // 尺寸沒變時不重建 ImageData
    expect(context.createImageData).toHaveBeenCalledTimes(2);
    await act(async () => {
      observerCallback?.();
      await Promise.resolve();
    });
    await flushFrames();
    expect(context.createImageData).toHaveBeenCalledTimes(2);
  });

  it("拖曳時降到 1 倍解析度，放開後恢復並只回寫有畫到的髒矩形", async () => {
    window.devicePixelRatio = 2;
    await mount();
    expect(canvas.width).toBe(600);

    const pointer = (type: string, x: number, y: number) => {
      const event = new MouseEvent(type, { bubbles: true, cancelable: true, button: 0, clientX: x, clientY: y });
      Object.defineProperty(event, "pointerId", { value: 7 });
      return event;
    };
    await act(async () => {
      canvas.dispatchEvent(pointer("pointerdown", 100, 100));
      await Promise.resolve();
    });
    // 互動中降到 1 倍
    await flushFrames();
    expect(canvas.width).toBe(300);

    // 三次 pointermove 同一幀內只重畫一次（requestAnimationFrame 節流）
    const trackWritesBefore = context.arc.mock.calls.length;
    await act(async () => {
      canvas.dispatchEvent(pointer("pointermove", 110, 100));
      canvas.dispatchEvent(pointer("pointermove", 120, 100));
      canvas.dispatchEvent(pointer("pointermove", 130, 100));
      await Promise.resolve();
    });
    // 同一幀內不重畫：軌跡完全沒有再被畫一次
    expect(context.arc.mock.calls.length).toBe(trackWritesBefore);
    await flushFrames();
    // 一幀只畫一次：軌跡的兩個標記圓各畫一次
    expect(context.arc.mock.calls.length).toBe(trackWritesBefore + 2);

    // 回寫範圍必須遠小於整張畫布（髒矩形，不是整個 300×200）
    const lastCall = context.putImageData.mock.calls.at(-1)!;
    const [, , , , dirtyWidth, dirtyHeight] = lastCall as unknown as [unknown, number, number, number, number, number];
    expect(dirtyWidth * dirtyHeight).toBeLessThan(300 * 200);

    await act(async () => {
      canvas.dispatchEvent(pointer("pointerup", 130, 100));
      await Promise.resolve();
    });
    await flushFrames();
    // 放開後恢復高解析度
    expect(canvas.width).toBe(600);
  });

  it("滾輪縮放會 preventDefault，不讓底層預覽頁跟著捲動", async () => {
    await mount();
    const before = container.textContent;

    const wheel = new WheelEvent("wheel", { deltaY: 120, bubbles: true, cancelable: true });
    await act(async () => { canvas.dispatchEvent(wheel); });
    await flushFrames();

    expect(wheel.defaultPrevented).toBe(true);
    // 視距變了，統計列跟著更新
    expect(container.textContent).not.toBe(before);
  });

  it("鍵盤可以旋轉、縮放與重設視角，且有無障礙標籤", async () => {
    await mount();

    expect(canvas.getAttribute("role")).toBe("img");
    expect(canvas.getAttribute("aria-label")).toContain("建图点云视图");
    expect(canvas.tabIndex).toBe(0);
    expect(container.querySelector(".pointcloud-stats")?.getAttribute("aria-live")).toBe("polite");

    const distance = () => {
      const values = Array.from(container.querySelectorAll(".pointcloud-stats dd"), (node) => node.textContent ?? "");
      return values.at(-1) ?? "";
    };
    const initial = distance();

    const press = async (key: string) => {
      const event = new KeyboardEvent("keydown", { key, bubbles: true, cancelable: true });
      await act(async () => { canvas.dispatchEvent(event); });
      await flushFrames();
      return event;
    };

    const zoomIn = await press("+");
    expect(zoomIn.defaultPrevented).toBe(true);
    expect(distance()).not.toBe(initial);

    const rotate = await press("ArrowLeft");
    expect(rotate.defaultPrevented).toBe(true);
    await press("ArrowUp");
    await press("-");

    await press("Home");
    // Home 回到自動取景的距離
    expect(Number(distance())).toBeCloseTo(Number(initial), 2);

    // 無關的按鍵不搶事件
    const other = await press("a");
    expect(other.defaultPrevented).toBe(false);
  });
});
