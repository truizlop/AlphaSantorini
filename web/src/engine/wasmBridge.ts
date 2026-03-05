export type StateSummary = {
  phase: "placement" | "play";
  turn: "one" | "two";
  boardHeights: number[];
  workers: Array<{ player: "one" | "two"; id: "one" | "two"; row: number; col: number }>;
};

type SantoriniWasmExports = {
  createGame: () => number;
  cloneState: (handle: number) => number;
  releaseState: (handle: number) => void;
  getStateSummary: (handle: number) => StateSummary;
  legalActions: (handle: number) => number[];
  applyAction: (handle: number, action: number) => number;
  isTerminal: (handle: number) => boolean;
  winner: (handle: number) => number;
  encodeState: (handle: number) => number[];
};

declare global {
  interface Window {
    SantoriniWasmInitPromise?: Promise<() => Promise<unknown>>;
  }
}

let wasmExports: SantoriniWasmExports | null = null;
let wasmReady: Promise<void> | null = null;

async function ensureWasmLoaderRegistered(): Promise<void> {
  if (window.SantoriniWasmInitPromise) {
    return;
  }

  // Resolve relative to the page URL (not this JS chunk URL), so it works
  // with Vite base paths and GitHub Pages subpaths.
  const loaderUrl = new URL(`${import.meta.env.BASE_URL}wasm-loader.js`, window.location.href).toString();
  await import(/* @vite-ignore */ loaderUrl);
}

export async function initWasm(): Promise<void> {
  if (wasmReady) {
    return wasmReady;
  }

  wasmReady = (async () => {
    await ensureWasmLoaderRegistered();
    const initPromise = window.SantoriniWasmInitPromise;
    if (!initPromise) {
      throw new Error("SantoriniWasm init promise not found after loading wasm-loader.js.");
    }
    const init = await initPromise;
    if (typeof init !== "function") {
      throw new Error("PackageToJS init function missing.");
    }
    await init();
    const exports = (globalThis as { SantoriniWasm?: SantoriniWasmExports }).SantoriniWasm;
    if (!exports) {
      throw new Error("SantoriniWasm exports not found. Did the WASM bundle load?");
    }
    wasmExports = exports;
  })();

  return wasmReady;
}

export function wasm(): SantoriniWasmExports {
  if (!wasmExports) {
    throw new Error("WASM not initialized. Call initWasm() first.");
  }
  return wasmExports;
}
