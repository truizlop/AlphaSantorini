import * as ort from "onnxruntime-web";
import { ACTION_TOTAL } from "../game/actions";

let session: ort.InferenceSession | null = null;

export function isModelReady(): boolean {
  return session !== null;
}

export function getModelReadyStatus(): { ready: boolean; reason: string } {
  if (session) {
    return { ready: true, reason: "Model ready" };
  }
  return { ready: false, reason: "Baseline MCTS (uniform policy)" };
}

export async function loadModel(): Promise<void> {
  if (session) {
    return;
  }

  const wasmBase = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.23.2/dist/";
  ort.env.wasm.wasmPaths = {
    mjs: `${wasmBase}ort-wasm-simd-threaded.mjs`,
    wasm: `${wasmBase}ort-wasm-simd-threaded.wasm`,
  };
  ort.env.wasm.numThreads = 1;

  const modelUrl = `${import.meta.env.BASE_URL}models/santorini.onnx`;
  const modelAvailable = await hasModelFile(modelUrl);
  if (!modelAvailable) {
    console.warn(`ONNX model not found at ${modelUrl}`);
    return;
  }
  console.log("[ONNX] Creating inference session...");
  session = await ort.InferenceSession.create(modelUrl, {
    executionProviders: ["wasm"],
    graphOptimizationLevel: "all",
  });
  console.log("[ONNX] Session created successfully. Input names:", session.inputNames, "Output names:", session.outputNames);
}

async function hasModelFile(url: string): Promise<boolean> {
  try {
    const response = await fetch(url, { method: "HEAD" });
    if (response.ok) {
      return true;
    }
  } catch {
    // fall through
  }
  try {
    const response = await fetch(url, {
      method: "GET",
      headers: { Range: "bytes=0-0" },
    });
    return response.ok;
  } catch {
    return false;
  }
}

let evalCount = 0;

export async function evaluate(encoded: Float32Array): Promise<{ policy: Float32Array; value: number }> {
  if (!session) {
    if (evalCount === 0) {
      console.warn("[ONNX] evaluate called but session is NULL — using uniform fallback");
    }
    evalCount++;
    return { policy: new Float32Array(ACTION_TOTAL), value: 0 };
  }
  const inputTensor = new ort.Tensor("float32", encoded, [1, 5, 5, 9]);
  const output = await session.run({ input: inputTensor });
  const policy = output.policy.data as Float32Array;
  const valueArray = output.value.data as Float32Array;
  if (evalCount < 3) {
    const topIdx = Array.from(policy).reduce((best, v, i, arr) => v > arr[best] ? i : best, 0);
    console.log(`[ONNX] eval #${evalCount}: value=${valueArray[0]?.toFixed(4)}, topAction=${topIdx} (${policy[topIdx]?.toFixed(4)}), policyMax=${Math.max(...policy).toFixed(4)}`);
  }
  evalCount++;
  return { policy, value: valueArray[0] ?? 0 };
}
