import * as ort from "onnxruntime-web";

let session: ort.InferenceSession | null = null;

export function isModelReady(): boolean {
  return session !== null;
}

export async function loadModel(): Promise<void> {
  if (session) {
    return;
  }

  const wasmPath = "https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/";
  ort.env.wasm.wasmPaths = wasmPath;
  const threaded =
    typeof crossOriginIsolated === "boolean" ? crossOriginIsolated : false;
  ort.env.wasm.numThreads = threaded
    ? Math.max(1, Math.min(4, navigator.hardwareConcurrency || 2))
    : 1;

  const modelUrl = `${import.meta.env.BASE_URL}models/santorini.onnx`;
  const modelAvailable = await hasModelFile(modelUrl);
  if (!modelAvailable) {
    return;
  }
  session = await ort.InferenceSession.create(modelUrl, {
    executionProviders: ["wasm"],
    graphOptimizationLevel: "all",
  });
}

async function hasModelFile(url: string): Promise<boolean> {
  try {
    const response = await fetch(url, { method: "HEAD" });
    return response.ok;
  } catch {
    return false;
  }
}

export async function evaluate(encoded: Float32Array): Promise<{ policy: Float32Array; value: number }> {
  if (!session) {
    throw new Error("ONNX session not initialized. Call loadModel() first.");
  }
  const inputTensor = new ort.Tensor("float32", encoded, [1, encoded.length]);
  const output = await session.run({ input: inputTensor });
  const policy = output.policy.data as Float32Array;
  const valueArray = output.value.data as Float32Array;
  return { policy, value: valueArray[0] ?? 0 };
}
