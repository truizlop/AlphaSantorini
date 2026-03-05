#!/usr/bin/env node
const fs = require("fs");
const path = require("path");

async function main() {
  let ort;
  try {
    ort = require("onnxruntime-node");
  } catch (error) {
    console.error("Missing dependency: onnxruntime-node. Install with: npm install --save-dev onnxruntime-node");
    process.exit(1);
  }

  const modelPath = process.argv[2]
    ? path.resolve(process.argv[2])
    : path.resolve(__dirname, "..", "web", "public", "models", "santorini.onnx");

  if (!fs.existsSync(modelPath)) {
    console.error(`Model not found at ${modelPath}`);
    process.exit(1);
  }

  const session = await ort.InferenceSession.create(modelPath, { executionProviders: ["cpu"] });

  // Auto-detect input shape from model metadata
  const inputMeta = session.inputNames.map((name) => ({
    name,
    dims: session.inputMetadata?.[name]?.dimensions,
  }));
  console.log("Input metadata:", JSON.stringify(inputMeta));

  // Determine input shape: conv model uses [1, 5, 5, 9] (225 floats),
  // legacy FC models use [1, N] (e.g. 200 floats).
  let inputSize = 225;
  let inputShape = [1, 5, 5, 9];
  try {
    const dims = session.inputMetadata?.["input"]?.dimensions;
    if (dims && dims.length === 2) {
      // Legacy FC model [batch, N]
      inputSize = dims[1];
      inputShape = [1, inputSize];
    }
  } catch {
    // Fall back to conv default
  }

  console.log(`Using input shape: [${inputShape}] (${inputSize} floats)`);
  const input = new ort.Tensor("float32", new Float32Array(inputSize).fill(0), inputShape);
  const output = await session.run({ input });
  const policy = output.policy.data;
  const value = output.value.data;

  console.log(`policy length: ${policy.length}`);
  console.log(`policy sum: ${Array.from(policy).reduce((a, b) => a + b, 0).toFixed(4)}`);
  console.log(`value: ${value[0]}`);

  // Basic sanity checks
  if (policy.length !== 153) {
    console.error(`ERROR: Expected 153 policy outputs, got ${policy.length}`);
    process.exit(1);
  }
  console.log("PASS: Model verification succeeded");
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
