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
  const input = new ort.Tensor("float32", new Float32Array(200).fill(0), [1, 200]);
  const output = await session.run({ input });
  const policy = output.policy.data;
  const value = output.value.data;

  console.log(`policy length: ${policy.length}`);
  console.log(`value: ${value[0]}`);
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
