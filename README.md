# AlphaSantorini

Swift Santorini engine + MLX training + web UI that runs the game rules in SwiftWasm and the AI in ONNX.

## Prerequisites

- Swift 6.2+ (native build + training)
- Python 3.9+ with `torch` and `safetensors` (ONNX export)
- Node 20+ (web UI)
- SwiftWasm SDK (e.g. `...-wasm32-unknown-wasip1-threads`) for the WASM bundle

Optional:
- `onnxruntime-node` (for ONNX verification script)

## Native build

```sh
swift build
```

## Build the SwiftWasm bundle (JavaScriptKit PackageToJS)

From the repo root, first confirm the WASM SDK is installed:

```sh
swift sdk list
```

If you don't see a WASM SDK in the list, install a SwiftWasm SDK/toolchain (see https://book.swiftwasm.org/getting-started/setup.html).

Then run the PackageToJS plugin inside the WASM package:

```sh
cd Packages/SantoriniWasm
swift package --swift-sdk <wasm-sdk-id-from-swift-sdk-list> \
  --allow-writing-to-directory ../../web/public/wasm \
  js --product SantoriniWasm --output ../../web/public/wasm
```

This should produce:
- `web/public/wasm/index.js`
- `web/public/wasm/SantoriniWasm.wasm`

## Export the MLX model to ONNX

```sh
python3 tools/export_onnx.py \
  --checkpoint <path/to/checkpoint.safetensors> \
  --output web/public/models/santorini.onnx
```

If your checkpoint uses a non-200 input size, pass `--input-dim <N>`.

Optional verification (requires `onnxruntime-node` installed in this repo):

```sh
npm install --save-dev onnxruntime-node
node tools/verify_onnx.js web/public/models/santorini.onnx
```

## Run the web UI (local dev)

```sh
cd web
npm install
npm run dev
```

The UI expects:
- `web/public/wasm/index.js` and `web/public/wasm/SantoriniWasm.wasm`
- `web/public/wasm-loader.js` (bootstrap loader)
- `web/public/models/santorini.onnx`

If the ONNX model is missing, the UI will fall back to a random AI player.

## Build the web UI (static)

```sh
cd web
npm run build
```

The static output is written to `web/dist`, and the GitHub Pages workflow publishes that directory.
