# AlphaSantorini

Swift Santorini engine + MLX training + web UI.  
Game rules run in Swift (native and SwiftWasm), and the web UI uses ONNX Runtime Web for neural inference.

## Repository layout

- `Packages/SantoriniCore`: game rules, encoding, MCTS implementation.
- `Sources/NeuralNetwork`: MLX model (`SantoriniNet`).
- `Sources/Training`: self-play, replay buffer, trainer/evaluation pipeline.
- `Sources/AlphaSantorini`: CLI commands (`train`, `inspect`, `arena`, etc.).
- `Packages/SantoriniWasm`: SwiftWasm build used by the browser UI.
- `web/`: Vite + Three.js frontend.
- `tools/export_onnx.py`: exports MLX safetensors checkpoints to ONNX.
- `tools/verify_onnx.js`: runtime sanity check for exported ONNX.
- `experiments.md`: experiment history and rationale for current defaults.

## Current modeling decisions

### State encoding (network input)

The model input is **NHWC** `[batch, 5, 5, 9]`.

Per-cell planes (one-hot):

1. `H0`
2. `H1`
3. `H2`
4. `H3`
5. `DOME`
6. `CURRENTW1`
7. `CURRENTW2`
8. `OTHERW1`
9. `OTHERW2`

Important:

- Worker planes are **turn-relative** (`CURRENT*` vs `OTHER*`), not absolute player-color planes.
- This encoding is defined in `Packages/SantoriniCore/Sources/Santorini/Encoding/GameStateEncoding.swift`.
- In the web UI, MCTS currently reconstructs this 9-plane tensor from `getStateSummary()` to match training format.

### Action encoding (policy output)

Total policy size is **153 actions**:

- `25` placements (`5x5`)
- `128` moves (`2 workers * 8 move dirs * 8 build dirs`)

Defined in `Packages/SantoriniCore/Sources/Santorini/Encoding/ActionEncoding.swift`.

## Current network architecture

`SantoriniNet` (MLX) is a residual convolutional model:

- Input block: `Conv3x3(9->256) + BatchNorm + ReLU`
- Residual tower: **8** residual blocks (default), each:
  - `Conv3x3 + BN + ReLU`
  - `Conv3x3 + BN`
  - skip add + ReLU
- Policy head:
  - `Conv1x1(256->2) + BN + ReLU`
  - flatten `2*5*5 = 50`
  - `Linear(50->153)` (logits; softmax applied at inference/loss)
- Value head:
  - `Conv1x1(256->1) + BN + ReLU`
  - flatten `25`
  - `Linear(25->64) + ReLU`
  - `Linear(64->1) + tanh` (value in `[-1, 1]`)

Implementation: `Sources/NeuralNetwork/SantoriniNet.swift`.

## Current training pipeline and defaults

Training loop phases per iteration:

1. Self-play (batched async MCTS + batched NN eval)
2. SGD updates from replay buffer
3. Arena evaluation vs best checkpointed model
4. Checkpoint save

### Key defaults

Defaults come from `TrainingConfig` and CLI `train` options:

- `iterations`: `500`
- `gamesPerIteration`: `100`
- `mctsSimulations`: `256`
- `trainingStepsPerIteration`: `100`
- `batchSize`: `128`
- `learningRate`: `0.001`
- `replayBufferSize`: `100000`
- `symmetryAugmentation`: `true` (8x via board symmetries)
- Dirichlet noise: `epsilon=0.25`, `alpha=0.3`
- Noise anneal: to floor `0.05` across `150` iterations
- Value target strategy: `terminal` (option: `mcts`)
- Evaluation: every `10` iterations, `20` games, promotion threshold `0.55`
- Checkpoints: every `10` iterations
- Early stop: `100` iterations without promotion

Batching defaults:

- `selfPlayBatchSize`: `128`
- `selfPlayConcurrency`: `150`
- `batchTimeoutMicroseconds`: `100`

Note: self-play runs `max(gamesPerIteration, selfPlayConcurrency)` games to keep concurrency saturated. With defaults, that means 150 games even if `gamesPerIteration=100`.

### Loss details

- **Policy loss**: cross-entropy over legal-action support with masked logits:
  illegal logits are set to `-1e9` before `logSoftmax`.
- **Value loss**: MSE.

Implementation: `Sources/Training/SantoriniTrainer.swift`.

## Prerequisites

- Swift 6.2+ (native build + training)
- Python 3.9+ with `torch`, `safetensors`, `onnx` (ONNX export)
- Node 20+ (web UI)
- SwiftWasm SDK (for WASM bundle generation)

## Build and test

Build:

```sh
xcrun swift build
```

Fast core tests (rules + MCTS only):

```sh
xcrun swift test --package-path Packages/SantoriniCore
```

Full suite (core + NN + training + integration):

```sh
xcrun swift test
```

Integration test note: it runs a tiny 1-iteration training loop and writes a temporary checkpoint.

## Training and evaluation commands

Default training run:

```sh
xcrun swift run AlphaSantorini
```

Explicit training with overrides:

```sh
xcrun swift run AlphaSantorini train \
  --iterations 500 \
  --games-per-iteration 100 \
  --mcts-simulations 256
```

Resume from checkpoint:

```sh
xcrun swift run AlphaSantorini train \
  --resume-checkpoint checkpoints/checkpoint_480.safetensors
```

Inspect priors/value on a position:

```sh
xcrun swift run AlphaSantorini inspect \
  --checkpoint checkpoints/final.safetensors \
  --advance 10 \
  --mcts-simulations 200 \
  --top-k 10
```

Checkpoint-vs-checkpoint arena:

```sh
xcrun swift run AlphaSantorini arena \
  --checkpoint-a checkpoints/checkpoint_350.safetensors \
  --checkpoint-b checkpoints/checkpoint_480.safetensors \
  --games 50
```

Checkpoint vs uniform-policy baseline:

```sh
xcrun swift run AlphaSantorini arena-baseline \
  --checkpoint checkpoints/final.safetensors \
  --games 50
```

## ONNX export and verification

Export checkpoint to ONNX:

```sh
python3 tools/export_onnx.py \
  --checkpoint checkpoints/final.safetensors \
  --output web/public/models/santorini.onnx
```

The exporter auto-detects architecture (current conv net and legacy FC variants) and writes a single self-contained `.onnx` file.

Verify ONNX output:

```sh
cd web
npm install
NODE_PATH=./node_modules node ../tools/verify_onnx.js public/models/santorini.onnx
```

## Build SwiftWasm bundle

List installed SDKs first:

```sh
xcrun swift sdk list
```

Then build WASM bundle (PackageToJS):

```sh
cd Packages/SantoriniWasm
swift package --swift-sdk <wasm-sdk-id> \
  --allow-writing-to-directory ../../web/public/wasm \
  js --product SantoriniWasm --output ../../web/public/wasm
```

Outputs:

- `web/public/wasm/index.js`
- `web/public/wasm/SantoriniWasm.wasm`

If you hit toolchain linker issues with `swift`, use `xcrun swift ...`.

## Run web UI

```sh
cd web
npm install
npm run dev
```

Expected runtime assets:

- `web/public/wasm/index.js`
- `web/public/wasm/SantoriniWasm.wasm`
- `web/public/wasm-loader.js`
- `web/public/models/santorini.onnx` (optional but recommended)

If the model is missing or fails to load, the UI falls back to baseline MCTS behavior (uniform-network prior/value).

Build static site:

```sh
cd web
npm run build
```

## Makefile shortcuts

Common wrappers from repo root:

```sh
make build
make wasm
make onnx CHECKPOINT=checkpoints/final.safetensors
make web-build
make web-dev
```

## Suggested newcomer flow

1. Run core tests: `xcrun swift test --package-path Packages/SantoriniCore`
2. Run full tests: `xcrun swift test`
3. Train a short run (`--iterations 1` or `--iterations 10`) to verify your environment
4. Export one checkpoint to ONNX
5. Build WASM bundle
6. Launch web UI and verify model-backed moves
