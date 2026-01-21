# Build A GitHub Pages Santorini Web UI With SwiftWasm And ONNX AI

This ExecPlan is a living document. The sections `Progress`, `Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must be kept up to date as work proceeds.

No `PLANS.md` exists in this repository, so this plan is self-contained.

## Purpose / Big Picture

Create a browser-based Santorini experience that feels as delightful as the physical board game: a 3D board, animated workers, and clear turn-by-turn guidance. A human should be able to place workers, play a full match against an AI, and do so entirely in a static site that can be hosted on GitHub Pages. The AI should use the existing Swift game logic and a model exported from MLX to ONNX, with the web app showing visibly correct play, legal move highlighting, and a smooth setup phase followed by normal play.

## Progress

- [x] (2026-01-19 00:00Z) Authored initial ExecPlan for the Santorini web UI, SwiftWasm integration, and ONNX inference.
- [x] (2026-01-19 13:40Z) Extracted a Swift-only core package for Santorini + MCTS that can compile on native Swift and SwiftWasm.
- [ ] Implemented SwiftWasm bridge API and JS bindings; bundling with the JavaScriptKit PackageToJS plugin still required.
- [ ] Added ONNX export pipeline and verification script; requires a checkpoint to generate the model artifact.
- [x] (2026-01-19 13:40Z) Built the 3D UI, game flow, and AI loop; added GitHub Pages workflow.

## Surprises & Discoveries

- No MLX checkpoint is stored in-repo, so `web/public/models/santorini.onnx` must be generated locally before the UI can run.
- JavaScriptKit PackageToJS tooling is required to bundle the WASM assets into `web/public/wasm`.

## Decision Log

- Decision: Use SwiftWasm to run Santorini game rules in the browser and JavaScript to run ONNX inference and MCTS.
  Rationale: ONNX runtime in the browser exposes async APIs that are awkward to call from synchronous Swift MCTS; running MCTS in JavaScript keeps inference async and fast while still “consuming the Swift code” for game rules and state transitions.
  Date/Author: 2026-01-19 / Codex

- Decision: Use a Vite-based static site with Three.js for 3D rendering and deploy to GitHub Pages.
  Rationale: Vite creates a small static bundle that works well for GitHub Pages, and Three.js is widely used for stylized 3D scenes.
  Date/Author: 2026-01-19 / Codex

- Decision: Export the MLX model to ONNX via a Python script that re-creates the network and loads weights from safetensors.
  Rationale: ONNX tooling is strongest in Python, and this keeps the browser inference stack minimal and standard.
  Date/Author: 2026-01-19 / Codex

## Outcomes & Retrospective

Not started. At completion, summarize the working web UI, AI quality, and any lessons learned.

## Context and Orientation

This repository is a Swift Package Manager workspace. The relevant modules now live in `Packages/SantoriniCore/Sources/Santorini` (rules, game state, move legality), `Packages/SantoriniCore/Sources/MCTS` (tree search), `Sources/NeuralNetwork` (MLX network), and `Sources/Training` (self-play and training). The Swift game logic is pure Swift and does not require MLX, but the package’s top-level dependencies include MLX, which does not target WebAssembly. The core package isolates the game rules and MCTS so that it can be compiled to WebAssembly via SwiftWasm without pulling in MLX. The web app lives under `web/` and builds as a static site that can be served from GitHub Pages.

Definitions used below: WebAssembly (WASM) is a portable binary format that browsers can run. SwiftWasm is a Swift toolchain that compiles Swift to WASM. ONNX is a portable neural network format. onnxruntime-web is the browser runtime for ONNX models. Three.js is a JavaScript 3D rendering library.

## Interfaces and Dependencies

Web UI stack: Vite (build and dev server), Three.js (3D board), and onnxruntime-web (model inference). The web app should be a static build in `web/dist` and must use a base path that matches the GitHub Pages repo name.

SwiftWasm stack: SwiftWasm toolchain and JavaScriptKit for JS interop. The WASM module will export functions for game state creation, legal action enumeration, state cloning, state application, and state encoding.

Model interface: input is a Float32 array of length 200 (the current encoding), shaped as `[1, 200]`. Output is `policy` (Float32 length 153, already softmaxed if exporting the current model), and `value` (Float32 scalar in [-1, 1]). The JavaScript AI loop should treat `policy` as probabilities and renormalize over legal actions.

WASM bridge API (to be implemented in Swift and called from JS):

    createGame() -> Int
    cloneState(_ handle: Int) -> Int
    releaseState(_ handle: Int)
    getStateSummary(_ handle: Int) -> StateSummary
    legalActions(_ handle: Int) -> [Int]
    applyAction(_ handle: Int, _ action: Int) -> Int
    isTerminal(_ handle: Int) -> Bool
    winner(_ handle: Int) -> Int // 1 = player one, -1 = player two, 0 = draw/none
    encodeState(_ handle: Int) -> [Float]

`StateSummary` should include board heights (25 ints), worker positions (4 positions with player and worker ID), current player, and phase. It must be a plain JSON-serializable object when exposed to JS.

## Milestones

Milestone 1: Extract a WASM-safe core package. Create `Packages/SantoriniCore` with its own `Package.swift` that defines the `Santorini` and `MCTS` targets, using the existing source code moved from `Sources/Santorini` and `Sources/MCTS`. Update the root `Package.swift` to depend on `SantoriniCore` via `.package(path: "Packages/SantoriniCore")` and point existing targets at those products. Acceptance: the native Swift package still builds and the tests or training code compile without MLX-related errors. Command: run `swift build` from the repo root and expect success.

Milestone 2 (prototype): Prove SwiftWasm builds and JS bridge. Add a new `Packages/SantoriniWasm` package that depends on `SantoriniCore` and `JavaScriptKit`. Implement the bridge API above and build it to a `.wasm` and `.js` loader (via the JavaScriptKit PackageToJS plugin). Acceptance: a tiny HTML page can call `createGame()`, read the initial state, and list legal placements.

Milestone 3 (prototype): Model export and browser inference. Add a `tools/export_onnx.py` script that loads a `.safetensors` checkpoint and produces `web/public/models/santorini.onnx`. The script should recreate the network (Linear -> ReLU x3, policy head + value head with softmax/tanh) and map weights by name. In the web app, load the ONNX model with onnxruntime-web and run a single inference from a known state encoding; verify output shapes and finite numbers. Acceptance: `node tools/verify_onnx.js` prints policy length 153 and value in [-1, 1], and the browser demo prints the same.

Milestone 4: AI loop and gameplay flow. Implement a JavaScript MCTS that uses the SwiftWasm bridge for legal moves and state application, and uses ONNX for policy/value. Support two phases: placement (alternating human and AI) and play (human move then AI move). Add a difficulty setting that controls MCTS iterations. Acceptance: in the browser, the AI makes legal placements and moves, and the game ends with a winner or draw.

Milestone 5: 3D UI and polish. Build a stylized 3D board scene with Three.js: board base, stepped tower blocks, dome caps, and worker pieces. Use gentle lighting, a warm Mediterranean palette, subtle animations on hover and move, and clear highlights for legal moves/builds. Ensure the UI renders on mobile and desktop and feels “board-game like.” Acceptance: users can play a full game with an appealing presentation and clear turn feedback.

Milestone 6: GitHub Pages deployment. Configure Vite base path and add a GitHub Actions workflow that builds `web/` and publishes `web/dist` to Pages. Acceptance: a clean Pages URL loads the game and can play a full match without local tooling.

## Plan of Work

Start by moving the pure Swift game logic into a new local package to avoid MLX dependencies in WASM. Update the existing package to reference the new core package so native training still works. Next, create a SwiftWasm package that exposes a small, stable JS bridge for game state operations, and prove it works with a simple HTML harness. In parallel, create a Python export script that turns the MLX checkpoint into ONNX, and confirm onnxruntime-web can execute it. Once both prototypes pass, implement the AI loop in JavaScript, using SwiftWasm for rules and ONNX for evaluations. Build the 3D UI and map it to the game flow. Finally, wire up the build/deploy pipeline so GitHub Pages can serve the static site.

## Concrete Steps

All commands run from `/Users/tomasruizlopez/Development/AlphaSantorini` unless noted.

1. Core package extraction:
    - Move `Sources/Santorini` and `Sources/MCTS` into `Packages/SantoriniCore/Sources/`.
    - Create `Packages/SantoriniCore/Package.swift` that defines `Santorini` and `MCTS` targets.
    - Update root `Package.swift` to depend on the local core package and remove the old targets.
    - Run:
        swift build
      Expect `Build complete!` with no errors.

2. SwiftWasm bridge prototype:
    - Create `Packages/SantoriniWasm/Package.swift` with `JavaScriptKit` dependency and a `SantoriniWasm` target.
    - Add bridge functions using `JavaScriptKit` to export to JS and manage handles.
    - Build with the JavaScriptKit PackageToJS plugin:
        cd Packages/SantoriniWasm
        swift package --swift-sdk <wasm-sdk-id-from-swift-sdk-list> \
          --allow-writing-to-directory ../../web/public/wasm \
          js --product SantoriniWasm --output ../../web/public/wasm
      Expect `web/public/wasm/SantoriniWasm.wasm` and a JS loader (index.js).
    - In `web/bridge-test.html`, call `createGame()`, `getStateSummary()`, and `legalActions()` and log results.

3. ONNX export and verification:
    - Add `tools/export_onnx.py` that loads a `.safetensors` file and writes `web/public/models/santorini.onnx`.
    - Add `tools/verify_onnx.js` to run a single inference using onnxruntime-node or onnxruntime-web in Node.
    - Run:
        python3 tools/export_onnx.py --checkpoint checkpoints/best.safetensors --output web/public/models/santorini.onnx
        node tools/verify_onnx.js
      Expect logs showing `policy=153` and `value` in range.

4. Web app scaffolding:
    - Create `web/` with Vite, TypeScript, and Three.js.
    - Add `web/src/ai/onnx.ts` (load model), `web/src/ai/mcts.ts` (search), `web/src/engine/wasmBridge.ts` (SwiftWasm calls), and `web/src/ui/` (render, interactions).
    - Ensure `vite.config.ts` uses `base: "/<repo-name>/"` for GitHub Pages.

5. UI flow and AI integration:
    - Implement placement flow: human places, AI responds, repeat until four workers placed.
    - Implement move flow: human selects worker, move, build; AI responds.
    - Update the 3D scene after each action using `getStateSummary()`.
    - Add a “thinking” indicator while AI is searching.

6. Deployment:
    - Add `.github/workflows/pages.yml` to build `web/` and publish to Pages.
    - Run:
        cd web
        npm install
        npm run build
      Expect `dist/` with assets and the WASM + ONNX files included.

## Validation and Acceptance

Local validation: run `npm run dev` in `web/`, open the dev server URL, and verify:
1) The board renders with workers and towers.
2) The setup phase alternates human and AI placements.
3) The play phase enforces legal moves, highlights build options, and ends when a player wins.
4) The AI takes a move within a few seconds at default difficulty.

Model validation: run `node tools/verify_onnx.js` and ensure the policy length is 153 and the value is finite. In the browser, log the first five policy values and ensure they sum to ~1 after legal-move renormalization.

Deployment validation: after GitHub Pages publishes, open the Pages URL and play a full game.

## Idempotence and Recovery

Moving the core sources is a one-time change; keep a clean commit boundary when doing it so it can be reverted with `git checkout` if needed. The ONNX export can be re-run safely at any time. The Vite build output should always be safe to delete and regenerate. If the SwiftWasm build fails, re-run after clearing the output directory `web/public/wasm`.

## Artifacts and Notes

Example `StateSummary` payload returned to JS (shape only):

    {
      "phase": "placement",
      "turn": "one",
      "boardHeights": [0,0,0,0,0, ...],
      "workers": [
        {"player": "one", "id": "one", "row": 0, "col": 0},
        {"player": "one", "id": "two", "row": 4, "col": 4},
        {"player": "two", "id": "one", "row": 2, "col": 2},
        {"player": "two", "id": "two", "row": 3, "col": 1}
      ]
    }

Example AI request/response in JS:

    const encoded = wasm.encodeState(handle) // Float32Array length 200
    const { policy, value } = await onnx.evaluate(encoded)
    const legal = wasm.legalActions(handle) // Int array
    const normalized = renormalize(policy, legal)
