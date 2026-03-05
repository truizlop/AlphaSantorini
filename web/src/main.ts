import "./styles.css";

import { initWasm } from "./engine/wasmBridge";
import { loadModel } from "./ai/onnx";
import { BoardScene } from "./ui/scene";
import { GameController } from "./game/controller";

const sceneContainer = document.getElementById("scene");
if (!sceneContainer) {
  throw new Error("Missing #scene container");
}

const statusEl = document.getElementById("status");
const aiStatusEl = document.getElementById("ai-status");
const loadingOverlay = document.getElementById("loading-overlay");
const loadingStage = document.getElementById("loading-stage");

let controller: GameController;
const scene = new BoardScene(sceneContainer, (row, col) => {
  if (controller) {
    void controller.handleTileClick(row, col);
  }
});
controller = new GameController(scene);

function dismissLoading(): void {
  if (loadingOverlay) {
    loadingOverlay.classList.add("fade-out");
    loadingOverlay.addEventListener("transitionend", () => {
      loadingOverlay.remove();
    }, { once: true });
  }
}

async function bootstrap(): Promise<void> {
  if (loadingStage) {
    loadingStage.textContent = "Booting engine...";
  }
  if (statusEl) {
    statusEl.textContent = "Booting Santorini engine...";
  }
  if (aiStatusEl) {
    aiStatusEl.textContent = "Loading AI model...";
  }

  try {
    await initWasm();
  } catch (error) {
    console.error(error);
    if (statusEl) {
      statusEl.textContent = "Failed to load game engine. Check console for details.";
    }
    if (aiStatusEl) {
      aiStatusEl.textContent = "AI offline";
    }
    if (loadingStage) {
      loadingStage.textContent = "Engine failed to load.";
    }
    return;
  }

  if (loadingStage) {
    loadingStage.textContent = "Loading AI...";
  }

  try {
    await loadModel();
  } catch (error) {
    console.warn("ONNX model unavailable, falling back to baseline MCTS.", error);
  }

  dismissLoading();
  await controller.newGame();
}

void bootstrap();
