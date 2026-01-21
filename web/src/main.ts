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

let controller: GameController;
const scene = new BoardScene(sceneContainer, (row, col) => {
  if (controller) {
    void controller.handleTileClick(row, col);
  }
});
controller = new GameController(scene);

async function bootstrap(): Promise<void> {
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
    return;
  }

  try {
    await loadModel();
  } catch (error) {
    console.warn("ONNX model unavailable, falling back to random AI.", error);
  }

  await controller.newGame();
}

void bootstrap();
