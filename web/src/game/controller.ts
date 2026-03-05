import { BoardScene, HighlightType } from "../ui/scene";
import { wasm, StateSummary } from "../engine/wasmBridge";
import { BOARD_SIZE, decodeAction, destination, tileKey, DIRECTIONS } from "./actions";
import { selectAction } from "../ai/mcts";
import { getModelReadyStatus } from "../ai/onnx";

export type MoveOption = {
  actionId: number;
  workerId: "one" | "two";
  moveDir: number;
  buildDir: number;
  moveRow: number;
  moveCol: number;
  buildRow: number;
  buildCol: number;
};

export class GameController {
  private scene: BoardScene;
  private stateHandle = 0;
  private summary: StateSummary | null = null;
  private placements: Map<string, number> = new Map();
  private moveOptions: MoveOption[] = [];
  private selectedWorker: "one" | "two" | null = null;
  private selectedMoveKey: string | null = null;
  private aiThinking = false;
  private humanPlayer: "one" | "two" = "one";

  private phaseEl: HTMLElement;
  private turnEl: HTMLElement;
  private statusEl: HTMLElement;
  private aiStatusEl: HTMLElement;
  private difficultyInput: HTMLSelectElement;
  private cancelMoveButton: HTMLButtonElement;
  private gameOverEl: HTMLElement;
  private gameOverWinnerEl: HTMLElement;
  private gameOverSubtitleEl: HTMLElement;
  private gameOverButton: HTMLButtonElement;

  constructor(scene: BoardScene) {
    this.scene = scene;
    this.phaseEl = document.getElementById("phase")!;
    this.turnEl = document.getElementById("turn")!;
    this.statusEl = document.getElementById("status")!;
    this.aiStatusEl = document.getElementById("ai-status")!;
    this.difficultyInput = document.getElementById("difficulty") as HTMLSelectElement;
    this.cancelMoveButton = document.getElementById("cancel-move") as HTMLButtonElement;
    this.gameOverEl = document.getElementById("game-over")!;
    this.gameOverWinnerEl = document.getElementById("game-over-winner")!;
    this.gameOverSubtitleEl = document.getElementById("game-over-subtitle")!;
    this.gameOverButton = document.getElementById("game-over-restart") as HTMLButtonElement;

    const newGameButton = document.getElementById("new-game")!;
    newGameButton.addEventListener("click", () => this.newGame());
    this.gameOverButton.addEventListener("click", () => this.newGame());
    this.cancelMoveButton.addEventListener("click", () => void this.cancelMoveSelection());
    const swallowPointer = (event: Event) => event.stopPropagation();
    ["pointerdown", "pointermove", "pointerup", "click"].forEach((type) => {
      this.gameOverEl.addEventListener(type, swallowPointer);
    });
  }

  async newGame(): Promise<void> {
    if (this.stateHandle) {
      wasm().releaseState(this.stateHandle);
    }
    this.stateHandle = wasm().createGame();
    this.selectedWorker = null;
    this.selectedMoveKey = null;
    this.setAiReadyStatus();
    this.hideGameOver();
    this.updateCancelState();
    await this.refresh();
  }

  async handleTileClick(row: number, col: number): Promise<void> {
    if (!this.summary || this.aiThinking) {
      return;
    }
    if (wasm().isTerminal(this.stateHandle)) {
      return;
    }
    if (this.summary.turn !== this.humanPlayer) {
      return;
    }

    if (this.summary.phase === "placement") {
      const placementKey = tileKey(row, col);
      const actionId = this.placements.get(placementKey);
      if (actionId === undefined) {
        return;
      }
      this.applyAction(actionId);
      return;
    }

    if (!this.selectedWorker) {
      await this.selectWorkerAt(row, col);
      return;
    }

    if (!this.selectedMoveKey) {
      const moveKey = tileKey(row, col);
      const moveOption = this.moveOptions.find(
        (option) =>
          option.workerId === this.selectedWorker &&
          option.moveRow === row &&
          option.moveCol === col
      );
      if (!moveOption) {
        await this.selectWorkerAt(row, col);
        return;
      }
      if (this.isWinningMove(moveOption)) {
        this.applyAction(moveOption.actionId);
        return;
      }
      this.selectedMoveKey = moveKey;
      await this.refreshHighlights();
      return;
    }

    const buildOption = this.moveOptions.find(
      (option) =>
        option.workerId === this.selectedWorker &&
        option.moveRow === Number(this.selectedMoveKey?.split("-")[0]) &&
        option.moveCol === Number(this.selectedMoveKey?.split("-")[1]) &&
        option.buildRow === row &&
        option.buildCol === col
    );

    if (buildOption) {
      this.applyAction(buildOption.actionId);
    }
  }

  private applyAction(actionId: number): void {
    const newHandle = wasm().applyAction(this.stateHandle, actionId);
    if (newHandle < 0) {
      return;
    }
    wasm().releaseState(this.stateHandle);
    this.stateHandle = newHandle;
    this.selectedWorker = null;
    this.selectedMoveKey = null;
    void this.refresh();
  }

  private async applyActionAnimated(actionId: number): Promise<void> {
    if (!this.summary) {
      return;
    }
    const prevSummary = this.summary;
    const newHandle = wasm().applyAction(this.stateHandle, actionId);
    if (newHandle < 0) {
      return;
    }
    const nextSummary = wasm().getStateSummary(newHandle);
    await this.scene.animateAction(prevSummary, actionId, nextSummary);
    wasm().releaseState(this.stateHandle);
    this.stateHandle = newHandle;
    this.selectedWorker = null;
    this.selectedMoveKey = null;
    await this.refresh();
  }

  private async refresh(): Promise<void> {
    this.summary = wasm().getStateSummary(this.stateHandle);
    this.rebuildLegalMaps();
    await this.refreshHighlights();
    this.updateStatus();

    if (wasm().isTerminal(this.stateHandle)) {
      const winner = wasm().winner(this.stateHandle);
      if (winner === 1) {
        this.statusEl.textContent = "Player One wins!";
      } else if (winner === -1) {
        this.statusEl.textContent = "Player Two wins!";
      } else {
        this.statusEl.textContent = "Draw.";
      }
      this.showGameOver(winner);
      return;
    }

    this.hideGameOver();

    if (this.summary.turn !== this.humanPlayer) {
      // Double-rAF: first frame renders the scene, second ensures paint is
      // flushed to screen before the AI computation blocks the main thread.
      await new Promise<void>((r) =>
        requestAnimationFrame(() => requestAnimationFrame(() => r()))
      );
      await this.handleAiTurn();
    }
  }

  private async refreshHighlights(): Promise<void> {
    if (!this.summary) {
      return;
    }
    const highlights = new Map<string, HighlightType>();
    if (this.summary.phase === "placement") {
      this.placements.forEach((_action, key) => highlights.set(key, "placement"));
    } else if (this.selectedWorker && !this.selectedMoveKey) {
      this.moveOptions
        .filter((option) => option.workerId === this.selectedWorker)
        .forEach((option) => highlights.set(tileKey(option.moveRow, option.moveCol), "move"));
    } else if (this.selectedWorker && this.selectedMoveKey) {
      const [moveRow, moveCol] = this.selectedMoveKey.split("-").map(Number);
      this.moveOptions
        .filter((option) => option.workerId === this.selectedWorker && option.moveRow === moveRow && option.moveCol === moveCol)
        .forEach((option) => highlights.set(tileKey(option.buildRow, option.buildCol), "build"));
    }

    const selectedWorkerKey = this.selectedWorker ? `${this.humanPlayer}-${this.selectedWorker}` : null;
    this.scene.update(this.summary, highlights, selectedWorkerKey);
  }

  private updateStatus(): void {
    if (!this.summary) {
      return;
    }
    this.phaseEl.textContent = this.summary.phase === "placement" ? "Placement" : "Play";
    this.turnEl.textContent = this.summary.turn === "one" ? "Player One" : "Player Two";

    // Color-code turn indicator
    this.turnEl.classList.remove("turn-one", "turn-two", "human-turn");
    this.turnEl.classList.add(this.summary.turn === "one" ? "turn-one" : "turn-two");
    if (this.summary.turn === this.humanPlayer) {
      this.turnEl.classList.add("human-turn");
    }

    if (this.summary.turn !== this.humanPlayer) {
    const modelStatus = getModelReadyStatus();
    this.statusEl.textContent = modelStatus.ready
      ? "AI is making a move..."
      : "Baseline MCTS is making a move...";
    } else if (this.summary.phase === "placement") {
      this.statusEl.textContent = "Select a tile to place your worker.";
    } else if (!this.selectedWorker) {
      this.statusEl.textContent = "Select one of your workers.";
    } else if (!this.selectedMoveKey) {
      this.statusEl.textContent = "Choose a move destination.";
    } else {
      this.statusEl.textContent = "Choose where to build.";
    }
    this.updateCancelState();
  }

  private rebuildLegalMaps(): void {
    this.placements.clear();
    this.moveOptions = [];
    const legal = wasm().legalActions(this.stateHandle);
    if (!this.summary) {
      return;
    }

    for (const actionId of legal) {
      const decoded = decodeAction(actionId);
      if (decoded.type === "placement") {
        this.placements.set(tileKey(decoded.row, decoded.col), actionId);
      } else {
        const worker = this.summary.workers.find(
          (w) => w.player === this.summary!.turn && w.id === decoded.workerId
        );
        if (!worker) {
          continue;
        }
        const move = destination(worker.row, worker.col, decoded.moveDir);
        const build = destination(move.row, move.col, decoded.buildDir);
        this.moveOptions.push({
          actionId,
          workerId: decoded.workerId,
          moveDir: decoded.moveDir,
          buildDir: decoded.buildDir,
          moveRow: move.row,
          moveCol: move.col,
          buildRow: build.row,
          buildCol: build.col,
        });
      }
    }
  }

  private async handleAiTurn(): Promise<void> {
    if (!this.summary) {
      return;
    }
    this.aiThinking = true;
    const modelStatus = getModelReadyStatus();
    const modelReady = modelStatus.ready;
    this.aiStatusEl.textContent = modelReady ? "AI thinking..." : modelStatus.reason;
    this.aiStatusEl.classList.add("thinking");

    const difficulty = this.getDifficultyConfig();
    let actionId: number | null = null;
    try {
      const result = await selectAction(this.stateHandle, {
        iterations: difficulty.iterations,
        explorationConstant: 1.5,
        temperature: difficulty.temperature,
      });
      actionId = result.actionId;
      this.logAiConsiderations(result.distribution, result.rootValue);
    } catch (error) {
      console.error(error);
      this.statusEl.textContent = "AI failed to respond. Falling back to random moves.";
      this.aiStatusEl.textContent = "Random AI";
      actionId = this.randomLegalAction();
    }

    if (actionId !== null) {
      this.aiStatusEl.textContent = "AI moving...";
      await this.applyActionAnimated(actionId);
    }

    this.aiThinking = false;
    this.aiStatusEl.classList.remove("thinking");
    this.setAiReadyStatus();
  }

  private logAiConsiderations(
    distribution: Array<{ actionId: number; weight: number }>,
    value: number
  ): void {
    if (!distribution.length) {
      console.log(`AI move: no distribution (value=${value.toFixed(3)})`);
      return;
    }
    const top = [...distribution]
      .sort((a, b) => b.weight - a.weight)
      .slice(0, 3)
      .map((entry) => ({
        ...entry,
        label: this.describeAction(entry.actionId),
      }));
    const parts = top.map(
      (entry) => `${entry.label} p=${entry.weight.toFixed(3)}`
    );
    console.log(`AI move (value=${value.toFixed(3)}): ${parts.join(" | ")}`);
  }

  private describeAction(actionId: number): string {
    const decoded = decodeAction(actionId);
    if (decoded.type === "placement") {
      return `Place(${decoded.row},${decoded.col})`;
    }
    const moveDir = DIRECTIONS[decoded.moveDir]?.name ?? `${decoded.moveDir}`;
    const buildDir = DIRECTIONS[decoded.buildDir]?.name ?? `${decoded.buildDir}`;
    const worker = decoded.workerId === "one" ? "W1" : "W2";
    return `Move(${worker} ${moveDir}->${buildDir})`;
  }

  private randomLegalAction(): number | null {
    const legal = wasm().legalActions(this.stateHandle);
    if (legal.length === 0) {
      return null;
    }
    const index = Math.floor(Math.random() * legal.length);
    return legal[index] ?? null;
  }

  private setAiReadyStatus(): void {
    const modelStatus = getModelReadyStatus();
    this.aiStatusEl.textContent = modelStatus.ready ? "AI ready" : modelStatus.reason;
    this.aiStatusEl.classList.remove("thinking");
    this.updateCancelState();
  }

  private async cancelMoveSelection(): Promise<void> {
    if (!this.selectedMoveKey) {
      return;
    }
    this.selectedMoveKey = null;
    await this.refreshHighlights();
    this.updateStatus();
  }

  private updateCancelState(): void {
    const canCancel =
      !!this.summary &&
      this.summary.turn === this.humanPlayer &&
      this.summary.phase === "play" &&
      !this.aiThinking &&
      !wasm().isTerminal(this.stateHandle) &&
      !!this.selectedMoveKey;
    this.cancelMoveButton.disabled = !canCancel;
  }

  private async selectWorkerAt(row: number, col: number): Promise<boolean> {
    if (!this.summary) {
      return false;
    }
    const worker = this.summary.workers.find(
      (w) => w.player === this.humanPlayer && w.row === row && w.col === col
    );
    if (!worker) {
      return false;
    }
    this.selectedWorker = worker.id;
    this.selectedMoveKey = null;
    await this.refreshHighlights();
    return true;
  }

  private isWinningMove(option: MoveOption): boolean {
    if (!this.summary) {
      return false;
    }
    const height = this.summary.boardHeights[option.moveRow * BOARD_SIZE + option.moveCol] ?? 0;
    return height === 3;
  }

  private showGameOver(winner: number): void {
    const winnerClasses = ["winner-one", "winner-two", "winner-draw"];
    this.gameOverEl.classList.remove(...winnerClasses);
    this.gameOverEl.classList.remove("visible");

    if (winner === 1) {
      this.gameOverWinnerEl.textContent = "Player One Wins!";
      this.gameOverSubtitleEl.textContent = "The blue builders claim the summit.";
      this.gameOverEl.classList.add("winner-one");
    } else if (winner === -1) {
      this.gameOverWinnerEl.textContent = "Player Two Wins!";
      this.gameOverSubtitleEl.textContent = "The gray builders reach the heights.";
      this.gameOverEl.classList.add("winner-two");
    } else {
      this.gameOverWinnerEl.textContent = "Draw Game";
      this.gameOverSubtitleEl.textContent = "No one reaches the summit this time.";
      this.gameOverEl.classList.add("winner-draw");
    }

    void this.gameOverEl.offsetWidth;
    this.gameOverEl.classList.add("visible");
    this.aiStatusEl.textContent = "Game over";
    this.aiStatusEl.classList.remove("thinking");
  }

  private hideGameOver(): void {
    this.gameOverEl.classList.remove("visible", "winner-one", "winner-two", "winner-draw");
  }

  private getDifficultyConfig(): { iterations: number; temperature: number } {
    switch (this.difficultyInput.value) {
      case "easy":
        return { iterations: 128, temperature: 0.8 };
      case "hard":
        return { iterations: 512, temperature: 0.2 };
      case "extreme":
        return { iterations: 2048, temperature: 0 };
      case "medium":
      default:
        return { iterations: 256, temperature: 0.4 };
    }
  }
}
