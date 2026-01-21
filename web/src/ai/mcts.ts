import { decodeAction, destination } from "../game/actions";
import { wasm, StateSummary } from "../engine/wasmBridge";
import { evaluate } from "./onnx";

export type MCTSConfig = {
  iterations: number;
  explorationConstant: number;
  temperature: number;
};

type PolicyValue = { policy: Float32Array; value: number };

type ChildInfo = {
  actionId: number;
  prior: number;
  handle: number;
};

class Node {
  handle: number;
  actionId: number | null;
  parent: Node | null;
  prior: number;
  visits = 0;
  totalValue = 0;
  children: Node[] = [];
  ownsHandle: boolean;

  constructor(handle: number, prior: number, parent: Node | null, actionId: number | null, ownsHandle: boolean) {
    this.handle = handle;
    this.prior = prior;
    this.parent = parent;
    this.actionId = actionId;
    this.ownsHandle = ownsHandle;
  }

  get meanValue(): number {
    return this.visits > 0 ? this.totalValue / this.visits : 0;
  }

  isExpanded(): boolean {
    return this.children.length > 0 || wasm().isTerminal(this.handle);
  }

  puctScore(explorationConstant: number): number {
    const parentVisits = this.parent?.visits ?? 1;
    const q = this.meanValue;
    const u = explorationConstant * this.prior * Math.sqrt(parentVisits) / (1 + this.visits);
    return q + u;
  }

  bestChild(explorationConstant: number): Node | null {
    let best: Node | null = null;
    let bestScore = -Infinity;
    for (const child of this.children) {
      const score = child.puctScore(explorationConstant);
      if (score > bestScore) {
        bestScore = score;
        best = child;
      }
    }
    return best;
  }

  expand(children: ChildInfo[]): void {
    for (const child of children) {
      this.children.push(new Node(child.handle, child.prior, this, child.actionId, true));
    }
  }

  backpropagate(value: number): void {
    this.visits += 1;
    this.totalValue += value;
    if (this.parent) {
      this.parent.backpropagate(-value);
    }
  }

  dispose(): void {
    for (const child of this.children) {
      child.dispose();
    }
    if (this.ownsHandle) {
      wasm().releaseState(this.handle);
    }
  }
}

function terminalValue(handle: number, summary: StateSummary): number {
  const winner = wasm().winner(handle);
  const turnValue = summary.turn === "one" ? 1 : -1;
  if (winner === 0) {
    return 0;
  }
  return winner === turnValue ? 1 : -1;
}

async function evaluateState(handle: number): Promise<PolicyValue> {
  const encoded = wasm().encodeState(handle);
  const input = Float32Array.from(encoded);
  return evaluate(input);
}

function summarize(handle: number): StateSummary {
  return wasm().getStateSummary(handle);
}

function buildChildren(handle: number, policy: Float32Array): ChildInfo[] {
  const legal = wasm().legalActions(handle);
  const priors: number[] = [];
  for (const actionId of legal) {
    const prior = policy[actionId] ?? 0;
    priors.push(Number.isFinite(prior) ? Math.max(0, prior) : 0);
  }
  const priorSum = priors.reduce((acc, v) => acc + v, 0);
  const fallback = legal.length > 0 ? 1 / legal.length : 0;

  return legal.map((actionId, index) => {
    const nextHandle = wasm().applyAction(handle, actionId);
    const prior = priorSum > 0 ? priors[index] / priorSum : fallback;
    return { actionId, prior, handle: nextHandle };
  });
}

function temperatureDistribution(root: Node, temperature: number): Array<{ actionId: number; weight: number }> {
  const visits = root.children.map((child) => child.visits);
  if (visits.length === 0) {
    return [];
  }
  if (temperature === 0) {
    const maxVisits = Math.max(...visits);
    const count = visits.filter((v) => v === maxVisits).length || 1;
    return root.children.map((child) => ({
      actionId: child.actionId!,
      weight: child.visits === maxVisits ? 1 / count : 0,
    }));
  }
  const weights = visits.map((v) => Math.pow(v, 1 / temperature));
  const total = weights.reduce((acc, v) => acc + v, 0);
  if (total === 0) {
    return root.children.map((child) => ({ actionId: child.actionId!, weight: 1 / root.children.length }));
  }
  return root.children.map((child, index) => ({ actionId: child.actionId!, weight: weights[index] / total }));
}

function sampleAction(distribution: Array<{ actionId: number; weight: number }>): number | null {
  const total = distribution.reduce((acc, v) => acc + v.weight, 0);
  if (total === 0) {
    return distribution[0]?.actionId ?? null;
  }
  const random = Math.random();
  let cumulative = 0;
  for (const item of distribution) {
    cumulative += item.weight / total;
    if (random <= cumulative) {
      return item.actionId;
    }
  }
  return distribution[distribution.length - 1]?.actionId ?? null;
}

export async function selectAction(
  rootHandle: number,
  config: MCTSConfig
): Promise<{ actionId: number | null; distribution: Array<{ actionId: number; weight: number }> }> {
  const root = new Node(rootHandle, 0, null, null, false);
  const rootEval = await evaluateState(rootHandle);
  const rootChildren = buildChildren(rootHandle, rootEval.policy);
  root.expand(rootChildren);

  for (let i = 0; i < config.iterations; i += 1) {
    let node: Node = root;
    while (node.isExpanded() && !wasm().isTerminal(node.handle)) {
      const best = node.bestChild(config.explorationConstant);
      if (!best) {
        break;
      }
      node = best;
    }

    const summary = summarize(node.handle);
    let value: number;
    if (wasm().isTerminal(node.handle)) {
      value = terminalValue(node.handle, summary);
    } else {
      const evaluation = await evaluateState(node.handle);
      const children = buildChildren(node.handle, evaluation.policy);
      node.expand(children);
      value = evaluation.value;
    }

    node.backpropagate(value);
  }

  const distribution = temperatureDistribution(root, config.temperature);
  const actionId = sampleAction(distribution);
  root.dispose();
  return { actionId, distribution };
}

export function actionDestination(
  summary: StateSummary,
  actionId: number
): { moveRow: number; moveCol: number; buildRow: number; buildCol: number; workerId: "one" | "two" } | null {
  const decoded = decodeAction(actionId);
  if (decoded.type !== "move") {
    return null;
  }
  const worker = summary.workers.find(
    (w) => w.player === summary.turn && w.id === decoded.workerId
  );
  if (!worker) {
    return null;
  }
  const move = destination(worker.row, worker.col, decoded.moveDir);
  const build = destination(move.row, move.col, decoded.buildDir);
  return { moveRow: move.row, moveCol: move.col, buildRow: build.row, buildCol: build.col, workerId: decoded.workerId };
}
