export const BOARD_SIZE = 5;
export const PLACEMENT_TOTAL = BOARD_SIZE * BOARD_SIZE;
export const MOVE_TOTAL = 2 * 8 * 8;
export const ACTION_TOTAL = PLACEMENT_TOTAL + MOVE_TOTAL;

export type Direction = {
  id: number;
  name: string;
  dr: number;
  dc: number;
};

export const DIRECTIONS: Direction[] = [
  { id: 0, name: "nw", dr: -1, dc: -1 },
  { id: 1, name: "n", dr: -1, dc: 0 },
  { id: 2, name: "ne", dr: -1, dc: 1 },
  { id: 3, name: "w", dr: 0, dc: -1 },
  { id: 4, name: "e", dr: 0, dc: 1 },
  { id: 5, name: "sw", dr: 1, dc: -1 },
  { id: 6, name: "s", dr: 1, dc: 0 },
  { id: 7, name: "se", dr: 1, dc: 1 },
];

export type PlacementAction = {
  type: "placement";
  id: number;
  row: number;
  col: number;
};

export type MoveAction = {
  type: "move";
  id: number;
  workerId: "one" | "two";
  moveDir: number;
  buildDir: number;
};

export type DecodedAction = PlacementAction | MoveAction;

export function decodeAction(id: number): DecodedAction {
  if (id < PLACEMENT_TOTAL) {
    return {
      type: "placement",
      id,
      row: Math.floor(id / BOARD_SIZE),
      col: id % BOARD_SIZE,
    };
  }

  const moveEncoding = id - PLACEMENT_TOTAL;
  const workerId = moveEncoding < 64 ? "one" : "two";
  const local = moveEncoding % 64;
  const moveDir = Math.floor(local / 8);
  const buildDir = local % 8;

  return {
    type: "move",
    id,
    workerId,
    moveDir,
    buildDir,
  };
}

export function destination(row: number, col: number, dirId: number): { row: number; col: number } {
  const dir = DIRECTIONS[dirId];
  return { row: row + dir.dr, col: col + dir.dc };
}

export function tileKey(row: number, col: number): string {
  return `${row}-${col}`;
}
