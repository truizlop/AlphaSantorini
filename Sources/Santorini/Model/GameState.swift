//
//  GameState.swift
//  AlphaSantorini
//
//  Created by Tomás Ruiz-López on 1/15/26.
//

public struct GameState {
    var board: Board
    var turn: Player
    var workers: [Worker]
    var phase: Phase

    public init() {
        self.init(
            board: Board(),
            turn: .one,
            workers: [],
            phase: .placement
        )
    }

    public init(
        board: Board,
        turn: Player,
        workers: [Worker],
        phase: Phase
    ) {
        self.board = board
        self.turn = turn
        self.workers = workers
        self.phase = phase
    }

    public mutating func placement(_ placement: Placement) {
        guard phase == .placement,
              workers.count < 4 else {
            return
        }
        workers.append(
            Worker(
                id: workers.count < 2 ? .one : .two,
                player: turn,
                position: placement.position
            )
        )
        if workers.count == 4 {
            phase = .play
        }
        turn = turn.other
    }

    public mutating func play(move: Move) {
        guard phase == .play else { return }

        workers = workers.map { worker in
            guard worker.player == turn,
                  worker.id == move.id else {
                return worker
            }
            var newWorker = worker
            newWorker.move(direction: move.moveDirection)
            let buildPosition = newWorker.position.move(direction: move.buildDirection)
            board.build(at: buildPosition)
            turn = turn.other
            return newWorker
        }
    }

    public var isOver: Player? {
        guard phase == .play else { return nil }
        return workers.first { worker in
            board[worker.position] == .dome
        }?.player
    }

    public var legalPlacements: [Placement] {
        guard phase == .placement else { return [] }

        let placements = (0...4).flatMap { row in
            (0...4).compactMap { column in
                Position(row: row, column: column).map(Placement.init)
            }
        }
        return placements.filter { placement in
            workers.allSatisfy { worker in
                worker.position != placement.position
            }
        }
    }

    private func isWorkerOn(position: Position) -> Bool {
        workers.first { worker in
            worker.position == position
        } != nil
    }

    public var legalMoves: [Move] {
        guard phase == .play else { return [] }

        return workers.flatMap { worker -> [Move] in
            guard worker.player == turn else { return [] }

            let allowedMoveDirections = Direction.allCases.compactMap { moveDirection -> Direction? in
                guard worker.position.canMove(direction: moveDirection) else {
                    return nil
                }
                let targetPosition = worker.position.move(direction: moveDirection)
                let currentBuilding = board[worker.position]
                let targetBuilding = board[targetPosition]
                if currentBuilding.canMoveTowards(building: targetBuilding) && !isWorkerOn(position: targetPosition) {
                    return moveDirection
                } else {
                    return nil
                }
            }

            let allowedMoves = allowedMoveDirections.flatMap { moveDirection in
                let targetPosition = worker.position.move(direction: moveDirection)

                let allowedBuildDirections = Direction.allCases.compactMap { buildDirection in
                    if targetPosition.canMove(direction: buildDirection) {
                        let targetBuild = targetPosition.move(direction: buildDirection)
                        if board[targetBuild] != .dome && !isWorkerOn(position: targetBuild) {
                            return buildDirection
                        } else {
                            return nil
                        }
                    } else {
                        return nil
                    }
                }

                return allowedBuildDirections.map { buildDirection in
                    Move(
                        id: worker.id,
                        moveDirection: moveDirection,
                        buildDirection: buildDirection
                    )
                }
            }

            return allowedMoves
        }
    }
}
