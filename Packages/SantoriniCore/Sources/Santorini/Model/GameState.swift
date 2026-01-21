//
//  GameState.swift
//  AlphaSantorini
//
//  Created by Tomás Ruiz-López on 1/15/26.
//

public struct GameState {
    public var board: Board
    public var turn: Player
    public var workers: [Worker]
    public var phase: Phase

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
            return newWorker
        }
        turn = turn.other
    }

    public var isOver: Bool {
        winner != nil
    }

    public var winner: Player? {
        guard phase == .play else { return nil }
        if let worker = workers.first(where: { worker in board[worker.position] == .height3 }) {
            return worker.player
        }
        if legalActions.isEmpty {
            return turn.other
        }
        return nil
    }

    public var legalActions: [Action] {
        legalPlacements.map(Action.placement) + legalMoves.map(Action.move)
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

        func isOccupiedAfterMove(
            position: Position,
            movingWorker: Worker,
            destination: Position
        ) -> Bool {
            if position == destination {
                return true
            }
            return workers.contains { worker in
                if worker.player == movingWorker.player && worker.id == movingWorker.id {
                    return false
                }
                return worker.position == position
            }
        }

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
                        if board[targetBuild] != .dome &&
                            !isOccupiedAfterMove(
                                position: targetBuild,
                                movingWorker: worker,
                                destination: targetPosition
                            ) {
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

    public func show() {
        for row in 0 ..< board.board.count {
            print("")
            let workersInRow = workers
                .filter { $0.position.row == row }
            for col in 0 ..< board.board[row].count {
                if let worker = workersInRow.first(where: { $0.position.column == col }) {
                    let letter = switch worker.id {
                    case .one: worker.player == .one ? "A" : "a"
                    case .two: worker.player == .one ? "B" : "b"
                    }
                    print("\(letter) ", terminator: "")
                } else {
                    print("  ", terminator: "")
                }
            }
            print("")

            for col in 0 ..< board.board[row].count {
                let building = board.board[row][col]
                let buildingStr = switch building {
                    case .height0: "0"
                    case .height1: "1"
                    case .height2: "2"
                    case .height3: "3"
                    case .dome: "4"
                }
                print("\(buildingStr) ", terminator: "")
            }
        }
    }
}
