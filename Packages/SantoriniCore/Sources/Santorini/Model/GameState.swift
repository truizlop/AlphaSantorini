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

        var result: [Move] = []
        result.reserveCapacity(64)

        for worker in workers {
            guard worker.player == turn else { continue }
            let origin = worker.position
            let currentBuilding = board[origin]

            for moveDirection in Direction.allCases {
                guard origin.canMove(direction: moveDirection) else { continue }

                let destination = origin.move(direction: moveDirection)
                guard !isWorkerOn(position: destination) else { continue }

                let destinationBuilding = board[destination]
                guard currentBuilding.canMoveTowards(building: destinationBuilding) else { continue }

                for buildDirection in Direction.allCases {
                    guard destination.canMove(direction: buildDirection) else { continue }

                    let buildPosition = destination.move(direction: buildDirection)
                    guard board[buildPosition] != .dome else { continue }

                    var occupied = false
                    for other in workers {
                        if other.player == worker.player && other.id == worker.id {
                            continue
                        }
                        if other.position == buildPosition {
                            occupied = true
                            break
                        }
                    }
                    guard !occupied else { continue }

                    result.append(
                        Move(
                            id: worker.id,
                            moveDirection: moveDirection,
                            buildDirection: buildDirection
                        )
                    )
                }
            }
        }

        return result
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
