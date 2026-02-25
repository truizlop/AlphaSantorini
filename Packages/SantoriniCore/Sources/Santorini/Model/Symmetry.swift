//
//  Symmetry.swift
//  AlphaSantorini
//
//  Created by Codex on 2/25/26.
//

public enum BoardSymmetry: CaseIterable, Sendable {
    case identity
    case rotate90
    case rotate180
    case rotate270
    case mirrorVertical
    case mirrorHorizontal
    case mirrorMainDiagonal
    case mirrorAntiDiagonal

    public var inverse: BoardSymmetry {
        switch self {
        case .identity:
            .identity
        case .rotate90:
            .rotate270
        case .rotate180:
            .rotate180
        case .rotate270:
            .rotate90
        case .mirrorVertical:
            .mirrorVertical
        case .mirrorHorizontal:
            .mirrorHorizontal
        case .mirrorMainDiagonal:
            .mirrorMainDiagonal
        case .mirrorAntiDiagonal:
            .mirrorAntiDiagonal
        }
    }

    fileprivate func transform(row: Int, column: Int) -> (row: Int, column: Int) {
        switch self {
        case .identity:
            return (row, column)
        case .rotate90:
            return (column, 4 - row)
        case .rotate180:
            return (4 - row, 4 - column)
        case .rotate270:
            return (4 - column, row)
        case .mirrorVertical:
            return (row, 4 - column)
        case .mirrorHorizontal:
            return (4 - row, column)
        case .mirrorMainDiagonal:
            return (column, row)
        case .mirrorAntiDiagonal:
            return (4 - column, 4 - row)
        }
    }

    fileprivate func transform(deltaRow: Int, deltaColumn: Int) -> (deltaRow: Int, deltaColumn: Int) {
        switch self {
        case .identity:
            return (deltaRow, deltaColumn)
        case .rotate90:
            return (deltaColumn, -deltaRow)
        case .rotate180:
            return (-deltaRow, -deltaColumn)
        case .rotate270:
            return (-deltaColumn, deltaRow)
        case .mirrorVertical:
            return (deltaRow, -deltaColumn)
        case .mirrorHorizontal:
            return (-deltaRow, deltaColumn)
        case .mirrorMainDiagonal:
            return (deltaColumn, deltaRow)
        case .mirrorAntiDiagonal:
            return (-deltaColumn, -deltaRow)
        }
    }
}

extension Position {
    public func transformed(by symmetry: BoardSymmetry) -> Position {
        let transformed = symmetry.transform(row: row, column: column)
        return Position(row: transformed.row, column: transformed.column)!
    }
}

extension Direction {
    private static func from(deltaRow: Int, deltaColumn: Int) -> Direction? {
        allCases.first {
            $0.rowDelta == deltaRow && $0.columnDelta == deltaColumn
        }
    }

    public func transformed(by symmetry: BoardSymmetry) -> Direction {
        let transformed = symmetry.transform(deltaRow: rowDelta, deltaColumn: columnDelta)
        return Direction.from(deltaRow: transformed.deltaRow, deltaColumn: transformed.deltaColumn)!
    }
}

extension Placement {
    public func transformed(by symmetry: BoardSymmetry) -> Placement {
        Placement(position: position.transformed(by: symmetry))
    }
}

extension Move {
    public func transformed(by symmetry: BoardSymmetry) -> Move {
        Move(
            id: id,
            moveDirection: moveDirection.transformed(by: symmetry),
            buildDirection: buildDirection.transformed(by: symmetry)
        )
    }
}

extension Action {
    public func transformed(by symmetry: BoardSymmetry) -> Action {
        switch self {
        case .placement(let placement):
            .placement(placement.transformed(by: symmetry))
        case .move(let move):
            .move(move.transformed(by: symmetry))
        }
    }
}

extension GameState {
    public func transformed(by symmetry: BoardSymmetry) -> GameState {
        var transformedBoard = Array(
            repeating: Array(repeating: Building.height0, count: 5),
            count: 5
        )
        for row in 0 ..< 5 {
            for column in 0 ..< 5 {
                let transformed = symmetry.transform(row: row, column: column)
                transformedBoard[transformed.row][transformed.column] = board.board[row][column]
            }
        }

        let transformedWorkers = workers.map { worker in
            Worker(
                id: worker.id,
                player: worker.player,
                position: worker.position.transformed(by: symmetry)
            )
        }

        return GameState(
            board: Board(board: transformedBoard),
            turn: turn,
            workers: transformedWorkers,
            phase: phase
        )
    }
}
