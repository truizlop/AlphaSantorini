//
//  Position.swift
//  AlphaSantorini
//
//  Created by Tomás Ruiz-López on 1/15/26.
//

public struct Position: Equatable {
    public private(set) var row: Int
    public private(set) var column: Int

    public init?(row: Int, column: Int) {
        guard row >= 0 && row <= 4 else { return nil }
        guard column >= 0 && column <= 4 else { return nil }
        self.row = row
        self.column = column
    }

    public func move(direction: Direction) -> Position {
        let row = max(0, min(4, self.row + direction.rowDelta))
        let column = max(0, min(4, self.column + direction.columnDelta))
        return Position(row: row, column: column)!
    }

    public func canMove(direction: Direction) -> Bool {
        self.row + direction.rowDelta >= 0 &&
        self.row + direction.rowDelta <= 4 &&
        self.column + direction.columnDelta >= 0 &&
        self.column + direction.columnDelta <= 4
    }
}
