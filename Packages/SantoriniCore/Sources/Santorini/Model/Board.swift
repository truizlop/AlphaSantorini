//
//  Board.swift
//  AlphaSantorini
//
//  Created by Tomás Ruiz-López on 1/15/26.
//

public struct Board {
    package var board: [[Building]]

    public init() {
        board = Array(
            repeating: Array(
                repeating: .height0,
                count: 5
            ),
            count: 5
        )
    }

    init(board: [[Building]]) {
        self.board = board
    }

    subscript(_ position: Position) -> Building {
        get {
            board[position.row][position.column]
        }
        set {
            board[position.row][position.column] = newValue
        }
    }

    public func building(at position: Position) -> Building {
        self[position]
    }

    mutating func build(at position: Position) {
        let building = self[position]
        self[position] = building.next
    }
}
