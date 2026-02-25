//
//  GameStateEncoding.swift
//  AlphaSantorini
//
//  Created by Tomás Ruiz-López on 1/15/26.
//

// Plane index where we encode different aspects of the board
private let H0 = 0
private let H1 = 1
private let H2 = 2
private let H3 = 3
private let DOME = 4
private let CURRENTW1 = 5
private let CURRENTW2 = 6
private let OTHERW1 = 7
private let OTHERW2 = 8

extension GameState {
    public func encoded() -> [[[Float]]] {
        // One-hot encoding of different aspects of the board in 9 planes of 5x5 grids
        var encoded = Array(
            repeating: Array(
                repeating: Array<Float>(
                    repeating: 0.0,
                    count: 9
                ),
                count: 5
            ),
            count: 5
        )

        for row in 0 ..< 5 {
            for column in 0 ..< 5 {
                let building = board.board[row][column]
                switch building {
                case .height0:
                    encoded[row][column][H0] = 1.0
                case .height1:
                    encoded[row][column][H1] = 1.0
                case .height2:
                    encoded[row][column][H2] = 1.0
                case .height3:
                    encoded[row][column][H3] = 1.0
                case .dome:
                    encoded[row][column][DOME] = 1.0
                }
            }
        }

        for worker in workers {
            if worker.player == turn {
                switch worker.id {
                case .one:
                    encoded[worker.position.row][worker.position.column][CURRENTW1] = 1.0
                case .two:
                    encoded[worker.position.row][worker.position.column][CURRENTW2] = 1.0
                }
            } else {
                switch worker.id {
                case .one:
                    encoded[worker.position.row][worker.position.column][OTHERW1] = 1.0
                case .two:
                    encoded[worker.position.row][worker.position.column][OTHERW2] = 1.0
                }
            }
        }

        return encoded
    }
}
