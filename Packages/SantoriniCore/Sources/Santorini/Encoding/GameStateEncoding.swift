//
//  GameStateEncoding.swift
//  AlphaSantorini
//
//  Created by Tomás Ruiz-López on 1/15/26.
//

extension GameState {
    public func encoded() -> [Float] {
        var encoded = Array<Float>(repeating: 0.0, count: 200)
        var workerFlags = Array<Int>(repeating: 0, count: 25)

        for worker in workers {
            let index = worker.position.row * 5 + worker.position.column
            if worker.player == turn {
                workerFlags[index] = worker.id == .one ? 1 : 2
            } else {
                workerFlags[index] = 3
            }
        }

        for row in 0 ..< 5 {
            for column in 0 ..< 5 {
                let index = row * 5 + column
                let building = board.board[row][column]
                switch building {
                case .height0:
                    encoded[index] = 1.0
                case .height1:
                    encoded[25 + index] = 1.0
                case .height2:
                    encoded[50 + index] = 1.0
                case .height3:
                    encoded[75 + index] = 1.0
                case .dome:
                    encoded[100 + index] = 1.0
                }

                switch workerFlags[index] {
                case 1:
                    encoded[125 + index] = 1.0
                case 2:
                    encoded[150 + index] = 1.0
                case 3:
                    encoded[175 + index] = 1.0
                default:
                    break
                }
            }
        }

        return encoded
    }
}
