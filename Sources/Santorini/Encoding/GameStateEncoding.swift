//
//  GameStateEncoding.swift
//  AlphaSantorini
//
//  Created by Tomás Ruiz-López on 1/15/26.
//

extension GameState {
    func encoded() -> [Int] {
        let level0 = board.level(building: .height0)
        let level1 = board.level(building: .height1)
        let level2 = board.level(building: .height2)
        let level3 = board.level(building: .height3)
        let domes = board.level(building: .dome)

        // One-hot encoding for player's workers
        var currentPlayerWorkers = Array(repeating: 0, count: 25)
        var otherPlayerWorkers = Array(repeating: 0, count: 25)
        workers.forEach { worker in
            let index = worker.position.row * 5 + worker.position.column
            if worker.player == turn {
                currentPlayerWorkers[index] = 1
            } else {
                otherPlayerWorkers[index] = 1
            }
        }

        return level0 + level1 + level2 + level3 + domes + currentPlayerWorkers + otherPlayerWorkers
    }
}

extension Board {
    // One hot encoding of the items at a given level
    fileprivate func level(building: Building) -> [Int] {
        board.flatMap { row in
            row.map { item in
                (item == building) ? 1 : 0
            }
        }
    }
}
