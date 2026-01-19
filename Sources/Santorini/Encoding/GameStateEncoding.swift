//
//  GameStateEncoding.swift
//  AlphaSantorini
//
//  Created by Tomás Ruiz-López on 1/15/26.
//

extension GameState {
    public func encoded() -> [Float] {
        let level0 = board.level(building: .height0)
        let level1 = board.level(building: .height1)
        let level2 = board.level(building: .height2)
        let level3 = board.level(building: .height3)
        let domes = board.level(building: .dome)

        // One-hot encoding for player's workers
        var currentPlayerWorker1 = Array<Float>(repeating: 0.0, count: 25)
        var currentPlayerWorker2 = Array<Float>(repeating: 0.0, count: 25)
        var otherPlayerWorkers = Array<Float>(repeating: 0.0, count: 25)
        workers.forEach { worker in
            let index = worker.position.row * 5 + worker.position.column
            if worker.player == turn {
                if worker.id == .one {
                    currentPlayerWorker1[index] = 1.0
                } else {
                    currentPlayerWorker2[index] = 1.0
                }
            } else {
                otherPlayerWorkers[index] = 1.0
            }
        }

        return level0 + level1 + level2 + level3 + domes + currentPlayerWorker1 + currentPlayerWorker2 + otherPlayerWorkers
    }
}

extension Board {
    // One hot encoding of the items at a given level
    fileprivate func level(building: Building) -> [Float] {
        board.flatMap { row in
            row.map { item in
                (item == building) ? 1.0 : 0.0
            }
        }
    }
}
