//
//  TrainingConfig.swift
//  AlphaSantorini
//
//  Created by Tomás Ruiz-López on 1/16/26.
//

import Foundation
import MCTS

public struct TrainingConfig {
    // Self-play
    let hiddenDimension: Int = 256
    let gamesPerIteration: Int = 100
    let MCTSSimulations: Int = 200
    let mctsBatchSize: Int = 256
    let noise: DirichletNoise = DirichletNoise(epsilon: 0.25, alpha: 0.3)

    // Training
    let batchSize: Int = 64
    let trainingStepsPerIteration: Int = 100
    let learningRate: Float = 0.001
    let replayBufferSize: Int = 100_000

    // Evaluation
    let evaluationGames: Int = 10
    let promotionThreshold: Float = 0.55
    let evaluationInterval: Int = 5

    // Checkpointing
    let checkpointInterval: Int = 10
    public let checkpointDirectory: URL = URL(filePath: #file)
        .deletingLastPathComponent()
        .deletingLastPathComponent()
        .appending(
            path: "checkpoints",
            directoryHint: .isDirectory
        )

    public init() {}
}
