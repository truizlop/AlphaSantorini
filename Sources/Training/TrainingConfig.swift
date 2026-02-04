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
    public var hiddenDimension: Int
    public var gamesPerIteration: Int
    public var MCTSSimulations: Int
    public var mctsBatchSize: Int
    public var noise: DirichletNoise?
    public var maxMovesPerGame: Int?

    // Training
    public var batchSize: Int
    public var trainingStepsPerIteration: Int
    public var learningRate: Float
    public var replayBufferSize: Int

    // Evaluation
    public var evaluationGames: Int
    public var promotionThreshold: Float
    public var evaluationInterval: Int

    // Checkpointing
    public var checkpointInterval: Int
    public var checkpointDirectory: URL

    public init(
        hiddenDimension: Int = 256,
        gamesPerIteration: Int = 100,
        MCTSSimulations: Int = 200,
        mctsBatchSize: Int = 256,
        noise: DirichletNoise? = DirichletNoise(epsilon: 0.25, alpha: 0.3),
        maxMovesPerGame: Int? = nil,
        batchSize: Int = 64,
        trainingStepsPerIteration: Int = 100,
        learningRate: Float = 0.001,
        replayBufferSize: Int = 100_000,
        evaluationGames: Int = 10,
        promotionThreshold: Float = 0.55,
        evaluationInterval: Int = 5,
        checkpointInterval: Int = 10,
        checkpointDirectory: URL? = nil
    ) {
        self.hiddenDimension = hiddenDimension
        self.gamesPerIteration = gamesPerIteration
        self.MCTSSimulations = MCTSSimulations
        self.mctsBatchSize = mctsBatchSize
        self.noise = noise
        self.maxMovesPerGame = maxMovesPerGame
        self.batchSize = batchSize
        self.trainingStepsPerIteration = trainingStepsPerIteration
        self.learningRate = learningRate
        self.replayBufferSize = replayBufferSize
        self.evaluationGames = evaluationGames
        self.promotionThreshold = promotionThreshold
        self.evaluationInterval = evaluationInterval
        self.checkpointInterval = checkpointInterval
        if let checkpointDirectory {
            self.checkpointDirectory = checkpointDirectory
        } else {
            self.checkpointDirectory = URL(filePath: #file)
                .deletingLastPathComponent()
                .deletingLastPathComponent()
                .appending(
                    path: "checkpoints",
                    directoryHint: .isDirectory
                )
        }
    }
}
