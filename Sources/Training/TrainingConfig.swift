//
//  TrainingConfig.swift
//  AlphaSantorini
//
//  Created by Tomás Ruiz-López on 1/16/26.
//

import Foundation
import MCTS

public enum ValueTargetStrategy: Sendable {
    case terminalOutcome
    case mctsRootValue
}

public struct TrainingConfig {
    // Self-play
    public var gamesPerIteration: Int
    public var MCTSSimulations: Int
    public var noise: DirichletNoise?
    public var noiseAnnealIterations: Int
    public var noiseEpsilonFloor: Float
    public var valueTargetStrategy: ValueTargetStrategy
    public var selfPlayBatchSize: Int
    public var selfPlayConcurrency: Int
    public var batchTimeoutMicroseconds: UInt64

    // Training
    public var batchSize: Int
    public var symmetryAugmentation: Bool
    public var trainingStepsPerIteration: Int
    public var learningRate: Float
    public var replayBufferSize: Int
    public var valueEvaluationInterval: Int
    public var valueEvaluationStates: Int
    public var valueEvaluationPlayouts: Int
    public var sampleQualityInterval: Int
    public var sampleQualitySampleCount: Int

    // Evaluation
    public var evaluationGames: Int
    public var promotionThreshold: Float
    public var evaluationInterval: Int

    // Checkpointing
    public var checkpointInterval: Int
    public var checkpointDirectory: URL

    public init(
        gamesPerIteration: Int = 100,
        MCTSSimulations: Int = 256,
        noise: DirichletNoise? = DirichletNoise(epsilon: 0.25, alpha: 0.3),
        noiseAnnealIterations: Int = 150,
        noiseEpsilonFloor: Float = 0.05,
        valueTargetStrategy: ValueTargetStrategy = .terminalOutcome,
        selfPlayBatchSize: Int = 128,
        selfPlayConcurrency: Int = 0,
        batchTimeoutMicroseconds: UInt64 = 100,
        batchSize: Int = 128,
        symmetryAugmentation: Bool = true,
        trainingStepsPerIteration: Int = 100,
        learningRate: Float = 0.001,
        replayBufferSize: Int = 100_000,
        valueEvaluationInterval: Int = 10,
        valueEvaluationStates: Int = 16,
        valueEvaluationPlayouts: Int = 20,
        sampleQualityInterval: Int = 1,
        sampleQualitySampleCount: Int = 256,
        evaluationGames: Int = 20,
        promotionThreshold: Float = 0.55,
        evaluationInterval: Int = 10,
        checkpointInterval: Int = 10,
        checkpointDirectory: URL? = nil
    ) {
        self.gamesPerIteration = gamesPerIteration
        self.MCTSSimulations = MCTSSimulations
        self.noise = noise
        self.noiseAnnealIterations = noiseAnnealIterations
        self.noiseEpsilonFloor = noiseEpsilonFloor
        self.valueTargetStrategy = valueTargetStrategy
        self.selfPlayBatchSize = selfPlayBatchSize
        self.selfPlayConcurrency = selfPlayConcurrency
        self.batchTimeoutMicroseconds = batchTimeoutMicroseconds
        self.batchSize = batchSize
        self.symmetryAugmentation = symmetryAugmentation
        self.trainingStepsPerIteration = trainingStepsPerIteration
        self.learningRate = learningRate
        self.replayBufferSize = replayBufferSize
        self.valueEvaluationInterval = valueEvaluationInterval
        self.valueEvaluationStates = valueEvaluationStates
        self.valueEvaluationPlayouts = valueEvaluationPlayouts
        self.sampleQualityInterval = sampleQualityInterval
        self.sampleQualitySampleCount = sampleQualitySampleCount
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
