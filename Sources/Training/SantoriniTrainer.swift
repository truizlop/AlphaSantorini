//
//  SantoriniTrainer.swift
//  AlphaSantorini
//
//  Created by Tomás Ruiz-López on 1/15/26.
//

import Foundation
import MLX
import MLXNN
import MLXOptimizers
import NeuralNetwork
import MCTS

public class SantoriniTrainer {
    let config: TrainingConfig
    var model: SantoriniNet
    var bestModel: SantoriniNet
    var optimizer: Adam
    var replayBuffer: ReplayBuffer
    let selfPlay: SelfPlay
    let aiVSPlay: AIVSPlay

    var iterationsSincePromotion = 0
    var totalGamesPlayed = 0
    var trainingHistory: [(iteration: Int, policyLoss: Float, valueLoss: Float)] = []

    public init(config: TrainingConfig) {
        self.config = config
        self.model = SantoriniNet(hiddenDimension: config.hiddenDimension)
        self.bestModel = SantoriniNet(hiddenDimension: config.hiddenDimension)
        bestModel.copyWeights(from: model)
        self.optimizer = Adam(learningRate: config.learningRate)
        self.replayBuffer = ReplayBuffer(maxSize: config.replayBufferSize)
        self.selfPlay = SelfPlay()
        self.aiVSPlay = AIVSPlay()
    }

    public func train(iterations: Int) async {
        print("Starting Santorini AlphaZero training")

        for iteration in 1 ... iterations {
            print("""
                
                ============================================
                Iteration \(iteration)
                ============================================
                """)

            // Phase 1: Self-play
            await selfPlayPhase()
            replayBuffer.checkDiversity()
            // Phase 2: Training
            trainingPhase(iteration: iteration)
            // Phase 3: Evaluation
            if iteration % config.evaluationInterval == 0 {
                evaluationPhase(iteration: iteration)
            }
            // Phase 4: Checkpointing
            if iteration % config.checkpointInterval == 0 {
                checkpointPhase(iteration: iteration)
            }
            // Early stop
            if shouldStopEarly() {
                print("Early stop after seeing no improvements for \(iterationsSincePromotion) iterations.")
                break
            }
        }

        print("Training complete!")
        print("Total games played: \(totalGamesPlayed)")
        checkpointPhase(iteration: -1)
    }

    private func selfPlayPhase() async {
        print("Beginning self-play...")
        let iterations = self.config.MCTSSimulations
        let batchSize = self.config.mctsBatchSize
        let selfPlay = self.selfPlay
        let noise = self.config.noise
        let maxMovesPerGame = self.config.maxMovesPerGame
        let profilingEnabled = ProcessInfo.processInfo.environment["SANTORINI_PROFILE"] == "1"
        MCTSProfiler.enabled = profilingEnabled

        let start = Date().timeIntervalSince1970
        var truncatedGames = 0
        for game in 1 ... self.config.gamesPerIteration {
            let result = selfPlay.runWithDiagnostics(
                evaluator: model,
                iterations: iterations,
                noise: noise,
                batchSize: batchSize,
                maxMoves: maxMovesPerGame
            )
            if result.wasTruncated {
                truncatedGames += 1
            } else {
                replayBuffer.add(result.samples)
            }
            if game % 5 == 0 {
                print("Completed self-play \(game)/\(config.gamesPerIteration)")
            }
        }
        let end = Date().timeIntervalSince1970
        print("Self play took: \(end-start)")
        if profilingEnabled {
            print(MCTSProfiler.reportAndReset(prefix: "Self-play MCTS"))
        }

        totalGamesPlayed += config.gamesPerIteration
        if truncatedGames > 0 {
            print("Self-play ended (\(replayBuffer.count) training samples, \(truncatedGames) truncated games).")
        } else {
            print("Self-play ended (\(replayBuffer.count) training samples).")
        }
    }

    private func trainingPhase(iteration: Int) {
        guard replayBuffer.count >= config.batchSize else {
            print("⚠️ WARNING: Not enough samples in buffer (\(replayBuffer.count))")
            return
        }

        print("Beginning training...")
        var totalPolicyLoss: Float = 0.0
        var totalValueLoss: Float = 0.0

        for step in 1 ... config.trainingStepsPerIteration {
            let batch = replayBuffer.sample(batchSize: config.batchSize)
            let encodedStates = batch.map { $0.state.encoded() }
            let encodedPolicies = batch.map { $0.encodedPolicy }
            let valuesPerPolicy = encodedPolicies[0].count
            let encodedValues = batch.map { $0.outcome }

            let states = MLXArray(encodedStates.flatMap { $0 }, [batch.count, encodedStates[0].count])
            let targetPolicies = MLXArray(encodedPolicies.flatMap { $0 }, [batch.count, encodedPolicies[0].count])
            let targetValues = MLXArray(encodedValues, [batch.count, 1])
            let targets = concatenated([targetPolicies, targetValues], axis: 1)

            var stepPolicyLoss: MLXArray?
            var stepValueLoss: MLXArray?
            let (loss, grad) = valueAndGrad(model: model) { net, input, targets in
                let (policy, value) = net(input)
                let splits = targets.split(indices: [valuesPerPolicy], axis: 1)
                let policyLoss = self.policyLoss(predicted: policy, target: splits[0])
                let valueLoss = self.valueLoss(predicted: value, target: splits[1])
                stepPolicyLoss = policyLoss
                stepValueLoss = valueLoss
                return policyLoss + valueLoss
            }(model, states, targets)

            if step % 50 == 0 || step == config.trainingStepsPerIteration {
                if let stepPolicyLoss, let stepValueLoss {
                    eval(stepPolicyLoss, stepValueLoss)
                    totalPolicyLoss = stepPolicyLoss.item(Float.self)
                    totalValueLoss = stepValueLoss.item(Float.self)
                }
            }

            optimizer.update(model: model, gradients: grad)
            eval(model, optimizer)
        }

        print("Training ended: policy_loss=\(String(format: "%.4f", totalPolicyLoss)), value_loss=\(String(format: "%.4f", totalValueLoss))")
        trainingHistory.append((iteration, totalPolicyLoss, totalValueLoss))
    }

    private func policyLoss(
        predicted: MLXArray,
        target: MLXArray
    ) -> MLXArray {
        let batchSize = max(1, predicted.shape[0])
        return -sum(target * log(predicted + 1e-8)) / Float(batchSize)
    }

    private func valueLoss(
        predicted: MLXArray,
        target: MLXArray
    ) -> MLXArray {
        mseLoss(predictions: predicted, targets: target, reduction: .mean)
    }

    private func evaluationPhase(iteration: Int) {
        print("Beginning evaluation...")
        let result = playMatches(
            current: model,
            best: bestModel,
            games: config.evaluationGames
        )

        let totalDecisive = result.winsCurrent + result.winsBest
        let winRate = if totalDecisive > 0 {
            Float(result.winsCurrent) / Float(totalDecisive)
        } else {
            Float(0.5) // All draws
        }

        if winRate >= config.promotionThreshold {
            print("✅ New network promoted! Win rate: \(winRate)")
            bestModel.copyWeights(from: model)
            do {
                try bestModel.save(to: config.checkpointDirectory.appending(path: "best.safetensors"))
            } catch {
                print("💥 Could not save best network to disk")
            }
            iterationsSincePromotion = 0
        } else {
            print("👎🏻 New network not promoted. Win rate: \(winRate)")
            iterationsSincePromotion += config.evaluationInterval
        }

        print("Evaluation ended.")
    }

    private func playMatches(
        current: SantoriniNet,
        best: SantoriniNet,
        games: Int
    ) -> (winsCurrent: Int, winsBest: Int, draws: Int) {
        var winsCurrent = 0
        var winsBest = 0
        var draws = 0

        for game in 0 ..< games {
            let currentNetworkPlaysFirst = (game % 2 == 0)
            if currentNetworkPlaysFirst {
                if let winner = aiVSPlay.play(
                    player1: current,
                    player2: best,
                    maxMoves: config.maxMovesPerGame
                ) {
                    winsCurrent += (winner == .one) ? 1 : 0
                    winsBest += (winner == .two) ? 1 : 0
                } else {
                    draws += 1
                }
            } else {
                if let winner = aiVSPlay.play(
                    player1: best,
                    player2: current,
                    maxMoves: config.maxMovesPerGame
                ) {
                    winsCurrent += (winner == .two) ? 1 : 0
                    winsBest += (winner == .one) ? 1 : 0
                } else {
                    draws += 1
                }
            }
        }
        return (winsCurrent, winsBest, draws)
    }

    private func checkpointPhase(iteration: Int) {
        print("Saving checkpoint...")
        let fileURL = if iteration < 0 {
            config.checkpointDirectory.appending(path: "final.safetensors")
        } else {
            config.checkpointDirectory.appending(path: "checkpoint_\(iteration).safetensors")
        }

        do {
            try model.save(to: fileURL)
            print("Checkpoint saved: \(fileURL.lastPathComponent)")
        } catch {
            print("Failed to save network: \(error)")
        }
    }

    private func shouldStopEarly() -> Bool {
        iterationsSincePromotion >= 100
    }
}
