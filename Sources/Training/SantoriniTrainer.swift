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
        var lastPolicyLoss: Float = 0.0
        var lastValueLoss: Float = 0.0

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

            let shouldLog = step == 1 || step % 50 == 0 || step == config.trainingStepsPerIteration
            var stepPolicyLoss: MLXArray?
            var stepValueLoss: MLXArray?
            var stepTargetPolicySum: MLXArray?
            var stepPredPolicySum: MLXArray?
            var stepTargetEntropy: MLXArray?
            var stepPredEntropy: MLXArray?
            var stepKLDivergence: MLXArray?
            var stepTargetMax: MLXArray?
            var stepPredMax: MLXArray?
            let (_, grad) = valueAndGrad(model: model) { net, input, targets in
                let (policy, value) = net(input)
                let splits = targets.split(indices: [valuesPerPolicy], axis: 1)
                if shouldLog {
                    stepTargetPolicySum = splits[0].sum(axis: 1)
                    stepPredPolicySum = policy.sum(axis: 1)
                    let eps: Float = 1e-8
                    let logTarget = log(splits[0] + eps)
                    let logPred = log(policy + eps)
                    stepTargetEntropy = -sum(splits[0] * logTarget, axis: 1)
                    stepPredEntropy = -sum(policy * logPred, axis: 1)
                    let crossEntropy = -sum(splits[0] * logPred, axis: 1)
                    if let stepTargetEntropy {
                        stepKLDivergence = crossEntropy - stepTargetEntropy
                    } else {
                        stepKLDivergence = crossEntropy
                    }
                    stepTargetMax = splits[0].max(axis: 1)
                    stepPredMax = policy.max(axis: 1)
                }
                let policyLoss = self.policyLoss(predicted: policy, target: splits[0])
                let valueLoss = self.valueLoss(predicted: value, target: splits[1])
                stepPolicyLoss = policyLoss
                stepValueLoss = valueLoss
                return policyLoss + valueLoss
            }(model, states, targets)

            if let stepPolicyLoss, let stepValueLoss {
                eval(stepPolicyLoss, stepValueLoss)
                let policyLossValue = stepPolicyLoss.item(Float.self)
                let valueLossValue = stepValueLoss.item(Float.self)
                totalPolicyLoss += policyLossValue
                totalValueLoss += valueLossValue
                lastPolicyLoss = policyLossValue
                lastValueLoss = valueLossValue
            }

            if shouldLog {
                if let stepTargetPolicySum, let stepPredPolicySum {
                    eval(stepTargetPolicySum, stepPredPolicySum)
                }
                if let stepTargetEntropy, let stepPredEntropy, let stepKLDivergence, let stepTargetMax, let stepPredMax {
                    eval(stepTargetEntropy, stepPredEntropy, stepKLDivergence, stepTargetMax, stepPredMax)
                }
                logPolicyDiagnostics(
                    targetSums: stepTargetPolicySum,
                    predictedSums: stepPredPolicySum,
                    targetEntropy: stepTargetEntropy,
                    predictedEntropy: stepPredEntropy,
                    klDivergence: stepKLDivergence,
                    targetMax: stepTargetMax,
                    predictedMax: stepPredMax,
                    step: step
                )
            }

            optimizer.update(model: model, gradients: grad)
            eval(model, optimizer)
        }

        let steps = Float(config.trainingStepsPerIteration)
        let meanPolicyLoss = steps > 0 ? totalPolicyLoss / steps : 0
        let meanValueLoss = steps > 0 ? totalValueLoss / steps : 0
        print("""
            Training ended: policy_loss=\(String(format: "%.4f", lastPolicyLoss)) \
            (mean \(String(format: "%.4f", meanPolicyLoss))), \
            value_loss=\(String(format: "%.4f", lastValueLoss)) \
            (mean \(String(format: "%.4f", meanValueLoss)))
            """)
        trainingHistory.append((iteration, meanPolicyLoss, meanValueLoss))
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

    private func logPolicyDiagnostics(
        targetSums: MLXArray?,
        predictedSums: MLXArray?,
        targetEntropy: MLXArray?,
        predictedEntropy: MLXArray?,
        klDivergence: MLXArray?,
        targetMax: MLXArray?,
        predictedMax: MLXArray?,
        step: Int
    ) {
        if let targetSums, let predictedSums {
            let targetStats = summarize(targetSums)
            let predictedStats = summarize(predictedSums)
            let message = String(
                format: "Policy sum stats at step %d (target mean=%.6f min=%.6f max=%.6f, predicted mean=%.6f min=%.6f max=%.6f)",
                step,
                targetStats.mean, targetStats.min, targetStats.max,
                predictedStats.mean, predictedStats.min, predictedStats.max
            )
            print(message)

            if abs(targetStats.mean - 1.0) > 1e-2 || targetStats.min < 0.98 || targetStats.max > 1.02 {
                print("⚠️ Target policy sums are outside tolerance at step \(step).")
            }
            if abs(predictedStats.mean - 1.0) > 1e-2 || predictedStats.min < 0.98 || predictedStats.max > 1.02 {
                print("⚠️ Predicted policy sums are outside tolerance at step \(step).")
            }
        }

        if let targetEntropy, let predictedEntropy, let klDivergence {
            let targetStats = summarize(targetEntropy)
            let predictedStats = summarize(predictedEntropy)
            let klStats = summarize(klDivergence)
            let message = String(
                format: "Policy entropy/KL at step %d (target mean=%.4f min=%.4f max=%.4f, predicted mean=%.4f min=%.4f max=%.4f, KL mean=%.4f min=%.4f max=%.4f)",
                step,
                targetStats.mean, targetStats.min, targetStats.max,
                predictedStats.mean, predictedStats.min, predictedStats.max,
                klStats.mean, klStats.min, klStats.max
            )
            print(message)
        }

        if let targetMax, let predictedMax {
            let targetStats = summarize(targetMax)
            let predictedStats = summarize(predictedMax)
            let message = String(
                format: "Policy peak probs at step %d (target mean=%.4f min=%.4f max=%.4f, predicted mean=%.4f min=%.4f max=%.4f)",
                step,
                targetStats.mean, targetStats.min, targetStats.max,
                predictedStats.mean, predictedStats.min, predictedStats.max
            )
            print(message)
        }
    }

    private func summarize(_ array: MLXArray) -> (min: Float, max: Float, mean: Float) {
        let values = array.asArray(Float.self)
        guard let first = values.first else { return (0, 0, 0) }
        var minValue = first
        var maxValue = first
        var sum: Float = 0
        for value in values {
            minValue = min(minValue, value)
            maxValue = max(maxValue, value)
            sum += value
        }
        let mean = sum / Float(values.count)
        return (minValue, maxValue, mean)
    }
}
