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
import Santorini

public class SantoriniTrainer {
    let config: TrainingConfig
    var model: SantoriniNet
    var bestModel: SantoriniNet
    var optimizer: Adam
    var replayBuffer: ReplayBuffer
    let selfPlay: SelfPlay
    let aiVSPlay: AIVSPlay

    var iterationsSincePromotion = 0
    var lastPromotionIteration: Int? = nil
    var totalGamesPlayed = 0
    var trainingHistory: [(iteration: Int, policyLoss: Float, valueLoss: Float)] = []
    private var valueEvalSet: [(state: Santorini.GameState, rolloutValue: Float)] = []
    private let valueEvalSeed: UInt64 = 42

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
            await selfPlayPhase(iteration: iteration)
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
            if shouldStopEarly(currentIteration: iteration) {
                if let lastPromotionIteration {
                    print("Early stop after seeing no improvements for \(iterationsSincePromotion) iterations (last promotion at iteration \(lastPromotionIteration)).")
                } else {
                    print("Early stop after seeing no improvements for \(iterationsSincePromotion) iterations.")
                }
                break
            }
        }

        print("Training complete!")
        print("Total games played: \(totalGamesPlayed)")
        checkpointPhase(iteration: -1)
    }

    private func selfPlayPhase(iteration: Int) async {
        print("Beginning self-play...")
        let iterations = self.config.MCTSSimulations
        let batchSize = self.config.mctsBatchSize
        let selfPlay = self.selfPlay
        let noise = annealedNoise(for: iteration)
        let valueTargetStrategy = config.valueTargetStrategy
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
                valueTargetStrategy: valueTargetStrategy
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

    private func annealedNoise(for iteration: Int) -> DirichletNoise? {
        guard let baseNoise = config.noise else { return nil }
        let annealIters = max(1, config.noiseAnnealIterations)
        guard annealIters > 1 else { return baseNoise }

        let clamped = min(max(iteration - 1, 0), annealIters - 1)
        let progress = Float(clamped) / Float(annealIters - 1)
        let epsilon = max(config.noiseEpsilonFloor, baseNoise.epsilon * (1 - progress))
        guard epsilon > 0 else { return nil }
        return DirichletNoise(epsilon: epsilon, alpha: baseNoise.alpha)
    }

    private func trainingPhase(iteration: Int) {
        guard replayBuffer.count >= config.batchSize else {
            print("⚠️ WARNING: Not enough samples in buffer (\(replayBuffer.count))")
            return
        }

        print("Beginning training...")
        var totalPolicyLoss: Float = 0.0
        var totalValueLoss: Float = 0.0
        var totalBaselineMSE: Float = 0.0
        var lastPolicyLoss: Float = 0.0
        var lastValueLoss: Float = 0.0
        var lastBaselineMSE: Float = 0.0

        var headerPrinted = false
        var lastSnapshot: PolicyLogSnapshot?
        var logSnapshots: [PolicyLogSnapshot] = []

        for step in 1 ... config.trainingStepsPerIteration {
            let batch = replayBuffer.sample(batchSize: config.batchSize)
            let encodedStates = batch.map { $0.state.encoded() }
            let encodedPolicies = batch.map { $0.encodedPolicy }
            let valuesPerPolicy = encodedPolicies[0].count
            let encodedValues = batch.map { $0.outcome }
            if !encodedValues.isEmpty {
                let mean = encodedValues.reduce(0, +) / Float(encodedValues.count)
                let mse = encodedValues.reduce(Float(0)) { acc, value in
                    let diff = value - mean
                    return acc + diff * diff
                } / Float(encodedValues.count)
                totalBaselineMSE += mse
                lastBaselineMSE = mse
            }

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
                let (policyLogits, value) = net(input)
                let splits = targets.split(indices: [valuesPerPolicy], axis: 1)
                if shouldLog {
                    stepTargetPolicySum = splits[0].sum(axis: 1)
                    let policyProbs = softmax(policyLogits, axis: -1)
                    stepPredPolicySum = policyProbs.sum(axis: 1)
                    let eps: Float = 1e-8
                    let logTarget = log(splits[0] + eps)
                    let logPred = logSoftmax(policyLogits, axis: -1)
                    stepTargetEntropy = -sum(splits[0] * logTarget, axis: 1)
                    stepPredEntropy = -sum(policyProbs * logPred, axis: 1)
                    let crossEntropy = -sum(splits[0] * logPred, axis: 1)
                    if let stepTargetEntropy {
                        stepKLDivergence = crossEntropy - stepTargetEntropy
                    } else {
                        stepKLDivergence = crossEntropy
                    }
                    stepTargetMax = splits[0].max(axis: 1)
                    stepPredMax = policyProbs.max(axis: 1)
                }
                let policyLoss = self.policyLoss(logits: policyLogits, target: splits[0])
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
                lastSnapshot = logPolicyDiagnostics(
                    targetSums: stepTargetPolicySum,
                    predictedSums: stepPredPolicySum,
                    targetEntropy: stepTargetEntropy,
                    predictedEntropy: stepPredEntropy,
                    klDivergence: stepKLDivergence,
                    targetMax: stepTargetMax,
                    predictedMax: stepPredMax,
                    step: step,
                    policyLoss: lastPolicyLoss,
                    valueLoss: lastValueLoss,
                    previous: lastSnapshot,
                    headerPrinted: &headerPrinted
                )
                if let lastSnapshot {
                    logSnapshots.append(lastSnapshot)
                }
            }

            optimizer.update(model: model, gradients: grad)
            eval(model, optimizer)
        }

        let steps = Float(config.trainingStepsPerIteration)
        let meanPolicyLoss = steps > 0 ? totalPolicyLoss / steps : 0
        let meanValueLoss = steps > 0 ? totalValueLoss / steps : 0
        let meanBaselineMSE = steps > 0 ? totalBaselineMSE / steps : 0
        let valueLossRatio = meanBaselineMSE > 0 ? meanValueLoss / meanBaselineMSE : 0
        print("""
            Training ended: policy_loss=\(String(format: "%.4f", lastPolicyLoss)) \
            (mean \(String(format: "%.4f", meanPolicyLoss))), \
            value_loss=\(String(format: "%.4f", lastValueLoss)) \
            (mean \(String(format: "%.4f", meanValueLoss)))
            """)
        if meanBaselineMSE > 0 {
            print(String(format: "Value baseline MSE=%.4f (last %.4f), value_loss/baseline=%.3f",
                         meanBaselineMSE, lastBaselineMSE, valueLossRatio))
        }
        logTrendSummary(logSnapshots)
        trainingHistory.append((iteration, meanPolicyLoss, meanValueLoss))

        logValueEvaluation(iteration: iteration)
    }

    private func logValueEvaluation(iteration: Int) {
        let interval = max(1, config.valueEvaluationInterval)
        guard iteration % interval == 0 else { return }

        if valueEvalSet.isEmpty {
            valueEvalSet = buildValueEvalSet(
                count: config.valueEvaluationStates,
                playouts: config.valueEvaluationPlayouts,
                seed: valueEvalSeed
            )
        }
        guard !valueEvalSet.isEmpty else { return }

        let inputs = valueEvalSet.map { $0.state.encoded() }
        let targets = valueEvalSet.map { $0.rolloutValue }
        let (_, predictions) = model.evaluateBatch(inputs)
        guard predictions.count == targets.count else { return }

        let corr = pearsonCorrelation(predictions, targets)
        let mse = meanSquaredError(predictions: predictions, targets: targets)
        let baseline = baselineMSE(outcomes: targets)
        let ratio = baseline > 0 ? mse / baseline : 0.0

        print(String(format: "Value eval: corr=%.3f mse=%.4f baseline=%.4f ratio=%.3f",
                     corr, mse, baseline, ratio))
    }

    private func buildValueEvalSet(
        count: Int,
        playouts: Int,
        seed: UInt64
    ) -> [(state: Santorini.GameState, rolloutValue: Float)] {
        guard count > 0 else { return [] }
        var rng = SeededGenerator(seed: seed)
        var samples: [Santorini.GameState] = []
        samples.reserveCapacity(count)

        while samples.count < count {
            var state = Santorini.GameState()
            var steps = 0
            while !state.isOver && samples.count < count && steps < 200 {
                samples.append(state)
                let legal = state.legalActions
                guard !legal.isEmpty else { break }
                let idx = Int.random(in: 0..<legal.count, using: &rng)
                state = state.applying(move: legal[idx])
                steps += 1
            }
        }

        return samples.map { state in
            let value = rolloutValueEstimate(state: state, playouts: playouts, rng: &rng)
            return (state, value)
        }
    }

    private func rolloutValueEstimate(
        state: Santorini.GameState,
        playouts: Int,
        rng: inout SeededGenerator
    ) -> Float {
        var wins = 0
        var losses = 0
        let perspective = state.turn
        for _ in 0..<playouts {
            var s = state
            var steps = 0
            while !s.isOver && steps < 200 {
                let legal = s.legalActions
                guard !legal.isEmpty else { break }
                let idx = Int.random(in: 0..<legal.count, using: &rng)
                s = s.applying(move: legal[idx])
                steps += 1
            }
            if let winner = s.winner {
                if winner == perspective { wins += 1 } else { losses += 1 }
            }
        }
        let total = wins + losses
        let winRate = total > 0 ? Float(wins) / Float(total) : 0.5
        return 2 * winRate - 1
    }

    private func meanSquaredError(predictions: [Float], targets: [Float]) -> Float {
        guard predictions.count == targets.count, !predictions.isEmpty else { return 0 }
        let mse = zip(predictions, targets).reduce(Float(0)) { acc, pair in
            let diff = pair.0 - pair.1
            return acc + diff * diff
        }
        return mse / Float(predictions.count)
    }

    private func baselineMSE(outcomes: [Float]) -> Float {
        guard !outcomes.isEmpty else { return 0 }
        let mean = outcomes.reduce(0, +) / Float(outcomes.count)
        let mse = outcomes.reduce(Float(0)) { acc, value in
            let diff = value - mean
            return acc + diff * diff
        }
        return mse / Float(outcomes.count)
    }

    private func pearsonCorrelation(_ xs: [Float], _ ys: [Float]) -> Float {
        guard xs.count == ys.count, !xs.isEmpty else { return 0 }
        let n = Float(xs.count)
        let meanX = xs.reduce(0, +) / n
        let meanY = ys.reduce(0, +) / n
        var cov: Float = 0
        var varX: Float = 0
        var varY: Float = 0
        for (x, y) in zip(xs, ys) {
            let dx = x - meanX
            let dy = y - meanY
            cov += dx * dy
            varX += dx * dx
            varY += dy * dy
        }
        let denom = sqrt(varX * varY)
        return denom > 0 ? cov / denom : 0
    }

    private struct SeededGenerator: RandomNumberGenerator {
        private var state: UInt64

        init(seed: UInt64) {
            self.state = seed
        }

        mutating func next() -> UInt64 {
            state = state &* 6364136223846793005 &+ 1442695040888963407
            return state
        }
    }

    private func policyLoss(
        logits: MLXArray,
        target: MLXArray
    ) -> MLXArray {
        let batchSize = max(1, logits.shape[0])
        let logProbs = logSoftmax(logits, axis: -1)
        return -sum(target * logProbs) / Float(batchSize)
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

        print("Evaluation results: currentWins=\(result.winsCurrent), bestWins=\(result.winsBest), draws=\(result.draws), decisive=\(totalDecisive), winRate=\(String(format: "%.3f", winRate))")
        logModelDifference(current: model, best: bestModel)

        if winRate >= config.promotionThreshold {
            print("✅ New network promoted! Win rate: \(winRate)")
            bestModel.copyWeights(from: model)
            do {
                try bestModel.save(to: config.checkpointDirectory.appending(path: "best.safetensors"))
            } catch {
                print("💥 Could not save best network to disk")
            }
            iterationsSincePromotion = 0
            lastPromotionIteration = iteration
        } else {
            print("👎🏻 New network not promoted. Win rate: \(winRate)")
            iterationsSincePromotion = iteration - (lastPromotionIteration ?? 0)
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
                    iterations: config.MCTSSimulations
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
                    iterations: config.MCTSSimulations
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

    private func shouldStopEarly(currentIteration: Int) -> Bool {
        iterationsSincePromotion = currentIteration - (lastPromotionIteration ?? 0)
        return iterationsSincePromotion >= 100
    }

    private func logModelDifference(current: SantoriniNet, best: SantoriniNet) {
        let currentParams = Dictionary(uniqueKeysWithValues: current.parameters().flattened())
        let bestParams = Dictionary(uniqueKeysWithValues: best.parameters().flattened())
        let keys = Set(currentParams.keys).intersection(bestParams.keys)
        guard !keys.isEmpty else {
            print("⚠️ Could not compare model parameters (no overlapping keys).")
            return
        }

        var maxAbsDiff: Float = 0
        var meanAbsDiffSum: Float = 0
        var count: Int = 0

        for key in keys {
            guard let currentArray = currentParams[key], let bestArray = bestParams[key] else { continue }
            let diff = abs(currentArray - bestArray)
            let maxDiff = diff.max().item(Float.self)
            let meanDiff = diff.mean().item(Float.self)
            maxAbsDiff = max(maxAbsDiff, maxDiff)
            meanAbsDiffSum += meanDiff
            count += 1
        }

        if count > 0 {
            let meanAbsDiff = meanAbsDiffSum / Float(count)
            print("Model param diff: meanAbs=\(String(format: "%.6f", meanAbsDiff)), maxAbs=\(String(format: "%.6f", maxAbsDiff)) across \(count) tensors.")
            if maxAbsDiff <= 1.0e-7 {
                print("⚠️ Model parameters are effectively identical; evaluation will tend toward 0.5 win rate or all draws.")
            }
        }
    }

    private struct PolicyLogSnapshot {
        let step: Int
        let policyLoss: Float
        let valueLoss: Float
        let targetSum: Float
        let predictedSum: Float
        let targetEntropy: Float
        let predictedEntropy: Float
        let klDivergence: Float
        let targetMax: Float
        let predictedMax: Float
    }

    private func logPolicyDiagnostics(
        targetSums: MLXArray?,
        predictedSums: MLXArray?,
        targetEntropy: MLXArray?,
        predictedEntropy: MLXArray?,
        klDivergence: MLXArray?,
        targetMax: MLXArray?,
        predictedMax: MLXArray?,
        step: Int,
        policyLoss: Float,
        valueLoss: Float,
        previous: PolicyLogSnapshot?,
        headerPrinted: inout Bool
    ) -> PolicyLogSnapshot? {
        guard let targetSums,
              let predictedSums,
              let targetEntropy,
              let predictedEntropy,
              let klDivergence,
              let targetMax,
              let predictedMax else {
            return previous
        }

        let targetSumStats = summarize(targetSums)
        let predictedSumStats = summarize(predictedSums)
        let targetEntropyStats = summarize(targetEntropy)
        let predictedEntropyStats = summarize(predictedEntropy)
        let klStats = summarize(klDivergence)
        let targetMaxStats = summarize(targetMax)
        let predictedMaxStats = summarize(predictedMax)

        let snapshot = PolicyLogSnapshot(
            step: step,
            policyLoss: policyLoss,
            valueLoss: valueLoss,
            targetSum: targetSumStats.mean,
            predictedSum: predictedSumStats.mean,
            targetEntropy: targetEntropyStats.mean,
            predictedEntropy: predictedEntropyStats.mean,
            klDivergence: klStats.mean,
            targetMax: targetMaxStats.mean,
            predictedMax: predictedMaxStats.mean
        )

        if !headerPrinted {
            print("""
                Step | P_Loss  dP     | V_Loss  dV     | KL      dKL    | P_Ent  | P_Max  | T_Ent  | T_Max  | SumP  | SumT
                -----+---------------+---------------+---------------+--------+--------+--------+--------+-------+------
                """)
            headerPrinted = true
        }

        func delta(_ current: Float, _ previous: Float?) -> String {
            guard let previous else { return "   --" }
            let value = current - previous
            return String(format: "%+7.4f", Double(value))
        }

        let line = String(
            format: "%4d | %7.4f %@ | %7.4f %@ | %7.4f %@ | %6.3f | %6.3f | %6.3f | %6.3f | %5.3f | %5.3f",
            step,
            Double(snapshot.policyLoss), delta(snapshot.policyLoss, previous?.policyLoss),
            Double(snapshot.valueLoss), delta(snapshot.valueLoss, previous?.valueLoss),
            Double(snapshot.klDivergence), delta(snapshot.klDivergence, previous?.klDivergence),
            Double(snapshot.predictedEntropy),
            Double(snapshot.predictedMax),
            Double(snapshot.targetEntropy),
            Double(snapshot.targetMax),
            Double(snapshot.predictedSum),
            Double(snapshot.targetSum)
        )
        print(line)

        if abs(targetSumStats.mean - 1.0) > 1e-2 || targetSumStats.min < 0.98 || targetSumStats.max > 1.02 {
            print("⚠️ Target policy sums are outside tolerance at step \(step).")
        }
        if abs(predictedSumStats.mean - 1.0) > 1e-2 || predictedSumStats.min < 0.98 || predictedSumStats.max > 1.02 {
            print("⚠️ Predicted policy sums are outside tolerance at step \(step).")
        }

        return snapshot
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

    private func logTrendSummary(_ snapshots: [PolicyLogSnapshot]) {
        guard let first = snapshots.first, let last = snapshots.last, snapshots.count > 1 else {
            return
        }

        func trend(_ first: Float, _ last: Float) -> (arrow: String, delta: Float, perLog: Float) {
            let delta = last - first
            let perLog = delta / Float(max(1, snapshots.count - 1))
            let arrow: String
            if delta < -0.01 {
                arrow = "↓"
            } else if delta > 0.01 {
                arrow = "↑"
            } else {
                arrow = "→"
            }
            return (arrow, delta, perLog)
        }

        let policyTrend = trend(first.policyLoss, last.policyLoss)
        let valueTrend = trend(first.valueLoss, last.valueLoss)
        let klTrend = trend(first.klDivergence, last.klDivergence)
        let pEntTrend = trend(first.predictedEntropy, last.predictedEntropy)
        let pMaxTrend = trend(first.predictedMax, last.predictedMax)

        print("""
            Trend summary (first→last over \(snapshots.count) logs):
              Policy loss \(policyTrend.arrow) Δ\(String(format: "%.4f", policyTrend.delta)) (per log \(String(format: "%.4f", policyTrend.perLog)))
              Value loss  \(valueTrend.arrow) Δ\(String(format: "%.4f", valueTrend.delta)) (per log \(String(format: "%.4f", valueTrend.perLog)))
              KL          \(klTrend.arrow) Δ\(String(format: "%.4f", klTrend.delta)) (per log \(String(format: "%.4f", klTrend.perLog)))
              Pred entropy\(pEntTrend.arrow) Δ\(String(format: "%.4f", pEntTrend.delta)) (per log \(String(format: "%.4f", pEntTrend.perLog)))
              Pred max    \(pMaxTrend.arrow) Δ\(String(format: "%.4f", pMaxTrend.delta)) (per log \(String(format: "%.4f", pMaxTrend.perLog)))
            """)
    }
}
