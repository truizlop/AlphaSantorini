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
        self.model = SantoriniNet()
        self.bestModel = SantoriniNet()
        bestModel.copyWeights(from: model)
        model.train(false)
        bestModel.train(false)
        self.optimizer = Adam(learningRate: config.learningRate)
        self.replayBuffer = ReplayBuffer(maxSize: config.replayBufferSize)
        self.selfPlay = SelfPlay()
        self.aiVSPlay = AIVSPlay()
    }

    public func loadCheckpoint(from url: URL) throws {
        try model.load(from: url)
        bestModel.copyWeights(from: model)
        model.train(false)
        bestModel.train(false)
        iterationsSincePromotion = 0
        lastPromotionIteration = nil
    }

    public func train(iterations: Int) async {
        print("Starting Santorini AlphaZero training")
        model.train(false)
        bestModel.train(false)

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
                await evaluationPhase(iteration: iteration)
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
        model.train(false)
        let iterations = self.config.MCTSSimulations
        let selfPlay = self.selfPlay
        let noise = annealedNoise(for: iteration)
        let valueTargetStrategy = config.valueTargetStrategy
        let qualityInterval = max(1, config.sampleQualityInterval)
        let shouldReportQuality = iteration % qualityInterval == 0 && config.sampleQualitySampleCount > 0

        let evaluator = BatchedEvaluator(
            net: model,
            maxBatchSize: config.selfPlayBatchSize,
            timeoutMicroseconds: config.batchTimeoutMicroseconds
        )

        let gamesPerIteration = config.gamesPerIteration
        let maxConcurrency = config.selfPlayConcurrency > 0
            ? config.selfPlayConcurrency
            : gamesPerIteration
        let baseSeed = UInt64(iteration) &* 6364136223846793005 &+ 1

        let start = Date().timeIntervalSince1970
        var results: [SelfPlayResult] = []
        results.reserveCapacity(gamesPerIteration)
        var completedGames = 0

        await withTaskGroup(of: SelfPlayResult.self) { group in
            for gameIndex in 0 ..< gamesPerIteration {
                if group.isCancelled { break }

                // Backpressure: limit concurrency
                if gameIndex >= maxConcurrency {
                    if let result = await group.next() {
                        results.append(result)
                        completedGames += 1
                        if completedGames % 5 == 0 {
                            print("Completed self-play \(completedGames)/\(gamesPerIteration)")
                        }
                    }
                }

                let gameSeed = baseSeed &+ UInt64(gameIndex)
                let net = self.model
                group.addTask {
                    await selfPlay.runBatched(
                        evaluator: evaluator,
                        net: net,
                        iterations: iterations,
                        noise: noise,
                        valueTargetStrategy: valueTargetStrategy,
                        seed: gameSeed
                    )
                }
            }

            // Collect remaining results
            for await result in group {
                results.append(result)
                completedGames += 1
                if completedGames % 5 == 0 || completedGames == gamesPerIteration {
                    print("Completed self-play \(completedGames)/\(gamesPerIteration)")
                }
            }
        }
        let end = Date().timeIntervalSince1970
        print("Self play took: \(String(format: "%.1f", end - start))s")

        // Log batching diagnostics
        let diag = await evaluator.diagnostics()
        print("Batching: \(diag.totalBatches) batches, \(diag.totalEvaluations) evals, avg batch size \(String(format: "%.1f", diag.avgBatchSize))")

        // Process results
        var truncatedGames = 0
        var sampler = ReservoirSampler<TrainingSample>(
            capacity: config.sampleQualitySampleCount,
            seed: UInt64(iteration)
        )
        for result in results {
            if result.wasTruncated {
                truncatedGames += 1
            } else {
                replayBuffer.add(result.samples)
                if shouldReportQuality {
                    sampler.add(contentsOf: result.samples)
                }
            }
        }

        totalGamesPlayed += gamesPerIteration
        if truncatedGames > 0 {
            print("Self-play ended (\(replayBuffer.count) training samples, \(truncatedGames) truncated games).")
        } else {
            print("Self-play ended (\(replayBuffer.count) training samples).")
        }

        if shouldReportQuality {
            logSampleQuality(iteration: iteration, samples: sampler.samples)
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
        model.train(true)
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
            let baseBatch = replayBuffer.sample(batchSize: config.batchSize)
            let batch: [TrainingSample]
            if config.symmetryAugmentation {
                batch = baseBatch.flatMap { $0.augmentedBySymmetry() }
            } else {
                batch = baseBatch
            }
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

            let flatStates = encodedStates.flatMap { $0.flatMap { $0.flatMap { $0 } } }
            let states = MLXArray(flatStates, [batch.count, 5, 5, 9])
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
        model.train(false)

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

    private func logSampleQuality(iteration: Int, samples: [TrainingSample]) {
        guard !samples.isEmpty else {
            print("Sample quality report: no samples available.")
            return
        }

        let metrics = computeSampleQualityMetrics(samples: samples)
        let inputs = samples.map { $0.state.encoded() }
        let targetValues = samples.map { $0.outcome }
        let (predPolicies, predValues) = model.evaluateBatch(inputs)

        var policyKL: Float?
        var valueCorr: Float?
        var valueMSE: Float?
        var valueRatio: Float?

        if predPolicies.count == samples.count {
            var klSum: Float = 0
            for (sample, predicted) in zip(samples, predPolicies) {
                klSum += policyKLDivergence(target: sample.encodedPolicy, predicted: predicted)
            }
            policyKL = klSum / Float(samples.count)
        }

        if predValues.count == samples.count {
            let mse = meanSquaredError(predictions: predValues, targets: targetValues)
            let baseline = baselineMSE(outcomes: targetValues)
            valueMSE = mse
            valueCorr = pearsonCorrelation(predValues, targetValues)
            valueRatio = baseline > 0 ? mse / baseline : 0
        }

        print("""
            Sample quality report (iteration \(iteration), \(metrics.sampleCount) samples):
              Diversity: \(String(format: "%.3f", metrics.uniqueStateRatio)) (unique \(metrics.uniqueStates)/\(metrics.sampleCount))
              Legal moves: mean \(String(format: "%.2f", metrics.meanLegalMoves)) \
            min \(metrics.minLegalMoves) max \(metrics.maxLegalMoves)
              Target policy sum: mean \(String(format: "%.4f", metrics.meanPolicySum)) \
            min \(String(format: "%.4f", metrics.minPolicySum)) \
            max \(String(format: "%.4f", metrics.maxPolicySum))
              Target entropy: mean \(String(format: "%.3f", metrics.meanEntropy)) \
            norm \(String(format: "%.3f", metrics.meanNormalizedEntropy)) \
            maxP \(String(format: "%.3f", metrics.meanMaxProb)) \
            peak×uni \(String(format: "%.2f", metrics.meanPeakRatio))
              Policy support: mean \(String(format: "%.3f", metrics.meanSupportRatio)) of legal moves
              Value targets: mean \(String(format: "%.3f", metrics.valueMean)) \
            std \(String(format: "%.3f", metrics.valueStd)) \
            min \(String(format: "%.2f", metrics.valueMin)) \
            max \(String(format: "%.2f", metrics.valueMax)) \
            |v|<0.1 \(String(format: "%.0f%%", metrics.fracNearZero * 100)) \
            |v|>0.99 \(String(format: "%.0f%%", metrics.fracAbsOne * 100)) \
            baseline \(String(format: "%.4f", metrics.baselineMSE))
            """)

        if let policyKL {
            print(String(format: "  Policy KL(target||net): %.4f", policyKL))
        }
        if let valueCorr, let valueMSE, let valueRatio {
            print(String(format: "  Value vs net: corr=%.3f mse=%.4f ratio=%.3f", valueCorr, valueMSE, valueRatio))
        }

        print("Sample quality notes:")
        for note in sampleQualityNotes(metrics: metrics, policyKL: policyKL, valueCorr: valueCorr, valueRatio: valueRatio) {
            print("  \(note)")
        }
    }

    private func computeSampleQualityMetrics(samples: [TrainingSample]) -> SampleQualityMetrics {
        let count = samples.count
        guard count > 0 else {
            return SampleQualityMetrics.empty
        }

        var uniqueHashes = Set<Int>()
        uniqueHashes.reserveCapacity(count)

        var legalSum: Float = 0
        var minLegal = Int.max
        var maxLegal = 0

        var policySumTotal: Float = 0
        var policySumMin: Float = .greatestFiniteMagnitude
        var policySumMax: Float = -.greatestFiniteMagnitude

        var entropySum: Float = 0
        var normalizedEntropySum: Float = 0
        var normalizedEntropyCount: Int = 0
        var maxProbSum: Float = 0
        var peakRatioSum: Float = 0
        var supportRatioSum: Float = 0
        var supportRatioCount: Int = 0

        var valueSum: Float = 0
        var valueSqSum: Float = 0
        var valueMin: Float = .greatestFiniteMagnitude
        var valueMax: Float = -.greatestFiniteMagnitude
        var nearZeroCount = 0
        var absOneCount = 0

        for sample in samples {
            uniqueHashes.insert(sample.stateHash)

            let legalCount = sample.state.legalActions.count
            legalSum += Float(legalCount)
            minLegal = min(minLegal, legalCount)
            maxLegal = max(maxLegal, legalCount)

            let policy = sample.encodedPolicy
            let sum = policy.reduce(0, +)
            policySumTotal += sum
            policySumMin = min(policySumMin, sum)
            policySumMax = max(policySumMax, sum)

            let entropy = policyEntropy(policy)
            entropySum += entropy
            let maxProb = policy.max() ?? 0
            maxProbSum += maxProb
            if legalCount > 0 {
                if legalCount > 1 {
                    normalizedEntropySum += entropy / logValue(Float(legalCount))
                    normalizedEntropyCount += 1
                }
                let support = policy.filter { $0 > 0 }.count
                supportRatioSum += Float(support) / Float(legalCount)
                supportRatioCount += 1
                peakRatioSum += maxProb * Float(legalCount)
            }

            let value = sample.outcome
            valueSum += value
            valueSqSum += value * value
            valueMin = min(valueMin, value)
            valueMax = max(valueMax, value)
            if abs(value) <= 0.1 { nearZeroCount += 1 }
            if abs(value) >= 0.99 { absOneCount += 1 }
        }

        let meanLegal = legalSum / Float(count)
        let meanPolicySum = policySumTotal / Float(count)
        let meanEntropy = entropySum / Float(count)
        let meanNormalizedEntropy = normalizedEntropyCount > 0
            ? normalizedEntropySum / Float(normalizedEntropyCount)
            : 0
        let meanMaxProb = maxProbSum / Float(count)
        let meanPeakRatio = supportRatioCount > 0 ? peakRatioSum / Float(supportRatioCount) : 0
        let meanSupportRatio = supportRatioCount > 0 ? supportRatioSum / Float(supportRatioCount) : 0

        let valueMean = valueSum / Float(count)
        let variance = max(0, valueSqSum / Float(count) - valueMean * valueMean)
        let valueStd = sqrt(variance)

        return SampleQualityMetrics(
            sampleCount: count,
            uniqueStates: uniqueHashes.count,
            meanLegalMoves: meanLegal,
            minLegalMoves: minLegal == Int.max ? 0 : minLegal,
            maxLegalMoves: maxLegal,
            meanPolicySum: meanPolicySum,
            minPolicySum: policySumMin == .greatestFiniteMagnitude ? 0 : policySumMin,
            maxPolicySum: policySumMax == -.greatestFiniteMagnitude ? 0 : policySumMax,
            meanEntropy: meanEntropy,
            meanNormalizedEntropy: meanNormalizedEntropy,
            meanMaxProb: meanMaxProb,
            meanPeakRatio: meanPeakRatio,
            meanSupportRatio: meanSupportRatio,
            valueMean: valueMean,
            valueStd: valueStd,
            valueMin: valueMin == .greatestFiniteMagnitude ? 0 : valueMin,
            valueMax: valueMax == -.greatestFiniteMagnitude ? 0 : valueMax,
            fracNearZero: Float(nearZeroCount) / Float(count),
            fracAbsOne: Float(absOneCount) / Float(count),
            baselineMSE: variance
        )
    }

    private func sampleQualityNotes(
        metrics: SampleQualityMetrics,
        policyKL: Float?,
        valueCorr: Float?,
        valueRatio: Float?
    ) -> [String] {
        var notes: [String] = []

        if metrics.uniqueStateRatio < 0.4 {
            notes.append(String(format: "Diversity is low (%.2f). Many repeats suggest self-play is stuck or too deterministic.", metrics.uniqueStateRatio))
        } else if metrics.uniqueStateRatio < 0.7 {
            notes.append(String(format: "Diversity is moderate (%.2f). Some repeats are expected, but more coverage is better.", metrics.uniqueStateRatio))
        } else {
            notes.append(String(format: "Diversity is high (%.2f). Coverage across positions looks healthy.", metrics.uniqueStateRatio))
        }

        if metrics.meanNormalizedEntropy < 0.3 {
            notes.append(String(format: "Policies are very sharp (norm entropy %.2f). This can mean temperature/noise is low or MCTS is overconfident.", metrics.meanNormalizedEntropy))
        } else if metrics.meanNormalizedEntropy > 0.75 {
            notes.append(String(format: "Policies are near-uniform (norm entropy %.2f). This suggests weak evaluations or too much noise.", metrics.meanNormalizedEntropy))
        } else {
            notes.append(String(format: "Policies are moderately peaked (norm entropy %.2f). Search has a meaningful preference without collapsing.", metrics.meanNormalizedEntropy))
        }

        if metrics.valueStd < 0.2 || metrics.fracNearZero > 0.6 {
            notes.append(String(format: "Value targets are mostly near 0 (std %.2f, |v|<0.1 %.0f%%). This is a weak learning signal.", metrics.valueStd, metrics.fracNearZero * 100))
        } else {
            notes.append(String(format: "Value targets show spread (std %.2f, |v|<0.1 %.0f%%). There is usable signal for learning.", metrics.valueStd, metrics.fracNearZero * 100))
        }

        if metrics.meanSupportRatio < 0.2 {
            notes.append(String(format: "Only %.0f%% of legal moves have non-zero target probability. Temperature=0 or very low visits may be over-pruning.", metrics.meanSupportRatio * 100))
        }

        if let policyKL {
            if policyKL < 0.05 {
                notes.append(String(format: "Policy KL %.3f is very low. MCTS targets are close to the network prior, limiting training signal.", policyKL))
            } else if policyKL > 1.0 {
                notes.append(String(format: "Policy KL %.3f is high. Targets differ a lot from the net, which can be good early but may indicate noisy search.", policyKL))
            } else {
                notes.append(String(format: "Policy KL %.3f indicates MCTS is improving on the net without being wildly off.", policyKL))
            }
        }

        if let valueCorr, let valueRatio {
            if valueRatio > 0.95 || abs(valueCorr) < 0.1 {
                notes.append(String(format: "Value corr %.2f and ratio %.2f show the net is near baseline on these targets (expected early).", valueCorr, valueRatio))
            } else if valueRatio < 0.8 || valueCorr > 0.3 {
                notes.append(String(format: "Value corr %.2f and ratio %.2f suggest the net is learning these targets.", valueCorr, valueRatio))
            } else {
                notes.append(String(format: "Value corr %.2f and ratio %.2f indicate partial alignment with targets.", valueCorr, valueRatio))
            }
        }

        return notes
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

    private func policyEntropy(_ policy: [Float]) -> Float {
        var entropy: Float = 0
        for p in policy where p > 0 {
            entropy -= p * logValue(p)
        }
        return entropy
    }

    private func policyKLDivergence(target: [Float], predicted: [Float]) -> Float {
        let eps: Float = 1e-8
        let count = min(target.count, predicted.count)
        var kl: Float = 0
        for i in 0..<count {
            let t = target[i]
            guard t > 0 else { continue }
            let p = max(predicted[i], eps)
            kl += t * (logValue(t + eps) - logValue(p))
        }
        return kl
    }

    private func logValue(_ value: Float) -> Float {
        Float(log(Double(value)))
    }

    private struct SampleQualityMetrics {
        let sampleCount: Int
        let uniqueStates: Int
        let meanLegalMoves: Float
        let minLegalMoves: Int
        let maxLegalMoves: Int
        let meanPolicySum: Float
        let minPolicySum: Float
        let maxPolicySum: Float
        let meanEntropy: Float
        let meanNormalizedEntropy: Float
        let meanMaxProb: Float
        let meanPeakRatio: Float
        let meanSupportRatio: Float
        let valueMean: Float
        let valueStd: Float
        let valueMin: Float
        let valueMax: Float
        let fracNearZero: Float
        let fracAbsOne: Float
        let baselineMSE: Float

        var uniqueStateRatio: Float {
            guard sampleCount > 0 else { return 0 }
            return Float(uniqueStates) / Float(sampleCount)
        }

        static let empty = SampleQualityMetrics(
            sampleCount: 0,
            uniqueStates: 0,
            meanLegalMoves: 0,
            minLegalMoves: 0,
            maxLegalMoves: 0,
            meanPolicySum: 0,
            minPolicySum: 0,
            maxPolicySum: 0,
            meanEntropy: 0,
            meanNormalizedEntropy: 0,
            meanMaxProb: 0,
            meanPeakRatio: 0,
            meanSupportRatio: 0,
            valueMean: 0,
            valueStd: 0,
            valueMin: 0,
            valueMax: 0,
            fracNearZero: 0,
            fracAbsOne: 0,
            baselineMSE: 0
        )
    }

    private struct ReservoirSampler<T> {
        private(set) var samples: [T] = []
        private let capacity: Int
        private var seen: Int = 0
        private var rng: SeededGenerator

        init(capacity: Int, seed: UInt64) {
            self.capacity = max(0, capacity)
            self.rng = SeededGenerator(seed: seed)
        }

        mutating func add(contentsOf newSamples: [T]) {
            for sample in newSamples {
                add(sample)
            }
        }

        mutating func add(_ sample: T) {
            guard capacity > 0 else { return }
            seen += 1
            if samples.count < capacity {
                samples.append(sample)
                return
            }
            let index = Int.random(in: 0..<seen, using: &rng)
            if index < capacity {
                samples[index] = sample
            }
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

    private func evaluationPhase(iteration: Int) async {
        print("Beginning evaluation...")
        model.train(false)
        bestModel.train(false)
        let result = await playMatches(
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
    ) async -> (winsCurrent: Int, winsBest: Int, draws: Int) {
        let currentEvaluator = BatchedEvaluator(
            net: current,
            maxBatchSize: config.selfPlayBatchSize,
            timeoutMicroseconds: config.batchTimeoutMicroseconds
        )
        let bestEvaluator = BatchedEvaluator(
            net: best,
            maxBatchSize: config.selfPlayBatchSize,
            timeoutMicroseconds: config.batchTimeoutMicroseconds
        )
        let mctsIterations = config.MCTSSimulations
        let baseSeed: UInt64 = 99999

        let results: [(winner: Player?, currentPlaysFirst: Bool)] = await withTaskGroup(
            of: (Int, Player?).self
        ) { group in
            for game in 0 ..< games {
                let gameSeed = baseSeed &+ UInt64(game)
                let currentPlaysFirst = (game % 2 == 0)
                group.addTask {
                    let winner = await SantoriniTrainer.playEvalGame(
                        currentEvaluator: currentEvaluator,
                        bestEvaluator: bestEvaluator,
                        currentPlaysFirst: currentPlaysFirst,
                        iterations: mctsIterations,
                        seed: gameSeed
                    )
                    return (game, winner)
                }
            }

            var collected: [(Int, Player?, Bool)] = []
            collected.reserveCapacity(games)
            for await (game, winner) in group {
                collected.append((game, winner, game % 2 == 0))
            }
            collected.sort { $0.0 < $1.0 }
            return collected.map { (winner: $0.1, currentPlaysFirst: $0.2) }
        }

        var winsCurrent = 0
        var winsBest = 0
        var draws = 0
        for result in results {
            guard let winner = result.winner else {
                draws += 1
                continue
            }
            if result.currentPlaysFirst {
                winsCurrent += (winner == .one) ? 1 : 0
                winsBest += (winner == .two) ? 1 : 0
            } else {
                winsCurrent += (winner == .two) ? 1 : 0
                winsBest += (winner == .one) ? 1 : 0
            }
        }
        return (winsCurrent, winsBest, draws)
    }

    private static func playEvalGame(
        currentEvaluator: BatchedEvaluator,
        bestEvaluator: BatchedEvaluator,
        currentPlaysFirst: Bool,
        iterations: Int,
        seed: UInt64
    ) async -> Player? {
        var state = Santorini.GameState()
        var rng = SeededGenerator(seed: seed)

        while !state.isOver {
            let useCurrentNet: Bool
            if currentPlaysFirst {
                useCurrentNet = (state.turn == .one)
            } else {
                useCurrentNet = (state.turn == .two)
            }
            let evaluator = useCurrentNet ? currentEvaluator : bestEvaluator
            let moveSeed = rng.next()

            let evaluateClosure: @Sendable (Santorini.GameState) async -> (policy: [Float], value: Float) = { gameState in
                let encoded = gameState.encoded()
                return await evaluator.evaluate(encodedState: encoded)
            }

            let (action, _) = await asyncMCTS(
                rootState: state,
                evaluate: evaluateClosure,
                iterations: iterations,
                temperature: 0.0,
                seed: moveSeed
            )
            guard let action else { return nil }
            state = state.applying(move: action)
        }
        return state.winner
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
