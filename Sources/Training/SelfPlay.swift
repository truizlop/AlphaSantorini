//
//  SelfPlay.swift
//  AlphaSantorini
//
//  Created by Tomás Ruiz-López on 1/15/26.
//

import Santorini
import MCTS
import NeuralNetwork

typealias Policy = [Action: Float]

public struct SelfPlayResult {
    public let samples: [TrainingSample]
    public let wasTruncated: Bool
    public let moveCount: Int
}

public class SelfPlay: @unchecked Sendable {
    public init() {}
    
    public func run(
        evaluator: SantoriniNet,
        iterations: Int,
        noise: DirichletNoise?,
        batchSize: Int,
        useTemperature: Bool = true,
        valueTargetStrategy: ValueTargetStrategy = .terminalOutcome
    ) -> [TrainingSample] {
        let result = runWithDiagnostics(
            evaluator: evaluator,
            iterations: iterations,
            noise: noise,
            batchSize: batchSize,
            useTemperature: useTemperature,
            valueTargetStrategy: valueTargetStrategy
        )
        return result.wasTruncated ? [] : result.samples
    }

    public func runWithDiagnostics(
        evaluator: SantoriniNet,
        iterations: Int,
        noise: DirichletNoise?,
        batchSize: Int,
        useTemperature: Bool = true,
        valueTargetStrategy: ValueTargetStrategy = .terminalOutcome
    ) -> SelfPlayResult {
        var rng = SystemRandomNumberGenerator()
        return runWithDiagnostics(
            evaluator: evaluator,
            iterations: iterations,
            noise: noise,
            batchSize: batchSize,
            useTemperature: useTemperature,
            valueTargetStrategy: valueTargetStrategy,
            rng: &rng
        )
    }

    public func runWithDiagnostics<R: RandomNumberGenerator>(
        evaluator: SantoriniNet,
        iterations: Int,
        noise: DirichletNoise?,
        batchSize: Int,
        useTemperature: Bool = true,
        valueTargetStrategy: ValueTargetStrategy = .terminalOutcome,
        rng: inout R
    ) -> SelfPlayResult {
        var state = GameState()
        var history: [(Santorini.GameState, Action, Policy, Float?)] = []
        var move = 0
        var wasTruncated = false

        while !state.isOver {
            let bestAction: Action?
            let policy: Policy
            let rootValue: Float?
            switch valueTargetStrategy {
            case .mctsRootValue:
                let result = mctsBatchedWithRootValue(
                    rootState: state,
                    evaluator: evaluator,
                    iterations: iterations,
                    temperature: temperature(for: move, enabled: useTemperature),
                    rng: &rng,
                    noise: noise,
                    batchSize: batchSize
                )
                bestAction = result.bestMove
                policy = result.distribution
                rootValue = result.rootValue
            case .terminalOutcome:
                let result = mctsBatched(
                    rootState: state,
                    evaluator: evaluator,
                    iterations: iterations,
                    temperature: temperature(for: move, enabled: useTemperature),
                    rng: &rng,
                    noise: noise,
                    batchSize: batchSize
                )
                bestAction = result.bestMove
                policy = result.distribution
                rootValue = nil
            }

            if let bestAction {
                let normalizedPolicy = normalizePolicy(policy, move: move)
                history.append((state, bestAction, normalizedPolicy, rootValue))
                state = state.applying(move: bestAction)
                move += 1
            } else {
                assertionFailure("No best action found yet the game is not over")
                wasTruncated = true
                break
            }
        }

        let terminalWinner = state.winner
        let samples = buildSamples(
            from: history,
            terminalWinner: terminalWinner,
            valueTargetStrategy: valueTargetStrategy
        )
        return SelfPlayResult(samples: samples, wasTruncated: wasTruncated, moveCount: move)
    }

    func buildSamples(
        from history: [(Santorini.GameState, Action, Policy, Float?)],
        terminalWinner: Player?,
        valueTargetStrategy: ValueTargetStrategy
    ) -> [TrainingSample] {
        history.map { item in
            let adjustedOutcome: Float
            switch valueTargetStrategy {
            case .terminalOutcome:
                if let terminalWinner {
                    adjustedOutcome = terminalWinner == item.0.turn ? 1 : -1
                } else {
                    adjustedOutcome = 0
                }
            case .mctsRootValue:
                if let rootValue = item.3, rootValue.isFinite {
                    adjustedOutcome = min(1, max(-1, rootValue))
                } else if let terminalWinner {
                    adjustedOutcome = terminalWinner == item.0.turn ? 1 : -1
                } else {
                    adjustedOutcome = 0
                }
            }
            return TrainingSample(
                state: item.0,
                action: item.1,
                policy: item.2,
                outcome: adjustedOutcome
            )
        }
    }

    private func normalizePolicy(_ policy: Policy, move: Int) -> Policy {
        let sum = policy.values.reduce(0, +)
        guard sum.isFinite else {
            print("⚠️ Policy sum is not finite at move \(move).")
            return policy
        }
        if sum <= 0 {
            let count = policy.count
            guard count > 0 else { return policy }
            print("⚠️ Policy sum is \(sum) at move \(move); using uniform distribution.")
            let uniform = 1.0 / Float(count)
            return Dictionary(uniqueKeysWithValues: policy.keys.map { ($0, uniform) })
        }
        let delta = abs(sum - 1.0)
        if delta > 1e-3 {
            print("⚠️ Policy not normalized (sum=\(String(format: "%.5f", sum))) at move \(move). Renormalizing.")
            var normalized: Policy = [:]
            normalized.reserveCapacity(policy.count)
            for (action, probability) in policy {
                normalized[action] = probability / sum
            }
            return normalized
        }
        return policy
    }

    private func temperature(
        for move: Int,
        enabled: Bool
    ) -> Float {
        guard enabled else { return 0.0 }
        return move < 30 ? 1.5 : 0.0
    }
}
