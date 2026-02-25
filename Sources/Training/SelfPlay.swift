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

public struct SelfPlayResult: @unchecked Sendable {
    public let samples: [TrainingSample]
    public let wasTruncated: Bool
    public let moveCount: Int
}

public class SelfPlay: @unchecked Sendable {
    public init() {}
    
    public func run(
        initialState: Santorini.GameState = GameState(),
        evaluator: SantoriniNet,
        iterations: Int,
        noise: DirichletNoise?,
        batchSize: Int,
        useTemperature: Bool = true,
        valueTargetStrategy: ValueTargetStrategy = .terminalOutcome,
        maxMoves: Int? = nil
    ) -> [TrainingSample] {
        let result = runWithDiagnostics(
            initialState: initialState,
            evaluator: evaluator,
            iterations: iterations,
            noise: noise,
            batchSize: batchSize,
            useTemperature: useTemperature,
            valueTargetStrategy: valueTargetStrategy,
            maxMoves: maxMoves
        )
        return result.wasTruncated ? [] : result.samples
    }

    public func runWithDiagnostics(
        initialState: Santorini.GameState = GameState(),
        evaluator: SantoriniNet,
        iterations: Int,
        noise: DirichletNoise?,
        batchSize: Int,
        useTemperature: Bool = true,
        valueTargetStrategy: ValueTargetStrategy = .terminalOutcome,
        maxMoves: Int? = nil,
    ) -> SelfPlayResult {
        var rng = SystemRandomNumberGenerator()
        return runWithDiagnostics(
            initialState: initialState,
            evaluator: evaluator,
            iterations: iterations,
            noise: noise,
            batchSize: batchSize,
            useTemperature: useTemperature,
            valueTargetStrategy: valueTargetStrategy,
            maxMoves: maxMoves,
            rng: &rng
        )
    }

    public func runWithDiagnostics<R: RandomNumberGenerator>(
        initialState: Santorini.GameState = GameState(),
        evaluator: SantoriniNet,
        iterations: Int,
        noise: DirichletNoise?,
        batchSize: Int,
        useTemperature: Bool = true,
        valueTargetStrategy: ValueTargetStrategy = .terminalOutcome,
        maxMoves: Int? = nil,
        rng: inout R
    ) -> SelfPlayResult {
        var state = initialState
        var history: [(Santorini.GameState, Action, Policy, Float?)] = []
        var move = 0
        var wasTruncated = false
        let totalMoves = maxMoves ?? .max

        _ = batchSize

        func searchStep<E: PolicyValueNetwork>(
            with mctsEvaluator: E,
            state: Santorini.GameState,
            move: Int
        ) -> (bestAction: Action?, policy: Policy, rootValue: Float?) where E.State == Santorini.GameState {
            switch valueTargetStrategy {
            case .mctsRootValue:
                let result = mctsWithRootValue(
                    rootState: state,
                    evaluator: mctsEvaluator,
                    iterations: iterations,
                    temperature: temperature(for: move, enabled: useTemperature),
                    rng: &rng,
                    noise: noise
                )
                return (result.bestMove, result.distribution, result.rootValue)
            case .terminalOutcome:
                let result = mcts(
                    rootState: state,
                    evaluator: mctsEvaluator,
                    iterations: iterations,
                    temperature: temperature(for: move, enabled: useTemperature),
                    rng: &rng,
                    noise: noise
                )
                return (result.bestMove, result.distribution, nil)
            }
        }

        while !state.isOver && move < totalMoves {
            let bestAction: Action?
            let policy: Policy
            let rootValue: Float?
            let legalActions = state.legalActions
            if legalActions.count == 1, let forced = legalActions.first {
                bestAction = forced
                policy = [forced: 1.0]
                if valueTargetStrategy == .mctsRootValue {
                    let value = evaluator.evaluate(state: state).value
                    rootValue = value.isFinite ? value : nil
                } else {
                    rootValue = nil
                }
            } else {
                let step = searchStep(with: evaluator, state: state, move: move)
                bestAction = step.bestAction
                policy = step.policy
                rootValue = step.rootValue
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
        return move < 30 ? 1.0 : 0.0
    }
}
