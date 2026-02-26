//
//  BatchedSelfPlay.swift
//  AlphaSantorini
//
//  Created by Tomás Ruiz-López on 2/26/26.
//

import Santorini
import MCTS
import NeuralNetwork

extension SelfPlay {
    public func runBatched(
        initialState: Santorini.GameState = GameState(),
        evaluator: BatchedEvaluator,
        net: SantoriniNet,
        iterations: Int,
        noise: DirichletNoise?,
        useTemperature: Bool = true,
        valueTargetStrategy: ValueTargetStrategy = .terminalOutcome,
        maxMoves: Int? = nil,
        seed: UInt64
    ) async -> SelfPlayResult {
        var state = initialState
        var history: [(Santorini.GameState, Action, Policy, Float?)] = []
        var move = 0
        var wasTruncated = false
        let totalMoves = maxMoves ?? .max
        var rng = SeededGenerator(seed: seed)

        while !state.isOver && move < totalMoves {
            let bestAction: Action?
            let policy: Policy
            let rootValue: Float?
            let legalActions = state.legalActions
            if legalActions.count == 1, let forced = legalActions.first {
                bestAction = forced
                policy = [forced: 1.0]
                if valueTargetStrategy == .mctsRootValue {
                    let encoded = state.encoded()
                    let result = await evaluator.evaluate(encodedState: encoded)
                    rootValue = result.value.isFinite ? result.value : nil
                } else {
                    rootValue = nil
                }
            } else {
                let mctsTemperature = temperature(for: move, enabled: useTemperature)
                let mctsSeed = rng.next()

                let evaluateClosure: @Sendable (Santorini.GameState) async -> (policy: [Float], value: Float) = { gameState in
                    let encoded = gameState.encoded()
                    return await evaluator.evaluate(encodedState: encoded)
                }

                switch valueTargetStrategy {
                case .mctsRootValue:
                    let result = await asyncMCTSWithRootValue(
                        rootState: state,
                        evaluate: evaluateClosure,
                        iterations: iterations,
                        temperature: mctsTemperature,
                        seed: mctsSeed,
                        noise: noise
                    )
                    bestAction = result.bestMove
                    policy = result.distribution
                    rootValue = result.rootValue
                case .terminalOutcome:
                    let result = await asyncMCTS(
                        rootState: state,
                        evaluate: evaluateClosure,
                        iterations: iterations,
                        temperature: mctsTemperature,
                        seed: mctsSeed,
                        noise: noise
                    )
                    bestAction = result.bestMove
                    policy = result.distribution
                    rootValue = nil
                }
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
}
