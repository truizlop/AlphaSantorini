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

public class SelfPlay: @unchecked Sendable {
    public init() {}
    
    public func run(
        evaluator: SantoriniNet,
        iterations: Int,
        noise: DirichletNoise?
    ) -> [TrainingSample] {
        var state = GameState()
        var history: [(Santorini.GameState, Action, Policy)] = []
        var move = 0

        while !state.isOver {
            let (bestAction, policy) = mcts(
                rootState: state,
                evaluator: evaluator,
                iterations: iterations,
                temperature: temperature(for: move, isTraining: noise != nil),
                noise: noise,
            )

            if let bestAction {
                history.append((state, bestAction, policy))
                state = state.applying(move: bestAction)
                move += 1
            } else {
                assertionFailure("No best action found yet the game is not over")
            }
        }

        let terminalWinner = state.winner
        return history.map { item in
            let adjustedOutcome: Float
            if let terminalWinner {
                adjustedOutcome = terminalWinner == item.0.turn ? 1 : -1
            } else {
                adjustedOutcome = 0
            }
            return TrainingSample(
                state: item.0,
                action: item.1,
                policy: item.2,
                outcome: adjustedOutcome
            )
        }
    }

    private func temperature(
        for move: Int,
        isTraining: Bool
    ) -> Float {
        guard isTraining else { return 0.0 }
        return move < 20 ? 1.5 : 0.0
    }
}
