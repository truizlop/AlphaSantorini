//
//  SantoriniNetEvaluator.swift
//  AlphaSantorini
//
//  Created by Tomás Ruiz-López on 1/15/26.
//

import Santorini
import NeuralNetwork
import MCTS

extension SantoriniNet: PolicyValueNetwork {
    public typealias State = Santorini.GameState

    public func evaluate(state: Santorini.GameState) -> (policy: [Santorini.Action: Float], value: Float) {
        let input = state.encoded()
        let (policy, value) = evaluate(input)
        var dict: [Action: Float] = [:]
        (0 ..< policy.count).forEach { encoding in
            guard let action = Action.from(encoding: encoding) else { return }
            dict[action] = policy[encoding]
        }
        return (dict, value)
    }
}
