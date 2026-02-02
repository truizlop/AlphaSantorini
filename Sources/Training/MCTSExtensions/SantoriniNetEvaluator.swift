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

    public func evaluate(state: Santorini.GameState) -> (policy: [Float], value: Float) {
        let input = state.encoded()
        return evaluate(input)
    }
}
