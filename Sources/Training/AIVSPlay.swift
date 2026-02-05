//
//  AIVSPlay.swift
//  AlphaSantorini
//
//  Created by Tomás Ruiz-López on 1/16/26.
//

import Santorini
import NeuralNetwork
import MCTS

class AIVSPlay {
    func play(
        player1: SantoriniNet,
        player2: SantoriniNet,
        iterations: Int,
        maxMoves: Int? = nil
    ) -> Player? {
        var state = GameState()
        var moves = 0

        while !state.isOver {
            if let maxMoves, moves >= maxMoves {
                return nil
            }
            let currentNetwork = state.turn == .one ? player1 : player2

            let (action, _) = mcts(
                rootState: state,
                evaluator: currentNetwork,
                iterations: iterations,
                temperature: 0.0
            )
            if let action {
                state = state.applying(move: action)
                moves += 1
            } else {
                return nil
            }
        }

        return state.winner
    }
}
