//
//  SantoriniGameState.swift
//  AlphaSantorini
//
//  Created by Tomás Ruiz-López on 1/15/26.
//

import Santorini
import MCTS

extension Santorini.GameState: MCTS.GameState {
    public typealias Move = Action

    public var isTerminal: Bool {
        self.isOver
    }
    
    public var terminalValue: Float {
        guard self.isOver else { return 0 }
        if let winner {
            return winner == turn ? 1 : -1
        } else {
            return 0
        }
    }
    
    public func legalMoves() -> [Action] {
        self.legalActions
    }

    public func applying(move: Action) -> Santorini.GameState {
        var newState = self
        switch move {
        case .placement(let placement):
            newState.placement(placement)
        case .move(let move):
            newState.play(move: move)
        }
        return newState
    }
}
