//
//  GameState.swift
//  AlphaSantorini
//
//  Created by Tomás Ruiz-López on 1/15/26.
//

public protocol GameState {
    associatedtype Move: Hashable

    var isTerminal: Bool { get }
    // Only valid if isTerminal == true
    var terminalValue: Float { get }
    func legalMoves() -> [Move]
    func applying(move: Move) -> Self
}
