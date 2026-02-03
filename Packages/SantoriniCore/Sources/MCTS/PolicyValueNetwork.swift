//
//  PolicyValueNetwork.swift
//  AlphaSantorini
//
//  Created by Tomás Ruiz-López on 1/15/26.
//

public protocol PolicyValueNetwork {
    associatedtype State: GameState

    /// Returns a flat policy array indexed by move encoding.
    /// The array length must match the action encoding size for the game.
    func evaluate(state: State) -> (policy: [Float], value: Float)
}

public protocol BatchPolicyValueNetwork {
    associatedtype State: GameState

    /// Returns flat policy arrays indexed by move encoding.
    /// The array length must match the action encoding size for the game.
    func evaluate(states: [State]) -> (policies: [[Float]], values: [Float])
}
