//
//  PolicyValueNetwork.swift
//  AlphaSantorini
//
//  Created by Tomás Ruiz-López on 1/15/26.
//

public protocol PolicyValueNetwork {
    associatedtype State: GameState

    func evaluate(state: State) -> (policy: [State.Move: Float], value: Float)
}
