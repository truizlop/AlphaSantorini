//
//  TrainingSample.swift
//  AlphaSantorini
//
//  Created by Tomás Ruiz-López on 1/15/26.
//

import Santorini

public struct TrainingSample: @unchecked Sendable {
    public var state: GameState
    public var action: Action
    public var policy: [Action: Float]
    public var outcome: Float

    init(
        state: GameState,
        action: Action,
        policy: [Action: Float],
        outcome: Float
    ) {
        self.state = state
        self.action = action
        self.policy = policy
        self.outcome = outcome
    }

    var encodedPolicy: [Float] {
        var encoded = Array<Float>(repeating: 0, count: Action.total)
        for (action, probability) in policy {
            let index = action.encoded()
            encoded[index] = probability
        }
        return encoded
    }
}
