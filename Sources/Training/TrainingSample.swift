//
//  TrainingSample.swift
//  AlphaSantorini
//
//  Created by Tomás Ruiz-López on 1/15/26.
//

import Santorini

public struct TrainingSample: @unchecked Sendable {
    public var state: GameState {
        didSet {
            stateHash = Self.hashState(state)
        }
    }
    public var action: Action
    public var policy: [Action: Float] {
        didSet {
            encodedPolicy = Self.encodePolicy(policy)
        }
    }
    public var outcome: Float
    public private(set) var encodedPolicy: [Float]
    public private(set) var stateHash: Int

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
        self.encodedPolicy = Self.encodePolicy(policy)
        self.stateHash = Self.hashState(state)
    }

    private static func encodePolicy(_ policy: [Action: Float]) -> [Float] {
        var encoded = Array<Float>(repeating: 0, count: Action.total)
        for (action, probability) in policy {
            let index = action.encoded()
            encoded[index] = probability
        }
        return encoded
    }

    private static func hashState(_ state: GameState) -> Int {
        let encoded = state.encoded()
        var hasher = Hasher()
        for value in encoded {
            hasher.combine(value.bitPattern)
        }
        return hasher.finalize()
    }
}
