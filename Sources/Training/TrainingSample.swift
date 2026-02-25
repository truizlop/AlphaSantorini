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

    private static func transformPolicy(
        _ policy: [Action: Float],
        by symmetry: BoardSymmetry
    ) -> [Action: Float] {
        var transformed: [Action: Float] = [:]
        transformed.reserveCapacity(policy.count)
        for (action, probability) in policy {
            let transformedAction = action.transformed(by: symmetry)
            transformed[transformedAction, default: 0] += probability
        }
        return transformed
    }

    private static func hashState(_ state: GameState) -> Int {
        let encoded = state.encoded()
        var hasher = Hasher()
        for row in encoded {
            for cell in row {
                for value in cell {
                    hasher.combine(value.bitPattern)
                }
            }
        }
        return hasher.finalize()
    }

    func transformed(by symmetry: BoardSymmetry) -> TrainingSample {
        TrainingSample(
            state: state.transformed(by: symmetry),
            action: action.transformed(by: symmetry),
            policy: Self.transformPolicy(policy, by: symmetry),
            outcome: outcome
        )
    }

    func augmentedBySymmetry() -> [TrainingSample] {
        BoardSymmetry.allCases.map { transformed(by: $0) }
    }
}
