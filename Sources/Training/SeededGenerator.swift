//
//  SeededGenerator.swift
//  AlphaSantorini
//
//  Created by Tomás Ruiz-López on 2/26/26.
//

public struct SeededGenerator: RandomNumberGenerator, Sendable {
    private var state: UInt64

    public init(seed: UInt64) {
        self.state = seed
    }

    public mutating func next() -> UInt64 {
        state = state &* 6364136223846793005 &+ 1442695040888963407
        return state
    }
}
