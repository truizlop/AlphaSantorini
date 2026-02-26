//
//  AsyncMCTS.swift
//  AlphaSantorini
//
//  Created by Tomás Ruiz-López on 2/26/26.
//

import Foundation

public func asyncMCTS<State: GameState>(
    rootState: State,
    evaluate: @Sendable (State) async -> (policy: [Float], value: Float),
    iterations: Int,
    temperature: Float,
    seed: UInt64,
    explorationConstant: Float = 1.5,
    noise: DirichletNoise? = nil
) async -> (bestMove: State.Move?, distribution: [State.Move: Float]) {
    let result = await runAsyncMctsSingle(
        rootState: rootState,
        evaluate: evaluate,
        iterations: iterations,
        temperature: temperature,
        seed: seed,
        explorationConstant: explorationConstant,
        noise: noise
    )
    return (result.bestMove, result.distribution)
}

public func asyncMCTSWithRootValue<State: GameState>(
    rootState: State,
    evaluate: @Sendable (State) async -> (policy: [Float], value: Float),
    iterations: Int,
    temperature: Float,
    seed: UInt64,
    explorationConstant: Float = 1.5,
    noise: DirichletNoise? = nil
) async -> (bestMove: State.Move?, distribution: [State.Move: Float], rootValue: Float) {
    let result = await runAsyncMctsSingle(
        rootState: rootState,
        evaluate: evaluate,
        iterations: iterations,
        temperature: temperature,
        seed: seed,
        explorationConstant: explorationConstant,
        noise: noise
    )
    return (result.bestMove, result.distribution, result.root.meanValue)
}

private func runAsyncMctsSingle<State: GameState>(
    rootState: State,
    evaluate: @Sendable (State) async -> (policy: [Float], value: Float),
    iterations: Int,
    temperature: Float,
    seed: UInt64,
    explorationConstant: Float,
    noise: DirichletNoise?
) async -> (root: MCTSNode<State>, bestMove: State.Move?, distribution: [State.Move: Float]) {
    var rng = SeedableRNG(seed: seed)

    let root = MCTSNode(
        state: rootState,
        move: nil,
        prior: 0
    )

    // Expand root using async evaluator
    if !rootState.isTerminal {
        let (policy, value) = await evaluate(rootState)
        root.expand(with: policy, value: value)
    }

    if let noise {
        root.addDirichletNoise(noise, rng: &rng)
    }

    for _ in 0 ..< iterations {
        var node = root

        while node.isExpanded && !node.state.isTerminal {
            guard let best = node.bestChild(explorationConstant: explorationConstant) else { break }
            node = best
        }

        let value: Float
        if node.state.isTerminal {
            value = node.state.terminalValue
        } else {
            let (policy, v) = await evaluate(node.state)
            value = node.expand(with: policy, value: v)
        }

        node.backpropagate(value: value)
    }

    return (
        root,
        bestMove(from: root, temperature: temperature, rng: &rng),
        visitDistribution(from: root, temperature: temperature)
    )
}

private struct SeedableRNG: RandomNumberGenerator {
    private var state: UInt64

    init(seed: UInt64) {
        self.state = seed
    }

    mutating func next() -> UInt64 {
        state = state &* 6364136223846793005 &+ 1442695040888963407
        return state
    }
}
