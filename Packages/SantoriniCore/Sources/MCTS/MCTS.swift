//
//  MCTS.swift
//  AlphaSantorini
//
//  Created by Tomás Ruiz-López on 1/15/26.
//

import Foundation

public func mcts<State: GameState, Evaluator: PolicyValueNetwork>(
    rootState: State,
    evaluator: Evaluator,
    iterations: Int,
    temperature: Float,
    explorationConstant: Float = 1.5,
    noise: DirichletNoise? = nil
) -> (bestMove: State.Move?, distribution: [State.Move: Float]) where Evaluator.State == State {
    var rng = SystemRandomNumberGenerator()
    return mcts(
        rootState: rootState,
        evaluator: evaluator,
        iterations: iterations,
        temperature: temperature,
        rng: &rng,
        explorationConstant: explorationConstant,
        noise: noise
    )
}

public func mcts<State: GameState, Evaluator: PolicyValueNetwork, R: RandomNumberGenerator>(
    rootState: State,
    evaluator: Evaluator,
    iterations: Int,
    temperature: Float,
    rng: inout R,
    explorationConstant: Float = 1.5,
    noise: DirichletNoise? = nil
) -> (bestMove: State.Move?, distribution: [State.Move: Float]) where Evaluator.State == State {
    let result = runMctsSingle(
        rootState: rootState,
        evaluator: evaluator,
        iterations: iterations,
        temperature: temperature,
        rng: &rng,
        explorationConstant: explorationConstant,
        noise: noise
    )
    return (result.bestMove, result.distribution)
}

public func mctsWithRootValue<State: GameState, Evaluator: PolicyValueNetwork>(
    rootState: State,
    evaluator: Evaluator,
    iterations: Int,
    temperature: Float,
    explorationConstant: Float = 1.5,
    noise: DirichletNoise? = nil
) -> (bestMove: State.Move?, distribution: [State.Move: Float], rootValue: Float) where Evaluator.State == State {
    var rng = SystemRandomNumberGenerator()
    return mctsWithRootValue(
        rootState: rootState,
        evaluator: evaluator,
        iterations: iterations,
        temperature: temperature,
        rng: &rng,
        explorationConstant: explorationConstant,
        noise: noise
    )
}

public func mctsWithRootValue<State: GameState, Evaluator: PolicyValueNetwork, R: RandomNumberGenerator>(
    rootState: State,
    evaluator: Evaluator,
    iterations: Int,
    temperature: Float,
    rng: inout R,
    explorationConstant: Float = 1.5,
    noise: DirichletNoise? = nil
) -> (bestMove: State.Move?, distribution: [State.Move: Float], rootValue: Float) where Evaluator.State == State {
    let result = runMctsSingle(
        rootState: rootState,
        evaluator: evaluator,
        iterations: iterations,
        temperature: temperature,
        rng: &rng,
        explorationConstant: explorationConstant,
        noise: noise
    )
    return (result.bestMove, result.distribution, result.root.meanValue)
}

private func runMctsSingle<State: GameState, Evaluator: PolicyValueNetwork, R: RandomNumberGenerator>(
    rootState: State,
    evaluator: Evaluator,
    iterations: Int,
    temperature: Float,
    rng: inout R,
    explorationConstant: Float,
    noise: DirichletNoise?
) -> (root: MCTSNode<State>, bestMove: State.Move?, distribution: [State.Move: Float]) where Evaluator.State == State {
    let root = MCTSNode(
        state: rootState,
        move: nil,
        prior: 0
    )
    root.expand(using: evaluator)

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
            value = node.expand(using: evaluator)
        }

        node.backpropagate(value: value)
    }

    return (
        root,
        bestMove(from: root, temperature: temperature, rng: &rng),
        visitDistribution(from: root, temperature: temperature)
    )
}

func bestMove<State, R: RandomNumberGenerator>(
    from node: MCTSNode<State>,
    temperature: Float,
    rng: inout R
) -> State.Move? {
    guard temperature != 0 else {
        return node.children.max { a, b in
            a.visits < b.visits
        }?.move
    }

    let visits = node.children.map { Float($0.visits) }

    let adjusted = visits.map { pow($0, 1.0 / temperature) }
    let total = adjusted.reduce(0, +)
    if total == 0 {
        let priors = node.children.map { max(0, $0.prior) }
        let priorTotal = priors.reduce(0, +)
        guard priorTotal > 0 else {
            guard !node.children.isEmpty else { return nil }
            let index = Int.random(in: 0 ..< node.children.count, using: &rng)
            return node.children[index].move
        }
        let random = Float.random(in: 0 ..< 1, using: &rng)
        var cumulative: Float = 0.0
        for (i, prior) in priors.enumerated() {
            cumulative += prior / priorTotal
            if random < cumulative {
                return node.children[i].move
            }
        }
        return node.children.last?.move
    }
    let probabilities = adjusted.map { $0 / total }

    let random = Float.random(in: 0 ..< 1, using: &rng)
    var cumulative: Float = 0.0

    for (i, probability) in probabilities.enumerated() {
        cumulative += probability
        if random < cumulative {
            return node.children[i].move
        }
    }

    return nil
}

func visitDistribution<State: GameState>(
    from node: MCTSNode<State>,
    temperature: Float
) -> [State.Move: Float] {
    let visits = node.children.map { Float($0.visits) }
    let totalVisits = visits.reduce(0, +)

    var distribution: [State.Move: Float] = [:]
    guard totalVisits > 0 else {
        let priors = node.children.map { max(0, $0.prior) }
        let priorTotal = priors.reduce(0, +)
        if priorTotal > 0 {
            for (index, child) in node.children.enumerated() {
                if let move = child.move {
                    distribution[move] = priors[index] / priorTotal
                }
            }
            return distribution
        }
        let count = Float(node.children.count)
        guard count > 0 else { return distribution }
        let uniform = 1.0 / count
        for child in node.children {
            if let move = child.move {
                distribution[move] = uniform
            }
        }
        return distribution
    }
    if temperature == 0 {
        let maxVisits = visits.max() ?? 0
        let maxCount = visits.filter { $0 == maxVisits }.count
        let weight = maxCount > 0 ? 1.0 / Float(maxCount) : 0
        for (index, child) in node.children.enumerated() {
            if let move = child.move {
                distribution[move] = visits[index] == maxVisits ? weight : 0
            }
        }
        return distribution
    }

    let adjusted = visits.map { pow($0, 1.0 / temperature) }
    let total = adjusted.reduce(0, +)
    if total > 0 {
        for (index, child) in node.children.enumerated() {
            if let move = child.move {
                distribution[move] = adjusted[index] / total
            }
        }
    }
    return distribution
}
