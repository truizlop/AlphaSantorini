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
    let root = MCTSNode(
        state: rootState,
        move: nil,
        prior: 0
    )
    root.expand(using: evaluator)

    if let noise {
        root.addDirichletNoise(noise)
    }

    for _ in 0 ..< iterations {
        var node = root

        while node.isExpanded && !node.state.isTerminal {
            guard let best = node.bestChild(explorationConstant: explorationConstant) else { break }
            node = best
        }

        let value: Float = if node.state.isTerminal {
            node.state.terminalValue
        } else {
            node.expand(using: evaluator)
        }

        node.backpropagate(value: value)
    }

    return (bestMove(from: root, temperature: temperature), visitDistribution(from: root))
}

private func bestMove<State>(
    from node: MCTSNode<State>,
    temperature: Float
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
            return node.children.randomElement()?.move
        }
        let random = Float.random(in: 0 ..< 1)
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

    let random = Float.random(in: 0 ..< 1)
    var cumulative: Float = 0.0

    for (i, probability) in probabilities.enumerated() {
        cumulative += probability
        if random < cumulative {
            return node.children[i].move
        }
    }

    return nil
}

private func visitDistribution<State: GameState>(
    from node: MCTSNode<State>
) -> [State.Move: Float] {
    let totalVisits = Float(node.children.map(\.visits).reduce(0, +))

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
    for child in node.children {
        if let move = child.move {
            distribution[move] = Float(child.visits) / totalVisits
        }
    }
    return distribution
}
