//
//  MCTS.swift
//  AlphaSantorini
//
//  Created by Tomás Ruiz-López on 1/15/26.
//

public func mcts<State: GameState, Evaluator: PolicyValueNetwork>(
    rootState: State,
    evaluator: Evaluator,
    iterations: Int,
    explorationConstant: Float = 1.5,
    isTraining: Bool = false
) -> State.Move? where Evaluator.State == State {
    let root = MCTSNode(
        state: rootState,
        move: nil,
        prior: 0
    )
    root.expand(using: evaluator)

    if isTraining {
        root.addDirichletNoise()
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

    return root.children.max { a, b in
        a.visits < b.visits
    }?.move
}
