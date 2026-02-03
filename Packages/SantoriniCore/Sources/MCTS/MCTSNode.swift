//
//  MCTSNode.swift
//  AlphaSantorini
//
//  Created by Tomás Ruiz-López on 1/15/26.
//

import Foundation

public class MCTSNode<State: GameState> {
    let state: State
    let move: State.Move?
    weak var parent: MCTSNode<State>?
    var children: [MCTSNode<State>] = []
    private lazy var cachedLegalMoves: [State.Move] = state.legalMoves()

    var visits: Int = 0
    var totalValue: Float = 0
    var prior: Float

    var meanValue: Float {
        visits > 0 ? totalValue / Float(visits) : 0
    }

    init(
        state: State,
        move: State.Move?,
        parent: MCTSNode<State>? = nil,
        prior: Float
    ) {
        self.state = state
        self.move = move
        self.parent = parent
        self.prior = prior
    }

    var isExpanded: Bool {
        !children.isEmpty || state.isTerminal
    }

    func puctScore(explorationConstant: Float) -> Float {
        let parentVisits = parent?.visits ?? 1
        let q = meanValue
        let u = explorationConstant * prior * sqrt(Float(parentVisits)) / Float(1 + visits)
        return q + u
    }

    func bestChild(explorationConstant: Float) -> MCTSNode<State>? {
        children.max { a, b in
            a.puctScore(explorationConstant: explorationConstant) < b.puctScore(explorationConstant: explorationConstant)
        }
    }

    @discardableResult
    func expand<Evaluator: PolicyValueNetwork>(using evaluator: Evaluator) -> Float where Evaluator.State == State {
        guard !state.isTerminal else {
            return state.terminalValue
        }
        let (policy, value) = evaluator.evaluate(state: state)
        return expand(with: policy, value: value)
    }

    @discardableResult
    func expand(with policy: [Float], value: Float) -> Float {
        let legalMoves = cachedLegalMoves
        guard !legalMoves.isEmpty else { return value }

        let rawPriors = legalMoves.map { move -> Float in
            let encoding = move.encoded()
            let raw = encoding < policy.count ? policy[encoding] : 0
            return raw.isFinite ? max(0, raw) : 0
        }
        let priorSum = rawPriors.reduce(0, +)
        let fallbackPrior = 1.0 / Float(legalMoves.count)

        for (index, move) in legalMoves.enumerated() {
            let childState = state.applying(move: move)
            let prior = priorSum > 0 ? rawPriors[index] / priorSum : fallbackPrior
            let child = MCTSNode(
                state: childState,
                move: move,
                parent: self,
                prior: prior
            )
            children.append(child)
        }

        return value
    }

    func backpropagate(value: Float) {
        visits += 1
        totalValue += value
        parent?.backpropagate(value: -value)
    }
}
