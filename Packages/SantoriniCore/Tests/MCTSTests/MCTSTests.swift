import XCTest
@testable import MCTS

private enum TestMove: Int, CaseIterable, Hashable, ActionEncodable {
    case win = 0
    case lose = 1

    func encoded() -> Int { rawValue }
}

private struct TestState: GameState {
    typealias Move = TestMove

    var toPlay: Int
    var terminal: Bool
    var terminalScore: Float
    var winningMove: TestMove

    var isTerminal: Bool { terminal }

    var terminalValue: Float {
        guard terminal else { return 0 }
        return terminalScore
    }

    func legalMoves() -> [TestMove] {
        terminal ? [] : TestMove.allCases
    }

    func applying(move: TestMove) -> TestState {
        guard !terminal else { return self }
        let nextToPlay = 1 - toPlay
        let valueForNextPlayer: Float = (move == winningMove) ? -1.0 : 1.0
        return TestState(
            toPlay: nextToPlay,
            terminal: true,
            terminalScore: valueForNextPlayer,
            winningMove: winningMove
        )
    }
}

private struct UniformEvaluator: PolicyValueNetwork, BatchPolicyValueNetwork {
    typealias State = TestState

    func evaluate(state: TestState) -> (policy: [Float], value: Float) {
        ([0.5, 0.5], 0.0)
    }

    func evaluate(states: [TestState]) -> (policies: [[Float]], values: [Float]) {
        (Array(repeating: [0.5, 0.5], count: states.count), Array(repeating: 0.0, count: states.count))
    }
}

final class MCTSTests: XCTestCase {
    func testMctsPrefersWinningMove() {
        let root = TestState(toPlay: 0, terminal: false, terminalScore: 0, winningMove: .win)
        let evaluator = UniformEvaluator()
        let result = mcts(rootState: root, evaluator: evaluator, iterations: 50, temperature: 0.0)
        XCTAssertEqual(result.bestMove, .win)
    }

    func testExpandIsIdempotent() {
        let state = TestState(toPlay: 0, terminal: false, terminalScore: 0, winningMove: .win)
        let node = MCTSNode(state: state, move: nil, prior: 0)
        _ = node.expand(with: [0.5, 0.5], value: 0.0)
        let firstCount = node.children.count
        _ = node.expand(with: [0.5, 0.5], value: 0.0)
        XCTAssertEqual(node.children.count, firstCount)
    }

    func testDirichletNoiseKeepsValidPriors() {
        let state = TestState(toPlay: 0, terminal: false, terminalScore: 0, winningMove: .win)
        let node = MCTSNode(state: state, move: nil, prior: 0)
        _ = node.expand(with: [0.5, 0.5], value: 0.0)

        let noise = DirichletNoise(epsilon: 0.25, alpha: 0.3)
        node.addDirichletNoise(noise)

        let priors = node.children.map(\.prior)
        let sum = priors.reduce(0, +)
        XCTAssertEqual(sum, 1.0, accuracy: 1e-5)
        XCTAssertTrue(priors.allSatisfy { $0 >= 0 && $0 <= 1 })
    }
}
