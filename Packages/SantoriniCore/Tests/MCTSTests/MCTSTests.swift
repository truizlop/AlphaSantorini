import XCTest
@testable import MCTS

private struct SeededGenerator: RandomNumberGenerator {
    private var state: UInt64

    init(seed: UInt64) {
        self.state = seed
    }

    mutating func next() -> UInt64 {
        state = state &* 6364136223846793005 &+ 1442695040888963407
        return state
    }
}

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

private struct SingleMoveState: GameState {
    typealias Move = TestMove

    var isTerminal: Bool { false }
    var terminalValue: Float { 0 }

    func legalMoves() -> [TestMove] { [.win] }

    func applying(move: TestMove) -> SingleMoveState {
        self
    }
}

private struct UniformEvaluator: PolicyValueNetwork {
    typealias State = TestState

    func evaluate(state: TestState) -> (policy: [Float], value: Float) {
        ([0.5, 0.5], 0.0)
    }
}

private struct SingleMoveEvaluator: PolicyValueNetwork {
    typealias State = SingleMoveState

    func evaluate(state: SingleMoveState) -> (policy: [Float], value: Float) {
        ([0.0, 1.0], 0.0)
    }
}

final class MCTSTests: XCTestCase {
    func testMctsPrefersWinningMove() {
        let root = TestState(toPlay: 0, terminal: false, terminalScore: 0, winningMove: .win)
        let evaluator = UniformEvaluator()
        let result = mcts(rootState: root, evaluator: evaluator, iterations: 50, temperature: 0.0)
        XCTAssertEqual(result.bestMove, .win)
    }

    func testMctsDeterministicWithSeed() {
        let root = TestState(toPlay: 0, terminal: false, terminalScore: 0, winningMove: .win)
        let evaluator = UniformEvaluator()
        var rng1 = SeededGenerator(seed: 42)
        var rng2 = SeededGenerator(seed: 42)
        let result1 = mcts(
            rootState: root,
            evaluator: evaluator,
            iterations: 50,
            temperature: 1.0,
            rng: &rng1
        )
        let result2 = mcts(
            rootState: root,
            evaluator: evaluator,
            iterations: 50,
            temperature: 1.0,
            rng: &rng2
        )
        XCTAssertEqual(result1.distribution.keys, result2.distribution.keys)
        for key in result1.distribution.keys {
            XCTAssertEqual(result1.distribution[key]!, result2.distribution[key]!, accuracy: 1e-6)
        }
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
        var rng = SeededGenerator(seed: 7)
        node.addDirichletNoise(noise, rng: &rng)

        let priors = node.children.map(\.prior)
        let sum = priors.reduce(0, +)
        XCTAssertEqual(sum, 1.0, accuracy: 1e-5)
        XCTAssertTrue(priors.allSatisfy { $0 >= 0 && $0 <= 1 })
    }

    func testMctsDistributionOnlyUsesLegalMoves() {
        let evaluator = SingleMoveEvaluator()
        let state = SingleMoveState()
        let result = mcts(rootState: state, evaluator: evaluator, iterations: 10, temperature: 1.0)
        XCTAssertEqual(result.distribution.count, 1)
        XCTAssertNotNil(result.distribution[.win])
        XCTAssertNil(result.distribution[.lose])
    }
}
