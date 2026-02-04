import XCTest
@testable import Training
import NeuralNetwork
import Santorini
import MCTS

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

final class TrainingTests: XCTestCase {
    private func makeSample(row: Int, col: Int, outcome: Float) -> TrainingSample {
        var state = GameState()
        let encoding = row * 5 + col
        let action = Action.from(encoding: encoding)!
        if case let .placement(placement) = action {
            state.placement(placement)
        }
        let policy: [Action: Float] = [action: 1.0]
        return TrainingSample(state: state, action: action, policy: policy, outcome: outcome)
    }

    func testReplayBufferRespectsCapacity() {
        let buffer = ReplayBuffer(maxSize: 3)
        buffer.add([makeSample(row: 0, col: 0, outcome: 1)])
        buffer.add([makeSample(row: 0, col: 1, outcome: -1)])
        buffer.add([makeSample(row: 0, col: 2, outcome: 1)])
        XCTAssertEqual(buffer.count, 3)

        buffer.add([makeSample(row: 0, col: 3, outcome: 0)])
        XCTAssertEqual(buffer.count, 3)
    }

    func testReplayBufferSampleIsUniqueWhenPossible() {
        let buffer = ReplayBuffer(maxSize: 10)
        buffer.add([
            makeSample(row: 0, col: 0, outcome: 1),
            makeSample(row: 0, col: 1, outcome: 1),
            makeSample(row: 0, col: 2, outcome: 1),
            makeSample(row: 0, col: 3, outcome: 1),
        ])

        let batch = buffer.sample(batchSize: 3)
        let hashes = batch.map(\.stateHash)
        XCTAssertEqual(Set(hashes).count, hashes.count)
    }

    func testTrainingSampleEncodingAndStateHash() {
        var sample = makeSample(row: 1, col: 1, outcome: 1)
        XCTAssertEqual(sample.encodedPolicy.count, Action.total)

        let oldHash = sample.stateHash
        var newState = GameState()
        if case let .placement(placement) = Action.from(encoding: 24)! {
            newState.placement(placement)
        }
        sample.state = newState
        XCTAssertNotEqual(sample.stateHash, oldHash)
    }

    func testSelfPlayProducesSamples() {
        let net = SantoriniNet(hiddenDimension: 16)
        let selfPlay = SelfPlay()
        let samples = selfPlay.run(evaluator: net, iterations: 1, noise: nil, batchSize: 1)
        XCTAssertFalse(samples.isEmpty)
        XCTAssertTrue(samples.allSatisfy { $0.encodedPolicy.count == Action.total })
        XCTAssertTrue(samples.allSatisfy { [-1.0, 0.0, 1.0].contains($0.outcome) })
    }

    func testSelfPlayTruncationSkipsSamples() {
        let net = SantoriniNet(hiddenDimension: 16)
        let selfPlay = SelfPlay()
        let result = selfPlay.runWithDiagnostics(
            evaluator: net,
            iterations: 1,
            noise: nil,
            batchSize: 1,
            maxMoves: 0
        )
        XCTAssertTrue(result.wasTruncated)
        XCTAssertTrue(result.samples.isEmpty)
    }

    func testSelfPlayDeterministicWithSeed() {
        let net = SantoriniNet(hiddenDimension: 16)
        let selfPlay = SelfPlay()
        let noise = DirichletNoise(epsilon: 0.25, alpha: 0.3)

        var rng1 = SeededGenerator(seed: 42)
        var rng2 = SeededGenerator(seed: 42)

        let result1 = selfPlay.runWithDiagnostics(
            evaluator: net,
            iterations: 4,
            noise: noise,
            batchSize: 2,
            maxMoves: 5,
            rng: &rng1
        )
        let result2 = selfPlay.runWithDiagnostics(
            evaluator: net,
            iterations: 4,
            noise: noise,
            batchSize: 2,
            maxMoves: 5,
            rng: &rng2
        )

        XCTAssertEqual(result1.samples.count, result2.samples.count)
        for (lhs, rhs) in zip(result1.samples, result2.samples) {
            XCTAssertEqual(lhs.action, rhs.action)
            XCTAssertEqual(lhs.outcome, rhs.outcome, accuracy: 1e-6)
            XCTAssertEqual(lhs.encodedPolicy.count, rhs.encodedPolicy.count)
            for (a, b) in zip(lhs.encodedPolicy, rhs.encodedPolicy) {
                XCTAssertEqual(a, b, accuracy: 1e-6)
            }
        }
    }

    func testAIVSPlayRespectsMaxMoves() {
        let net = SantoriniNet(hiddenDimension: 16)
        let aiPlay = AIVSPlay()
        let result = aiPlay.play(player1: net, player2: net, maxMoves: 0)
        XCTAssertNil(result)
    }

    func testTrainingConfigCustomValues() {
        let tempDir = URL(filePath: NSTemporaryDirectory()).appending(path: "santorini_tests")
        let config = TrainingConfig(
            hiddenDimension: 32,
            gamesPerIteration: 1,
            MCTSSimulations: 1,
            mctsBatchSize: 1,
            noise: nil,
            maxMovesPerGame: 10,
            batchSize: 2,
            trainingStepsPerIteration: 1,
            learningRate: 0.01,
            replayBufferSize: 10,
            evaluationGames: 0,
            promotionThreshold: 0.5,
            evaluationInterval: 100,
            checkpointInterval: 100,
            checkpointDirectory: tempDir
        )
        XCTAssertEqual(config.hiddenDimension, 32)
        XCTAssertEqual(config.gamesPerIteration, 1)
        XCTAssertEqual(config.checkpointDirectory, tempDir)
        XCTAssertNil(config.noise)
        XCTAssertEqual(config.maxMovesPerGame, 10)
    }
}
