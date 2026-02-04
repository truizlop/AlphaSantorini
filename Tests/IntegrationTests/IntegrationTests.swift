import XCTest
@testable import Training

final class IntegrationTests: XCTestCase {
    func testSingleIterationTrainingCreatesCheckpoint() async throws {
        let tempDir = URL(filePath: NSTemporaryDirectory()).appending(path: "santorini_integration_tests")
        try FileManager.default.createDirectory(at: tempDir, withIntermediateDirectories: true)

        let config = TrainingConfig(
            hiddenDimension: 16,
            gamesPerIteration: 1,
            MCTSSimulations: 1,
            mctsBatchSize: 1,
            noise: nil,
            batchSize: 2,
            trainingStepsPerIteration: 1,
            learningRate: 0.001,
            replayBufferSize: 100,
            evaluationGames: 0,
            promotionThreshold: 0.5,
            evaluationInterval: 1000,
            checkpointInterval: 1000,
            checkpointDirectory: tempDir
        )

        let trainer = SantoriniTrainer(config: config)
        await trainer.train(iterations: 1)

        let finalCheckpoint = tempDir.appending(path: "final.safetensors")
        XCTAssertTrue(FileManager.default.fileExists(atPath: finalCheckpoint.path()))
    }
}
