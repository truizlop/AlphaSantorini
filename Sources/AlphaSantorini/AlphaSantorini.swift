import Training
import NeuralNetwork
import Foundation

@main
struct AlphaSantorini {
    static func main() async throws {
        let config = TrainingConfig()
        try? FileManager.default.createDirectory(
            at: config.checkpointDirectory,
            withIntermediateDirectories: true
        )
        
        let trainer = SantoriniTrainer(config: config)
        await trainer.train(iterations: 500)
    }
}
