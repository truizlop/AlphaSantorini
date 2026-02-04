import ArgumentParser
import Foundation
import Training
import NeuralNetwork
import Santorini
import MCTS

@main
struct AlphaSantorini: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "alpha-santorini",
        subcommands: [Train.self, Inspect.self],
        defaultSubcommand: Train.self
    )

    struct Train: AsyncParsableCommand {
        @Option(help: "Number of training iterations.")
        var iterations: Int = 500

        @Option(help: "Override checkpoint output directory.")
        var checkpointDir: String?

        @Option(help: "Hidden dimension for the network.")
        var hiddenDimension: Int = 256

        func run() async throws {
            let checkpointURL = checkpointDir.map { URL(filePath: $0, directoryHint: .isDirectory) }
            let config = TrainingConfig(
                hiddenDimension: hiddenDimension,
                checkpointDirectory: checkpointURL
            )
            try? FileManager.default.createDirectory(
                at: config.checkpointDirectory,
                withIntermediateDirectories: true
            )
            let trainer = SantoriniTrainer(config: config)
            await trainer.train(iterations: iterations)
        }
    }

    struct Inspect: AsyncParsableCommand {
        @Option(help: "Checkpoint file to load (safetensors).")
        var checkpoint: String?

        @Option(help: "Hidden dimension for the network.")
        var hiddenDimension: Int = 256

        @Option(help: "MCTS simulations per move.")
        var mctsSimulations: Int = 200

        @Option(help: "Top-K actions to display.")
        var topK: Int = 10

        @Option(help: "Advance N random legal moves before inspection.")
        var advance: Int = 0

        @Option(help: "Optional RNG seed for deterministic inspection.")
        var seed: UInt64?

        func run() async throws {
            if let seed {
                var rng = SeededGenerator(seed: seed)
                try runInspect(rng: &rng)
            } else {
                var rng = SystemRandomNumberGenerator()
                try runInspect(rng: &rng)
            }
        }

        private func runInspect<R: RandomNumberGenerator>(rng: inout R) throws {
            let net = SantoriniNet(hiddenDimension: hiddenDimension)
            if let checkpoint {
                try net.load(from: URL(filePath: checkpoint))
            }

            var state = GameState()
            if advance > 0 {
                for _ in 0..<advance {
                    let legal = state.legalActions
                    guard !legal.isEmpty else { break }
                    let idx = Int.random(in: 0..<legal.count, using: &rng)
                    state = state.applying(move: legal[idx])
                }
            }

            let (policy, value) = net.evaluate(state.encoded())
            let legalActions = state.legalActions
            let netRanked = legalActions
                .map { action in (action, policy[action.encoded()]) }
                .sorted { $0.1 > $1.1 }

            var rngCopy = rng
            let (bestMove, mctsPolicy) = mcts(
                rootState: state,
                evaluator: net,
                iterations: mctsSimulations,
                temperature: 1.0,
                rng: &rngCopy
            )
            let mctsRanked = mctsPolicy
                .sorted { $0.value > $1.value }
                .map { ($0.key, $0.value) }

            print("State: turn=\(state.turn) phase=\(state.phase)")
            print(String(format: "Network value: %.4f", value))
            print("Best move (MCTS): \(bestMove?.description ?? "<none>")")
            printTopK(title: "Top network priors (legal only)", items: netRanked, topK: topK)
            printTopK(title: "Top MCTS policy", items: mctsRanked, topK: topK)
        }

        private func printTopK(title: String, items: [(Action, Float)], topK: Int) {
            print(title + ":")
            let count = min(topK, items.count)
            for i in 0..<count {
                let (action, prob) = items[i]
                print("  \(i + 1). \(action.description) \(String(format: "%.4f", prob))")
            }
        }
    }

    struct SeededGenerator: RandomNumberGenerator {
        private var state: UInt64

        init(seed: UInt64) {
            self.state = seed
        }

        mutating func next() -> UInt64 {
            state = state &* 6364136223846793005 &+ 1442695040888963407
            return state
        }
    }
}
