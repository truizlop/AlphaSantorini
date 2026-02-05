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
        subcommands: [Train.self, Inspect.self, SelfPlayInspect.self],
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

    struct SelfPlayInspect: ParsableCommand {
        static let configuration = CommandConfiguration(
            commandName: "self-play-inspect",
            abstract: "Run one self-play game and print per-sample policy statistics."
        )

        @Option(help: "Checkpoint file to load (safetensors).")
        var checkpoint: String?

        @Option(help: "Hidden dimension for the network.")
        var hiddenDimension: Int = 256

        @Option(help: "MCTS simulations per move.")
        var mctsSimulations: Int = 400

        @Option(help: "MCTS batch size.")
        var mctsBatchSize: Int = 16

        @Flag(help: "Disable Dirichlet noise for self-play.")
        var noNoise: Bool = false

        @Option(help: "Dirichlet noise epsilon.")
        var noiseEpsilon: Float = 0.25

        @Option(help: "Dirichlet noise alpha.")
        var noiseAlpha: Float = 0.3

        @Option(help: "Optional RNG seed for deterministic self-play.")
        var seed: UInt64?

        @Option(help: "Limit printed samples (0 prints all).")
        var maxSamples: Int = 0

        func run() throws {
            let net = SantoriniNet(hiddenDimension: hiddenDimension)
            if let checkpoint {
                try net.load(from: URL(filePath: checkpoint))
            }

            let noise: DirichletNoise? = noNoise ? nil : DirichletNoise(epsilon: noiseEpsilon, alpha: noiseAlpha)
            let selfPlay = SelfPlay()

            let result: SelfPlayResult
            if let seed {
                var rng = SeededGenerator(seed: seed)
                result = selfPlay.runWithDiagnostics(
                    evaluator: net,
                    iterations: mctsSimulations,
                    noise: noise,
                    batchSize: mctsBatchSize,
                    rng: &rng
                )
            } else {
                result = selfPlay.runWithDiagnostics(
                    evaluator: net,
                    iterations: mctsSimulations,
                    noise: noise,
                    batchSize: mctsBatchSize
                )
            }

            print("Self-play produced \(result.samples.count) samples (wasTruncated=\(result.wasTruncated)).")
            if result.samples.isEmpty {
                return
            }

            var oneHot99 = 0
            var oneHot95 = 0
            var meanEntropy: Float = 0
            var meanMax: Float = 0
            var meanNonZero: Float = 0

            let limit = maxSamples > 0 ? min(maxSamples, result.samples.count) : result.samples.count
            print("Idx | Sum    | Max    | 2nd    | NonZero | Entropy")
            print("----+--------+--------+--------+---------+--------")

            for (index, sample) in result.samples.enumerated() {
                let probs = sample.encodedPolicy
                let stats = policyStats(for: probs)

                meanEntropy += stats.entropy
                meanMax += stats.max
                meanNonZero += Float(stats.nonZero)
                if stats.max >= 0.99 { oneHot99 += 1 }
                if stats.max >= 0.95 { oneHot95 += 1 }

                if index < limit {
                    print(String(
                        format: "%3d | %.4f | %.4f | %.4f | %7d | %.4f",
                        index + 1,
                        stats.sum,
                        stats.max,
                        stats.second,
                        stats.nonZero,
                        stats.entropy
                    ))
                }
            }

            let count = Float(result.samples.count)
            print("Summary:")
            print(String(format: "  mean entropy: %.4f", meanEntropy / count))
            print(String(format: "  mean max:     %.4f", meanMax / count))
            print(String(format: "  mean nonZero: %.1f", meanNonZero / count))
            print(String(format: "  max>=0.99:    %d/%d", oneHot99, result.samples.count))
            print(String(format: "  max>=0.95:    %d/%d", oneHot95, result.samples.count))
        }

        private func policyStats(for probs: [Float]) -> (sum: Float, max: Float, second: Float, entropy: Float, nonZero: Int) {
            var sum: Float = 0
            var max1: Float = -Float.greatestFiniteMagnitude
            var max2: Float = -Float.greatestFiniteMagnitude
            var entropy: Float = 0
            var nonZero = 0

            for p in probs {
                sum += p
                if p > max1 {
                    max2 = max1
                    max1 = p
                } else if p > max2 {
                    max2 = p
                }
                if p > 1.0e-6 {
                    nonZero += 1
                    entropy -= p * log(p + 1.0e-12)
                }
            }

            return (sum, max1, max2 == -Float.greatestFiniteMagnitude ? 0 : max2, entropy, nonZero)
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
