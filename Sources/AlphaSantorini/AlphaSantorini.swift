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
        subcommands: [Train.self, Inspect.self, SelfPlayInspect.self, Arena.self, ArenaBaseline.self],
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

            struct Aggregate {
                var count: Int = 0
                var sumEntropy: Float = 0
                var sumMax: Float = 0
                var sumSum: Float = 0
                var sumRatio: Float = 0
                var hot95: Int = 0
                var hot99: Int = 0

                mutating func add(stats: (sum: Float, max: Float, second: Float, entropy: Float, nonZero: Int), uniform: Float) {
                    count += 1
                    sumEntropy += stats.entropy
                    sumMax += stats.max
                    sumSum += stats.sum
                    if uniform > 0 {
                        sumRatio += stats.max / uniform
                    }
                    if stats.max >= 0.99 { hot99 += 1 }
                    if stats.max >= 0.95 { hot95 += 1 }
                }

                func report(label: String) {
                    guard count > 0 else {
                        print("  \(label): no samples")
                        return
                    }
                    let meanEntropy = sumEntropy / Float(count)
                    let meanMax = sumMax / Float(count)
                    let meanSum = sumSum / Float(count)
                    let meanRatio = sumRatio / Float(count)
                    print(String(format: "  %@: count=%d meanSum=%.4f meanMax=%.4f meanEntropy=%.4f meanMax/Uni=%.2f hot>=0.95=%d hot>=0.99=%d",
                                 label, count, meanSum, meanMax, meanEntropy, meanRatio, hot95, hot99))
                }
            }

            var aggregateAll = Aggregate()
            var aggregateEarly = Aggregate()
            var aggregateMid = Aggregate()
            var aggregateLate = Aggregate()
            var aggregatePlacement = Aggregate()
            var aggregatePlay = Aggregate()

            let limit = maxSamples > 0 ? min(maxSamples, result.samples.count) : result.samples.count
            print("Idx | Ph | Leg | Sum    | Max    | 2nd    | M/Uni | Entropy")
            print("----+----+-----+--------+--------+--------+------+--------")

            for (index, sample) in result.samples.enumerated() {
                let probs = sample.encodedPolicy
                let stats = policyStats(for: probs)
                let legalCount = sample.state.legalActions.count
                let uniform = legalCount > 0 ? 1.0 / Float(legalCount) : 0
                let phaseLabel = sample.state.phase == .placement ? "Pl" : "Mv"
                let bucket = index * 3 / max(1, result.samples.count)
                let ratio = uniform > 0 ? stats.max / uniform : 0

                aggregateAll.add(stats: stats, uniform: uniform)
                switch bucket {
                case 0: aggregateEarly.add(stats: stats, uniform: uniform)
                case 1: aggregateMid.add(stats: stats, uniform: uniform)
                default: aggregateLate.add(stats: stats, uniform: uniform)
                }
                if sample.state.phase == .placement {
                    aggregatePlacement.add(stats: stats, uniform: uniform)
                } else {
                    aggregatePlay.add(stats: stats, uniform: uniform)
                }

                if index < limit {
                    print(String(
                        format: "%3d | %@ | %3d | %.4f | %.4f | %.4f | %4.1f | %.4f",
                        index + 1,
                        phaseLabel,
                        legalCount,
                        stats.sum,
                        stats.max,
                        stats.second,
                        ratio,
                        stats.entropy
                    ))
                }

                if abs(stats.sum - 1.0) > 1e-3 {
                    print(String(format: "⚠️ Policy sum off at sample %d (sum=%.6f).", index + 1, stats.sum))
                }
            }

            print("Summary:")
            aggregateAll.report(label: "All")
            aggregateEarly.report(label: "Early")
            aggregateMid.report(label: "Mid")
            aggregateLate.report(label: "Late")
            aggregatePlacement.report(label: "Placement")
            aggregatePlay.report(label: "Play")
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

    struct Arena: ParsableCommand {
        static let configuration = CommandConfiguration(
            commandName: "arena",
            abstract: "Play two checkpoints against each other with MCTS and report win rates."
        )

        @Option(help: "Checkpoint A (safetensors).")
        var checkpointA: String

        @Option(help: "Checkpoint B (safetensors).")
        var checkpointB: String

        @Option(help: "Hidden dimension for both networks.")
        var hiddenDimension: Int = 256

        @Option(help: "MCTS simulations per move.")
        var mctsSimulations: Int = 200

        @Option(help: "Number of games to play.")
        var games: Int = 50

        @Option(help: "Optional RNG seed for deterministic arena.")
        var seed: UInt64?

        func run() throws {
            let netA = SantoriniNet(hiddenDimension: hiddenDimension)
            let netB = SantoriniNet(hiddenDimension: hiddenDimension)
            try netA.load(from: URL(filePath: checkpointA))
            try netB.load(from: URL(filePath: checkpointB))

            var winsA = 0
            var winsB = 0
            var draws = 0

            if let seed {
                var rng = SeededGenerator(seed: seed)
                playMatches(netA: netA, netB: netB, rng: &rng, winsA: &winsA, winsB: &winsB, draws: &draws)
            } else {
                var rng = SystemRandomNumberGenerator()
                playMatches(netA: netA, netB: netB, rng: &rng, winsA: &winsA, winsB: &winsB, draws: &draws)
            }

            let decisive = winsA + winsB
            let winRateA = decisive > 0 ? Float(winsA) / Float(decisive) : 0.5
            print("Arena results: A wins=\(winsA), B wins=\(winsB), draws=\(draws), decisive=\(decisive), A winrate=\(String(format: "%.3f", winRateA))")
        }

        private func playMatches<R: RandomNumberGenerator>(
            netA: SantoriniNet,
            netB: SantoriniNet,
            rng: inout R,
            winsA: inout Int,
            winsB: inout Int,
            draws: inout Int
        ) {
            for game in 0..<games {
                let aPlaysFirst = (game % 2 == 0)
                let winner = playSingleGame(
                    player1: aPlaysFirst ? netA : netB,
                    player2: aPlaysFirst ? netB : netA,
                    rng: &rng
                )
                guard let winner else {
                    draws += 1
                    continue
                }

                if aPlaysFirst {
                    winsA += (winner == .one) ? 1 : 0
                    winsB += (winner == .two) ? 1 : 0
                } else {
                    winsA += (winner == .two) ? 1 : 0
                    winsB += (winner == .one) ? 1 : 0
                }
            }
        }

        private func playSingleGame<R: RandomNumberGenerator>(
            player1: SantoriniNet,
            player2: SantoriniNet,
            rng: inout R
        ) -> Player? {
            var state = GameState()
            while !state.isOver {
                let net = state.turn == .one ? player1 : player2
                let (action, _) = mcts(
                    rootState: state,
                    evaluator: net,
                    iterations: mctsSimulations,
                    temperature: 0.0,
                    rng: &rng
                )
                guard let action else { return nil }
                state = state.applying(move: action)
            }
            return state.winner
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

    struct ArenaBaseline: ParsableCommand {
        static let configuration = CommandConfiguration(
            commandName: "arena-baseline",
            abstract: "Play a checkpoint against MCTS with a uniform (no-network) evaluator."
        )

        @Option(help: "Checkpoint (safetensors) for the trained network.")
        var checkpoint: String

        @Option(help: "Hidden dimension for the network.")
        var hiddenDimension: Int = 256

        @Option(help: "MCTS simulations per move.")
        var mctsSimulations: Int = 200

        @Option(help: "Number of games to play.")
        var games: Int = 50

        @Option(help: "Optional RNG seed for deterministic arena.")
        var seed: UInt64?

        func run() throws {
            let trained = SantoriniNet(hiddenDimension: hiddenDimension)
            try trained.load(from: URL(filePath: checkpoint))

            let baseline = UniformPolicyEvaluator()

            var winsTrained = 0
            var winsBaseline = 0
            var draws = 0

            if let seed {
                var rng = SeededGenerator(seed: seed)
                playMatches(trained: trained, baseline: baseline, rng: &rng, winsTrained: &winsTrained, winsBaseline: &winsBaseline, draws: &draws)
            } else {
                var rng = SystemRandomNumberGenerator()
                playMatches(trained: trained, baseline: baseline, rng: &rng, winsTrained: &winsTrained, winsBaseline: &winsBaseline, draws: &draws)
            }

            let decisive = winsTrained + winsBaseline
            let winRate = decisive > 0 ? Float(winsTrained) / Float(decisive) : 0.5
            let winRateString = String(format: "%.3f", winRate)
            print("Arena baseline results: trained wins=\(winsTrained), baseline wins=\(winsBaseline), draws=\(draws), decisive=\(decisive), trained winrate=\(winRateString)")
        }

        private func playMatches<R: RandomNumberGenerator>(
            trained: SantoriniNet,
            baseline: UniformPolicyEvaluator,
            rng: inout R,
            winsTrained: inout Int,
            winsBaseline: inout Int,
            draws: inout Int
        ) {
            for game in 0..<games {
                let trainedPlaysFirst = (game % 2 == 0)
                let winner = playSingleGame(
                    trained: trained,
                    baseline: baseline,
                    trainedPlaysFirst: trainedPlaysFirst,
                    rng: &rng
                )
                guard let winner else {
                    draws += 1
                    continue
                }

                if trainedPlaysFirst {
                    winsTrained += (winner == .one) ? 1 : 0
                    winsBaseline += (winner == .two) ? 1 : 0
                } else {
                    winsTrained += (winner == .two) ? 1 : 0
                    winsBaseline += (winner == .one) ? 1 : 0
                }
            }
        }

        private func playSingleGame<R: RandomNumberGenerator>(
            trained: SantoriniNet,
            baseline: UniformPolicyEvaluator,
            trainedPlaysFirst: Bool,
            rng: inout R
        ) -> Player? {
            var state = Santorini.GameState()
            while !state.isOver {
                let trainedTurn: Bool
                if trainedPlaysFirst {
                    trainedTurn = (state.turn == .one)
                } else {
                    trainedTurn = (state.turn == .two)
                }

                let (action, _) = trainedTurn
                    ? mcts(rootState: state, evaluator: trained, iterations: mctsSimulations, temperature: 0.0, rng: &rng)
                    : mcts(rootState: state, evaluator: baseline, iterations: mctsSimulations, temperature: 0.0, rng: &rng)
                guard let action else { return nil }
                state = state.applying(move: action)
            }
            return state.winner
        }
    }
}

private struct UniformPolicyEvaluator: PolicyValueNetwork {
    typealias State = Santorini.GameState

    func evaluate(state: Santorini.GameState) -> (policy: [Float], value: Float) {
        let policy = Array(repeating: Float(0), count: Action.total)
        return (policy, 0)
    }
}
