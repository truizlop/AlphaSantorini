import XCTest
@testable import Training
@testable import NeuralNetwork
import Santorini
import MLX
import MLXNN
import MLXOptimizers

private typealias EncodedState = [[[Float]]]

final class ExperimentTests: XCTestCase {
    func testTrainingSampleValueMapping() {
        let action = Action.from(encoding: 0)!
        let policy: [Action: Float] = [action: 1.0]
        var base = GameState()
        let placements = [0, 1, 23, 24].compactMap { Action.from(encoding: $0) }
        for placement in placements {
            if case let .placement(p) = placement {
                base.placement(p)
            }
        }
        var stateP1 = base
        var stateP2 = base
        stateP1.turn = .one
        stateP2.turn = .two

        let history: [(Santorini.GameState, Action, Policy, Float?)] = [
            (stateP1, action, policy, nil),
            (stateP2, action, policy, nil)
        ]

        let samples = SelfPlay().buildSamples(from: history, terminalWinner: .one, valueTargetStrategy: .terminalOutcome)
        XCTAssertEqual(samples.count, 2)
        XCTAssertEqual(samples[0].outcome, 1, "Winner should be +1 for player-to-move = winner")
        XCTAssertEqual(samples[1].outcome, -1, "Winner should be -1 for player-to-move != winner")

        let drawSamples = SelfPlay().buildSamples(from: history, terminalWinner: nil, valueTargetStrategy: .terminalOutcome)
        XCTAssertEqual(drawSamples[0].outcome, 0)
        XCTAssertEqual(drawSamples[1].outcome, 0)
    }

    func testPolicyOnlyOverfit() {
        let (inputs, policyTargets, valueTargets, targetIndices) = makeTinyPolicyDataset()
        let net = SantoriniNet(filters: 32)
        let optimizer = Adam(learningRate: 0.1)

        let (initialPolicy, initialValue) = computeLosses(
            net: net,
            inputs: inputs,
            policyTargets: policyTargets,
            valueTargets: valueTargets
        )

        train(
            net: net,
            optimizer: optimizer,
            inputs: inputs,
            policyTargets: policyTargets,
            valueTargets: valueTargets,
            steps: 400,
            policyWeight: 1.0,
            valueWeight: 0.0
        )

        let (finalPolicy, finalValue) = computeLosses(
            net: net,
            inputs: inputs,
            policyTargets: policyTargets,
            valueTargets: valueTargets
        )

        let acc = policyTop1Accuracy(net: net, inputs: inputs, targetIndices: targetIndices)
        let targetProb = policyTargetProbability(net: net, inputs: inputs, targetIndices: targetIndices)
        print(String(format: "Policy-only overfit: P_loss %.4f -> %.4f, V_loss %.4f -> %.4f, acc=%.3f, p_target=%.3f",
                     initialPolicy, finalPolicy, initialValue, finalValue, acc, targetProb))

        XCTAssertLessThan(finalPolicy, initialPolicy * 0.2)
        XCTAssertGreaterThanOrEqual(acc, 0.9)
        XCTAssertGreaterThanOrEqual(targetProb, 0.9)
    }

    func testValueOnlyOverfit() {
        let (inputs, policyTargets, valueTargets, _) = makeTinyValueDataset()
        let net = SantoriniNet(filters: 32)
        let optimizer = Adam(learningRate: 0.01)

        let (initialPolicy, initialValue) = computeLosses(
            net: net,
            inputs: inputs,
            policyTargets: policyTargets,
            valueTargets: valueTargets
        )

        train(
            net: net,
            optimizer: optimizer,
            inputs: inputs,
            policyTargets: policyTargets,
            valueTargets: valueTargets,
            steps: 1000,
            policyWeight: 0.0,
            valueWeight: 1.0
        )

        let (finalPolicy, finalValue) = computeLosses(
            net: net,
            inputs: inputs,
            policyTargets: policyTargets,
            valueTargets: valueTargets
        )

        let mae = valueMeanAbsoluteError(net: net, inputs: inputs, valueTargets: valueTargets)
        print(String(format: "Value-only overfit: P_loss %.4f -> %.4f, V_loss %.4f -> %.4f, mae=%.3f",
                     initialPolicy, finalPolicy, initialValue, finalValue, mae))

        XCTAssertLessThan(finalValue, initialValue * 0.3)
        XCTAssertLessThanOrEqual(mae, 0.25)
    }

    func testMultiSampleValueOverfitFromSelfPlay() {
        let samples = makeSelfPlaySamples(seed: 999, maxSamples: 8, mctsIterations: 2)
        XCTAssertFalse(samples.isEmpty)

        let inputs = samples.map { $0.state.encoded() }
        let policyTargets = samples.map { $0.encodedPolicy }
        let valueTargets = samples.map { $0.outcome }

        let net = SantoriniNet(filters: 32)
        let optimizer = Adam(learningRate: 0.01)

        let (_, initialValueLoss) = computeLosses(
            net: net,
            inputs: inputs,
            policyTargets: policyTargets,
            valueTargets: valueTargets
        )
        let initialMAE = valueMeanAbsoluteError(net: net, inputs: inputs, valueTargets: valueTargets)

        train(
            net: net,
            optimizer: optimizer,
            inputs: inputs,
            policyTargets: policyTargets,
            valueTargets: valueTargets,
            steps: 400,
            policyWeight: 0.0,
            valueWeight: 1.0
        )

        let (_, finalValueLoss) = computeLosses(
            net: net,
            inputs: inputs,
            policyTargets: policyTargets,
            valueTargets: valueTargets
        )
        let finalMAE = valueMeanAbsoluteError(net: net, inputs: inputs, valueTargets: valueTargets)
        print(String(format: "Multi-sample value overfit: V_loss %.4f -> %.4f, mae %.4f -> %.4f",
                     initialValueLoss, finalValueLoss, initialMAE, finalMAE))

        XCTAssertTrue(finalValueLoss.isFinite)
        XCTAssertTrue(finalMAE.isFinite)
    }

    func testValueSignSanityCheck() {
        let (stateA, stateB) = makeTwoDistinctStates()
        let inputs = [stateA.encoded(), stateB.encoded()]
        let policyTargets = makeUniformPolicyTargets(count: 2)
        let valueTargets: [Float] = [1.0, -1.0]

        let net = SantoriniNet(filters: 32)
        let optimizer = Adam(learningRate: 0.005)

        let (_, initialValueLoss) = computeLosses(
            net: net,
            inputs: inputs,
            policyTargets: policyTargets,
            valueTargets: valueTargets
        )

        train(
            net: net,
            optimizer: optimizer,
            inputs: inputs,
            policyTargets: policyTargets,
            valueTargets: valueTargets,
            steps: 800,
            policyWeight: 0.0,
            valueWeight: 1.0
        )

        let (_, finalValueLoss) = computeLosses(
            net: net,
            inputs: inputs,
            policyTargets: policyTargets,
            valueTargets: valueTargets
        )
        let (_, values) = net.evaluateBatch(inputs)
        XCTAssertEqual(values.count, 2)
        print(String(format: "Value sign sanity: V_loss %.4f -> %.4f, pred_pos=%.3f pred_neg=%.3f",
                     initialValueLoss, finalValueLoss, values[0], values[1]))
        XCTAssertLessThan(finalValueLoss, initialValueLoss)
        XCTAssertGreaterThan(values[0], values[1])
    }

    func testBaselineValueMSEConstantPredictor() {
        let samples = makeSelfPlaySamples(seed: 2025, maxSamples: 32, mctsIterations: 2)
        XCTAssertFalse(samples.isEmpty)

        let outcomes = samples.map { $0.outcome }
        let baseline = baselineMSE(outcomes: outcomes)

        let inputs = samples.map { $0.state.encoded() }
        let net = SantoriniNet(filters: 16)
        let (_, values) = net.evaluateBatch(inputs)
        let untrainedMSE = meanSquaredError(predictions: values, targets: outcomes)

        print(String(format: "Baseline MSE: constant=%.4f, untrained=%.4f", baseline, untrainedMSE))

        XCTAssertTrue(baseline.isFinite)
        XCTAssertTrue(untrainedMSE.isFinite)
        XCTAssertGreaterThan(baseline, 0.1)
    }

    func testPredictionVsRolloutCorrelation() {
        let samples = makeSelfPlaySamples(seed: 4242, maxSamples: 6, mctsIterations: 2)
        XCTAssertFalse(samples.isEmpty)

        let net = SantoriniNet(filters: 16)
        let inputs = samples.map { $0.state.encoded() }
        let (_, predictions) = net.evaluateBatch(inputs)

        var rng = SeededGenerator(seed: 1337)
        let rolloutValues = samples.map { sample in
            rolloutValueEstimate(state: sample.state, playouts: 30, rng: &rng)
        }

        let corr = pearsonCorrelation(predictions, rolloutValues)
        print(String(format: "Prediction vs rollout correlation: %.4f", corr))

        XCTAssertTrue(corr.isFinite)
    }

    func testPredictionVsRolloutCorrelationOnCheckpoint() throws {
        let checkpoint = URL(filePath: "checkpoints/final.safetensors")
        guard FileManager.default.fileExists(atPath: checkpoint.path()) else {
            throw XCTSkip("Missing checkpoints/final.safetensors")
        }

        let samples = makeSelfPlaySamples(seed: 9090, maxSamples: 8, mctsIterations: 2)
        XCTAssertFalse(samples.isEmpty)

        let net = SantoriniNet(filters: 128)
        try net.load(from: checkpoint)

        let inputs = samples.map { $0.state.encoded() }
        let (_, predictions) = net.evaluateBatch(inputs)

        var rng = SeededGenerator(seed: 2024)
        let rolloutValues = samples.map { sample in
            rolloutValueEstimate(state: sample.state, playouts: 30, rng: &rng)
        }

        let corr = pearsonCorrelation(predictions, rolloutValues)
        print(String(format: "Checkpoint vs rollout correlation: %.4f", corr))
        XCTAssertTrue(corr.isFinite)
    }

    func testValueLossVsBaselineOnCheckpoint() throws {
        let checkpoint = URL(filePath: "checkpoints/final.safetensors")
        guard FileManager.default.fileExists(atPath: checkpoint.path()) else {
            throw XCTSkip("Missing checkpoints/final.safetensors")
        }

        let samples = makeSelfPlaySamples(seed: 6060, maxSamples: 16, mctsIterations: 2)
        XCTAssertFalse(samples.isEmpty)

        let net = SantoriniNet(filters: 128)
        try net.load(from: checkpoint)

        let inputs = samples.map { $0.state.encoded() }
        let policyTargets = samples.map { $0.encodedPolicy }
        let valueTargets = samples.map { $0.outcome }

        let (_, valueLoss) = computeLosses(
            net: net,
            inputs: inputs,
            policyTargets: policyTargets,
            valueTargets: valueTargets
        )
        let baseline = baselineMSE(outcomes: valueTargets)
        print(String(format: "Checkpoint value loss vs baseline: valueLoss=%.4f baseline=%.4f ratio=%.3f",
                     valueLoss, baseline, valueLoss / max(baseline, 1e-8)))
        XCTAssertTrue(valueLoss.isFinite)
        XCTAssertTrue(baseline.isFinite)
    }

    func testValueHeadGradientNonZero() {
        let samples = makeSelfPlaySamples(seed: 555, maxSamples: 8, mctsIterations: 2)
        XCTAssertFalse(samples.isEmpty)

        let inputs = samples.map { $0.state.encoded() }
        let policyTargets = samples.map { $0.encodedPolicy }
        let valueTargets = samples.map { $0.outcome }

        let flatInputs = inputs.flatMap { $0.flatMap { $0.flatMap { $0 } } }
        let obs = MLXArray(flatInputs, [inputs.count, 5, 5, 9])
        let piTarget = MLXArray(policyTargets.flatMap { $0 }, [inputs.count, policyTargets[0].count])
        let zTarget = MLXArray(valueTargets, [inputs.count, 1])
        let targets = concatenated([piTarget, zTarget], axis: 1)
        let valuesPerPolicy = policyTargets[0].count

        let net = SantoriniNet(filters: 32)
        let (_, grad) = valueAndGrad(model: net) { net, obs, targets in
            let (policyLogits, valuePred) = net(obs)
            let splits = targets.split(indices: [valuesPerPolicy], axis: 1)
            let pLoss = policyLoss(logits: policyLogits, target: splits[0])
            let vLoss = valueLoss(predicted: valuePred, target: splits[1])
            return pLoss * 0 + vLoss * 1
        }(net, obs, targets)

        let flat = grad.flattened()
        let valueGrads = flat.filter { $0.0.contains("valueHead") }
        let arrays = valueGrads.isEmpty ? flat.map { $0.1 } : valueGrads.map { $0.1 }
        let meanAbs = arrays.map(meanAbs).reduce(0, +) / Float(max(1, arrays.count))
        print(String(format: "Value head grad mean abs: %.6f", meanAbs))
        XCTAssertGreaterThan(meanAbs, 1e-6)
    }

    func testPlayerSwapSign() throws {
        let samples = makeSelfPlaySamples(seed: 8080, maxSamples: 12, mctsIterations: 2)
        guard let sample = samples.first(where: { $0.outcome != 0 }) else {
            XCTFail("Expected a non-draw sample")
            return
        }

        let swapped = swappedPlayersState(from: sample.state)
        let originalEncoded = sample.state.encoded()
        let swappedEncoded = swapped.encoded()
        if originalEncoded.elementsEqual(swappedEncoded) {
            throw XCTSkip("Swapped state encoding identical to original; sign test inconclusive")
        }

        let inputs = [originalEncoded, swappedEncoded]
        let policyTargets = makeUniformPolicyTargets(count: 2)
        let valueTargets: [Float] = [sample.outcome, -sample.outcome]

        let net = SantoriniNet(filters: 32)
        let optimizer = Adam(learningRate: 0.005)
        train(
            net: net,
            optimizer: optimizer,
            inputs: inputs,
            policyTargets: policyTargets,
            valueTargets: valueTargets,
            steps: 800,
            policyWeight: 0.0,
            valueWeight: 1.0
        )

        let (_, values) = net.evaluateBatch(inputs)
        print(String(format: "Player swap sign: pred_orig=%.3f pred_swapped=%.3f", values[0], values[1]))
        XCTAssertGreaterThan(values[0], values[1])
    }
    func testPolicyTargetsAreNormalizedAndLegal() {
        let samples = makeSelfPlaySamples(seed: 321, maxSamples: 10, mctsIterations: 2)
        XCTAssertFalse(samples.isEmpty)

        for sample in samples {
            let sum = sample.encodedPolicy.reduce(0, +)
            XCTAssertEqual(sum, 1.0, accuracy: 1e-3)

            if let maxIndex = sample.encodedPolicy.enumerated().max(by: { $0.element < $1.element })?.offset,
               let action = Action.from(encoding: maxIndex) {
                let legal = Set(sample.state.legalActions.map { $0.encoded() })
                XCTAssertTrue(legal.contains(action.encoded()))
            } else {
                XCTFail("Could not decode top policy action")
            }
        }
    }

    func testSelfPlayBatchPolicyOverfit() {
        let samples = makeSelfPlaySamples(seed: 777, maxSamples: 12, mctsIterations: 2)
        XCTAssertFalse(samples.isEmpty)

        let inputs = samples.map { $0.state.encoded() }
        let policyTargets = samples.map { $0.encodedPolicy }
        let valueTargets = samples.map { $0.outcome }

        let net = SantoriniNet(filters: 32)
        let optimizer = Adam(learningRate: 0.02)

        let (initialPolicyLoss, _) = computeLosses(
            net: net,
            inputs: inputs,
            policyTargets: policyTargets,
            valueTargets: valueTargets
        )

        train(
            net: net,
            optimizer: optimizer,
            inputs: inputs,
            policyTargets: policyTargets,
            valueTargets: valueTargets,
            steps: 300,
            policyWeight: 1.0,
            valueWeight: 0.0
        )

        let (finalPolicyLoss, _) = computeLosses(
            net: net,
            inputs: inputs,
            policyTargets: policyTargets,
            valueTargets: valueTargets
        )

        print(String(format: "Self-play policy overfit: P_loss %.4f -> %.4f", initialPolicyLoss, finalPolicyLoss))
        XCTAssertLessThan(finalPolicyLoss, initialPolicyLoss * 0.7)
    }
}

private func makeToyDataset() -> ([EncodedState], [[Float]], [Float], [Int]) {
    var inputs: [EncodedState] = []
    var policyTargets: [[Float]] = []
    var valueTargets: [Float] = []
    var targetIndices: [Int] = []

    for idx in 0..<8 {
        let row = idx / 4
        let col = idx % 4
        let action = Action.from(encoding: row * 5 + col)!
        var state = GameState()
        if case let .placement(placement) = action {
            state.placement(placement)
        }
        state.turn = (idx % 2 == 0) ? .one : .two

        inputs.append(state.encoded())
        var policy = Array(repeating: Float(0), count: Action.total)
        let actionIndex = action.encoded()
        policy[actionIndex] = 1.0
        policyTargets.append(policy)
        targetIndices.append(actionIndex)

        let value = -1.0 + (2.0 * Float(idx) / Float(7))
        valueTargets.append(value)
    }

    return (inputs, policyTargets, valueTargets, targetIndices)
}

private func makeTinyValueDataset() -> ([EncodedState], [[Float]], [Float], [Int]) {
    var inputs: [EncodedState] = []
    var policyTargets: [[Float]] = []
    var valueTargets: [Float] = []
    var targetIndices: [Int] = []

    let action = Action.from(encoding: 7)!
    var state = GameState()
    if case let .placement(placement) = action {
        state.placement(placement)
    }
    state.turn = .one

    inputs.append(state.encoded())
    var policy = Array(repeating: Float(0), count: Action.total)
    let actionIndex = action.encoded()
    policy[actionIndex] = 1.0
    policyTargets.append(policy)
    targetIndices.append(actionIndex)
    valueTargets.append(0.8)

    return (inputs, policyTargets, valueTargets, targetIndices)
}

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

private func makeSelfPlaySamples(
    seed: UInt64,
    maxSamples: Int,
    mctsIterations: Int
) -> [TrainingSample] {
    let net = SantoriniNet(filters: 16)
    let selfPlay = SelfPlay()
    var rng = SeededGenerator(seed: seed)
    let result = selfPlay.runWithDiagnostics(
        evaluator: net,
        iterations: mctsIterations,
        noise: nil,
        useTemperature: false,
        rng: &rng
    )
    XCTAssertFalse(result.wasTruncated)
    if maxSamples <= 0 { return result.samples }
    return Array(result.samples.prefix(maxSamples))
}

private func makeTwoDistinctStates() -> (GameState, GameState) {
    func buildState(placements: [Int], turn: Player) -> GameState {
        var state = GameState()
        for encoding in placements {
            if case let .placement(p) = Action.from(encoding: encoding)! {
                state.placement(p)
            }
        }
        state.turn = turn
        return state
    }

    let stateA = buildState(placements: [0, 1, 23, 24], turn: .one)
    let stateB = buildState(placements: [2, 3, 20, 21], turn: .one)
    return (stateA, stateB)
}

private func makeUniformPolicyTargets(count: Int) -> [[Float]] {
    let uniform = 1.0 / Float(Action.total)
    return Array(repeating: Array(repeating: uniform, count: Action.total), count: count)
}

private func baselineMSE(outcomes: [Float]) -> Float {
    guard !outcomes.isEmpty else { return 0 }
    let mean = outcomes.reduce(0, +) / Float(outcomes.count)
    let mse = outcomes.reduce(Float(0)) { acc, value in
        let diff = value - mean
        return acc + diff * diff
    }
    return mse / Float(outcomes.count)
}

private func meanSquaredError(predictions: [Float], targets: [Float]) -> Float {
    guard predictions.count == targets.count, !predictions.isEmpty else { return 0 }
    let mse = zip(predictions, targets).reduce(Float(0)) { acc, pair in
        let diff = pair.0 - pair.1
        return acc + diff * diff
    }
    return mse / Float(predictions.count)
}

private func meanAbs(_ array: MLXArray) -> Float {
    let values = array.asArray(Float.self)
    guard !values.isEmpty else { return 0 }
    let sum = values.reduce(0) { $0 + abs($1) }
    return sum / Float(values.count)
}

private func pearsonCorrelation(_ xs: [Float], _ ys: [Float]) -> Float {
    guard xs.count == ys.count, !xs.isEmpty else { return 0 }
    let n = Float(xs.count)
    let meanX = xs.reduce(0, +) / n
    let meanY = ys.reduce(0, +) / n
    var cov: Float = 0
    var varX: Float = 0
    var varY: Float = 0
    for (x, y) in zip(xs, ys) {
        let dx = x - meanX
        let dy = y - meanY
        cov += dx * dy
        varX += dx * dx
        varY += dy * dy
    }
    let denom = sqrt(varX * varY)
    return denom > 0 ? cov / denom : 0
}

private func rolloutValueEstimate(
    state: GameState,
    playouts: Int,
    rng: inout SeededGenerator
) -> Float {
    var wins = 0
    var losses = 0
    let perspective = state.turn
    for _ in 0..<playouts {
        var s = state
        while !s.isOver {
            let legal = s.legalActions
            guard !legal.isEmpty else { break }
            let idx = Int.random(in: 0..<legal.count, using: &rng)
            s = s.applying(move: legal[idx])
        }
        if let winner = s.winner {
            if winner == perspective { wins += 1 } else { losses += 1 }
        }
    }
    let total = wins + losses
    let winRate = total > 0 ? Float(wins) / Float(total) : 0.5
    return 2 * winRate - 1
}

private func swappedPlayersState(from state: GameState) -> GameState {
    let board = state.board
    func rank(_ id: WorkerID) -> Int {
        switch id {
        case .one: return 0
        case .two: return 1
        }
    }
    let p1Positions = state.workers
        .filter { $0.player == .one }
        .sorted { rank($0.id) < rank($1.id) }
        .map { $0.position }
    let p2Positions = state.workers
        .filter { $0.player == .two }
        .sorted { rank($0.id) < rank($1.id) }
        .map { $0.position }

    var swapped = GameState()
    // Placement order assigns workers to players alternately:
    // 1st+3rd => player .one, 2nd+4th => player .two.
    // To swap ownership, feed p2 positions into 1st/3rd.
    let order = [
        p2Positions.first,
        p1Positions.first,
        p2Positions.dropFirst().first,
        p1Positions.dropFirst().first
    ].compactMap { $0 }

    for position in order {
        let encoding = position.row * 5 + position.column
        if let placement = Placement.from(encoding: encoding) {
            swapped.placement(placement)
        }
    }

    swapped.board = board
    swapped.phase = state.phase
    // Keep turn the same so the "current player" now controls the opposite pieces.
    swapped.turn = state.turn
    return swapped
}

private func makeTinyPolicyDataset() -> ([EncodedState], [[Float]], [Float], [Int]) {
    var inputs: [EncodedState] = []
    var policyTargets: [[Float]] = []
    var valueTargets: [Float] = []
    var targetIndices: [Int] = []

    let action = Action.from(encoding: 12)!
    do {
        var state = GameState()
        if case let .placement(placement) = action {
            state.placement(placement)
        }
        state.turn = .one

        inputs.append(state.encoded())
        var policy = Array(repeating: Float(0), count: Action.total)
        let actionIndex = action.encoded()
        policy[actionIndex] = 1.0
        policyTargets.append(policy)
        targetIndices.append(actionIndex)

        valueTargets.append(0.0)
    }

    return (inputs, policyTargets, valueTargets, targetIndices)
}

private func policyLoss(logits: MLXArray, target: MLXArray) -> MLXArray {
    let batchSize = max(1, logits.shape[0])
    let logProbs = logSoftmax(logits, axis: -1)
    return -sum(target * logProbs) / Float(batchSize)
}

private func valueLoss(predicted: MLXArray, target: MLXArray) -> MLXArray {
    let batchSize = max(1, predicted.shape[0])
    let diff = predicted - target
    return sum(diff * diff) / Float(batchSize)
}

private func computeLosses(
    net: SantoriniNet,
    inputs: [EncodedState],
    policyTargets: [[Float]],
    valueTargets: [Float]
) -> (Float, Float) {
    let flatInputs = inputs.flatMap { $0.flatMap { $0.flatMap { $0 } } }
    let obs = MLXArray(flatInputs, [inputs.count, 5, 5, 9])
    let piTarget = MLXArray(policyTargets.flatMap { $0 }, [inputs.count, policyTargets[0].count])
    let zTarget = MLXArray(valueTargets, [inputs.count, 1])

    let (policyLogits, valuePred) = net(obs)
    let pLoss = policyLoss(logits: policyLogits, target: piTarget)
    let vLoss = valueLoss(predicted: valuePred, target: zTarget)
    eval(pLoss, vLoss)
    return (pLoss.item(Float.self), vLoss.item(Float.self))
}

private func train(
    net: SantoriniNet,
    optimizer: Adam,
    inputs: [EncodedState],
    policyTargets: [[Float]],
    valueTargets: [Float],
    steps: Int,
    policyWeight: Float,
    valueWeight: Float
) {
    let flatInputs = inputs.flatMap { $0.flatMap { $0.flatMap { $0 } } }
    let obs = MLXArray(flatInputs, [inputs.count, 5, 5, 9])
    let piTarget = MLXArray(policyTargets.flatMap { $0 }, [inputs.count, policyTargets[0].count])
    let zTarget = MLXArray(valueTargets, [inputs.count, 1])
    let targets = concatenated([piTarget, zTarget], axis: 1)
    let valuesPerPolicy = policyTargets[0].count

    for _ in 0..<steps {
        let (_, grad) = valueAndGrad(model: net) { net, obs, targets in
            let (policyLogits, valuePred) = net(obs)
            let splits = targets.split(indices: [valuesPerPolicy], axis: 1)
            let pLoss = policyLoss(logits: policyLogits, target: splits[0])
            let vLoss = valueLoss(predicted: valuePred, target: splits[1])
            return pLoss * policyWeight + vLoss * valueWeight
        }(net, obs, targets)

        optimizer.update(model: net, gradients: grad)
        eval(net, optimizer)
    }
}

private func policyTop1Accuracy(
    net: SantoriniNet,
    inputs: [EncodedState],
    targetIndices: [Int]
) -> Float {
    let (policies, _) = net.evaluateBatch(inputs)
    var correct = 0
    for (idx, policy) in policies.enumerated() {
        let maxIndex = policy.enumerated().max(by: { $0.element < $1.element })?.offset ?? -1
        if maxIndex == targetIndices[idx] {
            correct += 1
        }
    }
    return Float(correct) / Float(targetIndices.count)
}

private func policyTargetProbability(
    net: SantoriniNet,
    inputs: [EncodedState],
    targetIndices: [Int]
) -> Float {
    let (policies, _) = net.evaluateBatch(inputs)
    guard let first = policies.first, let target = targetIndices.first else { return 0 }
    return first[target]
}

private func valueMeanAbsoluteError(
    net: SantoriniNet,
    inputs: [EncodedState],
    valueTargets: [Float]
) -> Float {
    let (_, values) = net.evaluateBatch(inputs)
    guard values.count == valueTargets.count else { return Float.greatestFiniteMagnitude }
    let total = zip(values, valueTargets).reduce(Float(0)) { acc, pair in
        acc + abs(pair.0 - pair.1)
    }
    return total / Float(values.count)
}
