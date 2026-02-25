import XCTest
import MLX
import MLXNN
@testable import NeuralNetwork

final class NeuralNetworkTests: XCTestCase {
    private func makeInput() -> [[[Float]]] {
        Array(
            repeating: Array(
                repeating: Array(repeating: 0.1, count: 9),
                count: 5
            ),
            count: 5
        )
    }

    func testOutputShapesAndRanges() {
        let net = SantoriniNet(filters: 16)
        let (policy, value) = net.evaluate(makeInput())
        XCTAssertEqual(policy.count, 153)
        let sum = policy.reduce(0, +)
        XCTAssertEqual(sum, 1.0, accuracy: 1e-4)
        XCTAssertTrue(policy.allSatisfy { $0 >= 0 })
        XCTAssertTrue(value >= -1.0 && value <= 1.0)
    }

    func testSaveLoadRoundTrip() throws {
        let net = SantoriniNet(filters: 16)
        let input = makeInput()
        let original = net.evaluate(input)

        let dir = URL(filePath: NSTemporaryDirectory()).appending(path: "santorini_net_tests")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        let fileURL = dir.appending(path: "net.safetensors")
        try net.save(to: fileURL)

        let loaded = SantoriniNet(filters: 16)
        try loaded.load(from: fileURL)
        let reloaded = loaded.evaluate(input)

        XCTAssertEqual(original.policy.count, reloaded.policy.count)
        for (a, b) in zip(original.policy, reloaded.policy) {
            XCTAssertEqual(a, b, accuracy: 1e-5)
        }
        XCTAssertEqual(original.value, reloaded.value, accuracy: 1e-5)
    }

    func testCopyWeightsDoesNotAlias() {
        let net = SantoriniNet(filters: 16)
        let copy = SantoriniNet(filters: 16)
        copy.copyWeights(from: net)

        let input = makeInput()
        let original = net.evaluate(input)
        let before = copy.evaluate(input)

        XCTAssertEqual(original.policy.count, before.policy.count)
        for (a, b) in zip(original.policy, before.policy) {
            XCTAssertEqual(a, b, accuracy: 1e-5)
        }
        XCTAssertEqual(original.value, before.value, accuracy: 1e-5)

        // Mutate original network parameters.
        let params = net.parameters().flattened()
        var mutated: [String: MLXArray] = [:]
        mutated.reserveCapacity(params.count)
        for (key, value) in params {
            mutated[key] = value + 0.01
        }
        let nested = ModuleParameters.unflattened(mutated)
        net.update(parameters: nested)
        eval(net)

        let afterOriginal = net.evaluate(input)
        let afterCopy = copy.evaluate(input)

        XCTAssertNotEqual(afterOriginal.value, afterCopy.value)
        for (a, b) in zip(before.policy, afterCopy.policy) {
            XCTAssertEqual(a, b, accuracy: 1e-5)
        }
        XCTAssertEqual(before.value, afterCopy.value, accuracy: 1e-5)
    }

    func testSoftmaxWithoutAxisIsGlobalFor2D() {
        let logits: [Float] = [0, 0, 10, 10]
        let logitsArray = MLXArray(logits, [2, 2])
        let noAxis = softmax(logitsArray)
        let rowSums = noAxis.sum(axis: 1).asArray(Float.self)
        XCTAssertEqual(rowSums.count, 2)
        XCTAssertLessThan(rowSums[0], 0.01)
        XCTAssertGreaterThan(rowSums[1], 0.99)
        XCTAssertEqual(noAxis.sum().item(Float.self), 1.0, accuracy: 1e-5)
    }

    func testSoftmaxAxisNormalizesPerRow() {
        let logits: [Float] = [0, 0, 10, 10]
        let logitsArray = MLXArray(logits, [2, 2])
        let perRow = softmax(logitsArray, axis: -1)
        let rowSums = perRow.sum(axis: 1).asArray(Float.self)
        XCTAssertEqual(rowSums.count, 2)
        XCTAssertEqual(rowSums[0], 1.0, accuracy: 1e-5)
        XCTAssertEqual(rowSums[1], 1.0, accuracy: 1e-5)
    }
}
