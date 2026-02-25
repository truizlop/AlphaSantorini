//
//  SantoriniNet.swift
//  AlphaSantorini
//
//  Created by Tomás Ruiz-López on 1/15/26.
//

import Foundation
import MLX
import MLXNN

// Input: matrix of 9x5x5 - 9 planes (5 for building levels, 4 for workers) of 5x5 grids (board size)
// Hidden layers: 3 Linear layers with 256 neurons, ReLU
// Output: policy head (153 logits = 25 for worker placements + 128 for worker-move-build directions)
//         value head (tanh value -1...1 expected for outcome)
public class SantoriniNet: Module, @unchecked Sendable {
    let inputBlock: ConvolutionBlock
    let resTower: [ResidualBlock]
    let policyHead: PolicyHead
    let valueHead: ValueHead

    public init(
        filters: Int = 64,
        residualBlocks: Int = 5
    ) {
        self.inputBlock = ConvolutionBlock(
            inChannels: 9,
            outChannels: filters
        )
        self.resTower = (0 ..< residualBlocks).map { _ in
            ResidualBlock(channels: filters)
        }
        self.policyHead = PolicyHead(inChannels: filters)
        self.valueHead = ValueHead(
            inChannels: filters,
            hiddenDimension: 64
        )
    }

    public convenience init(hiddenDimension: Int) {
        self.init(filters: hiddenDimension)
    }

    public func callAsFunction(_ input: MLXArray) -> (policy: MLXArray, value: MLXArray) {
        var out = inputBlock(input)
        for block in resTower {
            out = block(out)
        }
        let policy = policyHead(out)
        let value = valueHead(out)
        return (policy, value)
    }

    public func evaluate(_ input: [[[Float]]]) -> (policy: [Float], value: Float) {
        let flat = input.flatMap { $0.flatMap { $0 } }
        let mlxInput = MLXArray(flat, [1, 5, 5, 9])
        let (policyLogits, value) = self(mlxInput)
        let policy = softmax(policyLogits, axis: -1)
        return (policy.asArray(Float.self), value.item(Float.self))
    }

    public func evaluateBatch(_ inputs: [[[[Float]]]]) -> (policies: [[Float]], values: [Float]) {
        guard !inputs.isEmpty else { return ([], []) }
        let flat = inputs.flatMap { $0.flatMap { $0.flatMap { $0 } } }
        let mlxInput = MLXArray(flat, [inputs.count, 5, 5, 9])
        let (policyLogits, value) = self(mlxInput)
        let policy = softmax(policyLogits, axis: -1).asArray(Float.self)
        let values = value.reshaped(inputs.count).asArray(Float.self)

        let policySize = 153
        var policies: [[Float]] = []
        policies.reserveCapacity(inputs.count)
        var offset = 0
        for _ in 0..<inputs.count {
            policies.append(Array(policy[offset..<(offset + policySize)]))
            offset += policySize
        }
        return (policies, values)
    }

    public func copyWeights(from other: SantoriniNet) {
        let sourceParams = other.parameters()
        self.update(parameters: sourceParams)
        eval(self)
    }

    public func save(to url: URL) throws {
        let params = self.parameters().flattened()
        let dict = Dictionary(uniqueKeysWithValues: params)
        try MLX.save(arrays: dict, url: url)
    }

    public func load(from url: URL) throws {
        let arrays = try MLX.loadArrays(url: url)
        let nestedParams = ModuleParameters.unflattened(arrays)
        self.update(parameters: nestedParams)
        eval(self)
    }
}

// MARK: Convolution Block (Conv2d + BatchNorm + ReLU)

class ConvolutionBlock: Module, UnaryLayer {
    let conv: Conv2d
    let norm: BatchNorm

    init(
        inChannels: Int,
        outChannels: Int,
        kernelSize: Int = 3
    ) {
        self.conv = Conv2d(
            inputChannels: inChannels,
            outputChannels: outChannels,
            kernelSize: IntOrPair(kernelSize),
            padding: IntOrPair(kernelSize / 2)
        )
        self.norm = BatchNorm(featureCount: outChannels)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        relu(norm(conv(x)))
    }
}

// MARK: Residual Block

class ResidualBlock: Module, UnaryLayer {
    let conv1: Conv2d
    let norm1: BatchNorm
    let conv2: Conv2d
    let norm2: BatchNorm

    init(channels: Int) {
        self.conv1 = Conv2d(
            inputChannels: channels,
            outputChannels: channels,
            kernelSize: IntOrPair(3),
            padding: IntOrPair(1)
        )
        self.norm1 = BatchNorm(featureCount: channels)
        self.conv2 = Conv2d(
            inputChannels: channels,
            outputChannels: channels,
            kernelSize: IntOrPair(3),
            padding: IntOrPair(1)
        )
        self.norm2 = BatchNorm(featureCount: channels)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let o1 = relu(norm1(conv1(x)))
        let o2 = norm2(conv2(o1))
        return relu(x + o2)
    }
}

// MARK: Policy Head

class PolicyHead: Module, UnaryLayer {
    let conv: Conv2d
    let norm: BatchNorm
    let linear: Linear

    init(inChannels: Int) {
        self.conv = Conv2d(
            inputChannels: inChannels,
            outputChannels: 2,
            kernelSize: IntOrPair(1)
        )
        self.norm = BatchNorm(featureCount: 2)
        // 2 output channels, 5x5 board
        self.linear = Linear(2 * 5 * 5, 153)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let batchSize = x.dim(0)
        let output = relu(norm(conv(x)))
        let reshapedOutput = output.reshaped(batchSize, 2 * 5 * 5)
        return linear(reshapedOutput) // Raw logits, apply softmax later
    }
}

// MARK: Value Head

class ValueHead: Module, UnaryLayer {
    let conv: Conv2d
    let norm: BatchNorm
    let linear1: Linear
    let linear2: Linear

    init(
        inChannels: Int,
        hiddenDimension: Int
    ) {
        self.conv = Conv2d(
            inputChannels: inChannels,
            outputChannels: 1,
            kernelSize: IntOrPair(1)
        )
        self.norm = BatchNorm(featureCount: 1)
        self.linear1 = Linear(5 * 5, hiddenDimension)
        self.linear2 = Linear(hiddenDimension, 1)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let batchSize = x.dim(0)
        let o1 = relu(norm(conv(x)))
        let reshapedO1 = o1.reshaped(batchSize, 5 * 5)
        let o2 = relu(linear1(reshapedO1))
        return tanh(linear2(o2))
    }
}
