//
//  SantoriniNet.swift
//  AlphaSantorini
//
//  Created by Tomás Ruiz-López on 1/15/26.
//

import Foundation
import MLX
import MLXNN

// Input: 200 values (one-hot encoding of 25 positions for 5 building levels, current player worker positions, other player workers)
// Hidden layers: 3 Linear layers with 256 neurons, ReLU
// Output: policy head (153 logits = 25 for worker placements + 128 for worker-move-build directions)
//         value head (tanh value -1...1 expected for outcome)
public class SantoriniNet: Module {
    let layer1: Linear
    let layer2: Linear
    let layer3: Linear
    let policyHead: Linear
    let valueHead: Linear

    public init(hiddenDimension: Int) {
        self.layer1 = Linear(200, hiddenDimension)
        self.layer2 = Linear(hiddenDimension, hiddenDimension)
        self.layer3 = Linear(hiddenDimension, hiddenDimension)
        self.policyHead = Linear(hiddenDimension, 153)
        self.valueHead = Linear(hiddenDimension, 1)
    }

    public func callAsFunction(_ input: MLXArray) -> (policy: MLXArray, value: MLXArray) {
        let o1 = relu(layer1(input))
        let o2 = relu(layer2(o1))
        let o3 = relu(layer3(o2))
        let policy = softmax(policyHead(o3))
        let value = tanh(valueHead(o3))
        return (policy: policy, value: value)
    }

    public func evaluate(_ input: [Float]) -> (policy: [Float], value: Float) {
        let mlxInput = MLXArray(input)
        let (policy, value) = self(mlxInput)
        return (policy.asArray(Float.self), value.asArray(Float.self)[0])
    }

    public func evaluateBatch(_ inputs: [[Float]]) -> (policies: [[Float]], values: [Float]) {
        guard let first = inputs.first else { return ([], []) }
        let flat = inputs.flatMap { $0 }
        let mlxInput = MLXArray(flat, [inputs.count, first.count])
        let (policy, value) = self(mlxInput)

        let policyFlat = policy.asArray(Float.self)
        let valueFlat = value.asArray(Float.self)

        let policySize = policyFlat.count / inputs.count
        var policies: [[Float]] = []
        policies.reserveCapacity(inputs.count)
        var offset = 0
        for _ in 0 ..< inputs.count {
            policies.append(Array(policyFlat[offset ..< offset + policySize]))
            offset += policySize
        }

        let values: [Float]
        if valueFlat.count == inputs.count {
            values = valueFlat
        } else {
            let strideSize = max(1, valueFlat.count / inputs.count)
            values = stride(from: 0, to: valueFlat.count, by: strideSize).map { valueFlat[$0] }
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
