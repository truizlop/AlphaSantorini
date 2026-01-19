//
//  SantoriniNet.swift
//  AlphaSantorini
//
//  Created by Tomás Ruiz-López on 1/15/26.
//

import MLX
import MLXNN

// Input: 175 values (one-hot encoding of 25 positions for 5 building levels, current player positions and other player positions)
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
        self.layer1 = Linear(175, hiddenDimension)
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
}
