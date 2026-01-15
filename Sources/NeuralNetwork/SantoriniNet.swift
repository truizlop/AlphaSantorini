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
    let layer1 = Linear(175, 256)
    let layer2 = Linear(256, 256)
    let layer3 = Linear(256, 256)
    let policyHead = Linear(256, 153)
    let valueHead = Linear(256, 1)

    public func callAsFunction(_ input: MLXArray) -> (policy: MLXArray, value: MLXArray) {
        let o1 = relu(layer1(input))
        let o2 = relu(layer2(o1))
        let o3 = relu(layer3(o2))
        let policy = softmax(policyHead(o3))
        let value = tanh(valueHead(o3))
        return (policy: policy, value: value)
    }
}
