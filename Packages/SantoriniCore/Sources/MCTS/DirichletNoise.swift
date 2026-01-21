//
//  DirichletNoise.swift
//  AlphaSantorini
//
//  Created by Tomás Ruiz-López on 1/15/26.
//

import Foundation

public struct DirichletNoise: Sendable {
    let epsilon: Float
    let alpha: Float

    public init(epsilon: Float, alpha: Float) {
        self.epsilon = epsilon
        self.alpha = alpha
    }
}

extension MCTSNode {
    func addDirichletNoise(
        _ noise: DirichletNoise
    ) {
        guard !children.isEmpty else { return }

        let alphas = Array(repeating: noise.alpha, count: children.count)
        let sample = sampleDirichlet(alpha: alphas)

        for (i, child) in children.enumerated() {
            child.prior = (1 - noise.epsilon) * child.prior + noise.epsilon * sample[i]
        }
    }
}

func sampleDirichlet(alpha: [Float]) -> [Float] {
    let gamma = alpha.map { a in
        sampleGamma(shape: a)
    }
    let sum = gamma.reduce(0, +)
    return gamma.map { g in g / sum }
}

func sampleGamma(shape: Float) -> Float {
    precondition(shape > 0, "Gamma shape must be > 0")
    if shape < 1 {
        // Gamma(k) = Gamma(k + 1) * U^(1/k) for k in (0, 1)
        let u = Float.random(in: Float.leastNonzeroMagnitude ..< 1)
        return sampleGamma(shape: shape + 1) * pow(u, 1.0 / shape)
    }
    let d = shape - 1.0 / 3.0
    let c = 1.0 / sqrt(9.0 * d)
    while true {
        var x: Float
        var v: Float

        repeat {
            x = sampleStandardNormal()
            v = 1.0 + c * x
        } while v <= 0

        v = pow(v, 3)
        let u = Float.random(in: 0 ..< 1)
        if u < 1.0 - 0.0331 * pow(x, 4) {
            return d * v
        }

        if log(u) < 0.5 * pow(x, 2) + d * (1.0 - v - log(v)) {
            return d * v
        }
    }
}

func sampleStandardNormal() -> Float {
    let u1 = Float.random(in: Float.leastNonzeroMagnitude ..< 1)
    let u2 = Float.random(in: 0 ..< 1)
    return sqrt(-2.0 * log(u1)) * cos(2.0 * .pi * u2)
}
