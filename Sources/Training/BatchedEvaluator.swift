//
//  BatchedEvaluator.swift
//  AlphaSantorini
//
//  Created by Tomás Ruiz-López on 2/26/26.
//

import Foundation
import NeuralNetwork

public actor BatchedEvaluator {
    private let net: SantoriniNet
    private let maxBatchSize: Int
    private let timeoutMicroseconds: UInt64
    private var pending: [EvaluationRequest] = []
    private var flushTask: Task<Void, Never>?
    private(set) var totalBatches: Int = 0
    private(set) var totalEvaluations: Int = 0

    public init(
        net: SantoriniNet,
        maxBatchSize: Int = 64,
        timeoutMicroseconds: UInt64 = 100
    ) {
        self.net = net
        self.maxBatchSize = maxBatchSize
        self.timeoutMicroseconds = timeoutMicroseconds
    }

    public func evaluate(encodedState: [[[Float]]]) async -> (policy: [Float], value: Float) {
        await withCheckedContinuation { continuation in
            pending.append(EvaluationRequest(
                encodedState: encodedState,
                continuation: continuation
            ))
            if pending.count >= maxBatchSize {
                flushNow()
            } else if pending.count == 1 {
                scheduleFlush()
            }
        }
    }

    public func diagnostics() -> (totalBatches: Int, totalEvaluations: Int, avgBatchSize: Float) {
        let avg = totalBatches > 0 ? Float(totalEvaluations) / Float(totalBatches) : 0
        return (totalBatches, totalEvaluations, avg)
    }

    private func scheduleFlush() {
        flushTask?.cancel()
        flushTask = Task { [timeoutMicroseconds] in
            try? await Task.sleep(nanoseconds: UInt64(timeoutMicroseconds) * 1000)
            guard !Task.isCancelled else { return }
            await self.flushIfNeeded()
        }
    }

    private func flushIfNeeded() {
        guard !pending.isEmpty else { return }
        flushNow()
    }

    private func flushNow() {
        let batch = pending
        pending = []
        flushTask?.cancel()
        flushTask = nil

        totalBatches += 1
        totalEvaluations += batch.count

        let inputs = batch.map { $0.encodedState }
        let (policies, values) = net.evaluateBatch(inputs)

        for (index, request) in batch.enumerated() {
            let policy = index < policies.count ? policies[index] : []
            let value = index < values.count ? values[index] : 0
            request.continuation.resume(returning: (policy, value))
        }
    }
}

private struct EvaluationRequest {
    let encodedState: [[[Float]]]
    let continuation: CheckedContinuation<(policy: [Float], value: Float), Never>
}
