//
//  ReplayBuffer.swift
//  AlphaSantorini
//
//  Created by Tomás Ruiz-López on 1/16/26.
//

class ReplayBuffer {
    private var samples: [TrainingSample] = []
    private let maxSize: Int

    init(maxSize: Int) {
        self.maxSize = maxSize
    }

    func add(_ newSamples: [TrainingSample]) {
        samples.append(contentsOf: newSamples)
        if samples.count > maxSize {
            samples.removeFirst(samples.count - maxSize)
        }
    }

    func sample(batchSize: Int) -> [TrainingSample] {
        let total = samples.count
        guard total > 0 else { return [] }
        let actual = min(batchSize, total)
        if actual == total {
            return samples.shuffled()
        }
        var indices = Set<Int>()
        indices.reserveCapacity(actual)
        while indices.count < actual {
            indices.insert(Int.random(in: 0 ..< total))
        }
        return indices.map { samples[$0] }
    }

    var count: Int {
        samples.count
    }

    func checkDiversity() {
        guard samples.count > 0 else { return }
        let hashes = samples.map(\.stateHash)
        let unique = Set(hashes).count

        print("Diversity: \(String(format: "%.1f", Float(unique) / Float(samples.count)))")
    }
}
