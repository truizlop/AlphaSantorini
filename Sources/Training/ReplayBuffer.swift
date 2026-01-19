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
        Array(samples.shuffled().prefix(batchSize))
    }

    var count: Int {
        samples.count
    }

    func checkDiversity() {
        guard samples.count > 0 else { return }
        let hashes = samples.map { $0.state.encoded().description.hashValue }
        let unique = Set(hashes).count

        print("Diversity: \(String(format: "%.1f", Float(unique) / Float(samples.count)))")
    }
}
