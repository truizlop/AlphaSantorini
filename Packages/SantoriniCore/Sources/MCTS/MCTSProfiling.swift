//
//  MCTSProfiling.swift
//  AlphaSantorini
//
//  Created by Tomás Ruiz-López on 2/2/26.
//

import Foundation

public final class MCTSProfiler: @unchecked Sendable {
    public static let shared = MCTSProfiler()

    private let lock = NSLock()
    private var isEnabled: Bool = false
    private var evaluateCount: Int = 0
    private var evaluateTime: TimeInterval = 0
    private var legalMovesCount: Int = 0
    private var legalMovesTime: TimeInterval = 0
    private var applyCount: Int = 0
    private var applyTime: TimeInterval = 0
    private var selectionCount: Int = 0
    private var selectionTime: TimeInterval = 0

    private init() {}

    public static var enabled: Bool {
        get { shared.enabledValue() }
        set { shared.setEnabled(newValue) }
    }

    public static func measureEvaluate<T>(_ work: () -> T) -> T {
        shared.measureEvaluate(work)
    }

    public static func measureLegalMoves<T>(_ work: () -> T) -> T {
        shared.measureLegalMoves(work)
    }

    public static func measureApply<T>(_ work: () -> T) -> T {
        shared.measureApply(work)
    }

    public static func measureSelection<T>(_ work: () -> T) -> T {
        shared.measureSelection(work)
    }

    public static func reportAndReset(prefix: String = "MCTS") -> String {
        shared.reportAndReset(prefix: prefix)
    }

    private func enabledValue() -> Bool {
        lock.lock()
        defer { lock.unlock() }
        return isEnabled
    }

    private func setEnabled(_ value: Bool) {
        lock.lock()
        isEnabled = value
        lock.unlock()
    }

    private func measureEvaluate<T>(_ work: () -> T) -> T {
        guard enabledValue() else { return work() }
        let start = CFAbsoluteTimeGetCurrent()
        let result = work()
        let end = CFAbsoluteTimeGetCurrent()
        lock.lock()
        evaluateCount += 1
        evaluateTime += (end - start)
        lock.unlock()
        return result
    }

    private func measureLegalMoves<T>(_ work: () -> T) -> T {
        guard enabledValue() else { return work() }
        let start = CFAbsoluteTimeGetCurrent()
        let result = work()
        let end = CFAbsoluteTimeGetCurrent()
        lock.lock()
        legalMovesCount += 1
        legalMovesTime += (end - start)
        lock.unlock()
        return result
    }

    private func measureApply<T>(_ work: () -> T) -> T {
        guard enabledValue() else { return work() }
        let start = CFAbsoluteTimeGetCurrent()
        let result = work()
        let end = CFAbsoluteTimeGetCurrent()
        lock.lock()
        applyCount += 1
        applyTime += (end - start)
        lock.unlock()
        return result
    }

    private func measureSelection<T>(_ work: () -> T) -> T {
        guard enabledValue() else { return work() }
        let start = CFAbsoluteTimeGetCurrent()
        let result = work()
        let end = CFAbsoluteTimeGetCurrent()
        lock.lock()
        selectionCount += 1
        selectionTime += (end - start)
        lock.unlock()
        return result
    }

    private func reportAndReset(prefix: String) -> String {
        lock.lock()
        defer { lock.unlock() }

        let report = """
        \(prefix) profiling:
          evaluate: \(evaluateCount) calls, \(String(format: "%.3f", evaluateTime))s
          legalMoves: \(legalMovesCount) calls, \(String(format: "%.3f", legalMovesTime))s
          apply: \(applyCount) calls, \(String(format: "%.3f", applyTime))s
          selection: \(selectionCount) calls, \(String(format: "%.3f", selectionTime))s
        """

        evaluateCount = 0
        evaluateTime = 0
        legalMovesCount = 0
        legalMovesTime = 0
        applyCount = 0
        applyTime = 0
        selectionCount = 0
        selectionTime = 0

        return report
    }
}
