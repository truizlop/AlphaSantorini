//
//  Move.swift
//  AlphaSantorini
//
//  Created by Tomás Ruiz-López on 1/15/26.
//

public struct Move: Hashable, CustomStringConvertible {
    var id: WorkerID
    var moveDirection: Direction
    var buildDirection: Direction

    public var description: String {
        "move \(id.description) \(moveDirection.description) build \(buildDirection.description)"
    }

    public func show() {
        print("Worker \(id.description) moved \(moveDirection.description) and built \(buildDirection.description)")
    }
}
