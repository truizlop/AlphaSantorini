//
//  Move.swift
//  AlphaSantorini
//
//  Created by Tomás Ruiz-López on 1/15/26.
//

public struct Move: Hashable {
    var id: WorkerID
    var moveDirection: Direction
    var buildDirection: Direction

    public func show() {
        print("Worker \(id.description) moved \(moveDirection.description) and built \(buildDirection.description)")
    }
}
