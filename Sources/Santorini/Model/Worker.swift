//
//  Worker.swift
//  AlphaSantorini
//
//  Created by Tomás Ruiz-López on 1/15/26.
//

public struct Worker {
    let id: WorkerID
    let player: Player
    private(set) var position: Position

    mutating func move(direction: Direction) {
        self.position = self.position.move(direction: direction)
    }
}
