//
//  Placement.swift
//  AlphaSantorini
//
//  Created by Tomás Ruiz-López on 1/15/26.
//

public struct Placement: Hashable {
    var position: Position

    public func show() {
        print("Worker placed at (\(position.row), \(position.column))")
    }
}
