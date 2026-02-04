//
//  Placement.swift
//  AlphaSantorini
//
//  Created by Tomás Ruiz-López on 1/15/26.
//

public struct Placement: Hashable, CustomStringConvertible {
    var position: Position

    public var description: String {
        "place (\(position.row), \(position.column))"
    }

    public func show() {
        print("Worker placed at (\(position.row), \(position.column))")
    }
}
