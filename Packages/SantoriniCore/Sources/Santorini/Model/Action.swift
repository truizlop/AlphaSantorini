//
//  Action.swift
//  AlphaSantorini
//
//  Created by Tomás Ruiz-López on 1/15/26.
//

public enum Action: Hashable, CustomStringConvertible {
    case placement(Placement)
    case move(Move)

    public var description: String {
        switch self {
        case .placement(let placement):
            return placement.description
        case .move(let move):
            return move.description
        }
    }

    public func show() {
        switch self {
        case .placement(let placement):
            placement.show()
        case .move(let move):
            move.show()
        }
    }
}
