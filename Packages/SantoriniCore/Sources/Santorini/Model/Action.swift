//
//  Action.swift
//  AlphaSantorini
//
//  Created by Tomás Ruiz-López on 1/15/26.
//

public enum Action: Hashable {
    case placement(Placement)
    case move(Move)

    public func show() {
        switch self {
        case .placement(let placement):
            placement.show()
        case .move(let move):
            move.show()
        }
    }
}
