//
//  Player.swift
//  AlphaSantorini
//
//  Created by Tomás Ruiz-López on 1/15/26.
//

public enum Player: Sendable {
    case one
    case two

    var other: Player {
        switch self {
        case .one: .two
        case .two: .one
        }
    }
}
