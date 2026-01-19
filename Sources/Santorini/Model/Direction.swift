//
//  Direction.swift
//  AlphaSantorini
//
//  Created by Tomás Ruiz-López on 1/15/26.
//

public enum Direction: Int, CaseIterable, Hashable {
    case nw = 0
    case n = 1
    case ne = 2
    case w = 3
    case e = 4
    case sw = 5
    case s = 6
    case se = 7

    var rowDelta: Int {
        switch self {
        case .nw: -1
        case .n: -1
        case .ne: -1
        case .w: 0
        case .e: 0
        case .sw: 1
        case .s: 1
        case .se: 1
        }
    }

    var columnDelta: Int {
        switch self {
        case .nw: -1
        case .n: 0
        case .ne: 1
        case .w: -1
        case .e: 1
        case .sw: -1
        case .s: 0
        case .se: 1
        }
    }

    var description: String {
        switch self {
        case .nw:
            "NW"
        case .n:
            "N"
        case .ne:
            "NE"
        case .w:
            "W"
        case .e:
            "E"
        case .sw:
            "SW"
        case .s:
            "S"
        case .se:
            "SE"
        }
    }
}
