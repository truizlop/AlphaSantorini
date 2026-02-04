//
//  Worker.swift
//  AlphaSantorini
//
//  Created by Tomás Ruiz-López on 1/15/26.
//

public enum WorkerID: Hashable, CustomStringConvertible {
    case one
    case two

    public var description: String {
        switch self {
        case .one: "1"
        case .two: "2"
        }
    }
}
