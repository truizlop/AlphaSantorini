//
//  Worker.swift
//  AlphaSantorini
//
//  Created by Tomás Ruiz-López on 1/15/26.
//

public enum WorkerID: Hashable {
    case one
    case two

    var description: String {
        switch self {
        case .one: "1"
        case .two: "2"
        }
    }
}
