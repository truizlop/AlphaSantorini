//
//  Building.swift
//  AlphaSantorini
//
//  Created by Tomás Ruiz-López on 1/15/26.
//

public enum Building: Int {
    case height0
    case height1
    case height2
    case height3
    case dome

    var next: Building {
        switch self {
        case .height0: .height1
        case .height1: .height2
        case .height2: .height3
        case .height3: .dome
        case .dome: .dome
        }
    }

    func canMoveTowards(building: Building) -> Bool {
        if building == .dome {
            // Not moving onto domes
            return false
        } else if self.rawValue >= building.rawValue {
            // Can move down or same level
            return true
        } else if building.rawValue - self.rawValue == 1 {
            // Can move up only one level
            return true
        } else {
            return false
        }
    }
}
