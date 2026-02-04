//
//  ActionEncoding.swift
//  AlphaSantorini
//
//  Created by Tomás Ruiz-López on 1/15/26.
//

extension Action {
    public static let total = Placement.total + Move.total

    public static func from(encoding: Int) -> Action? {
        guard encoding >= 0 && encoding < total else {
            return nil
        }
        if encoding < Placement.total {
            return Placement.from(encoding: encoding).map(Action.placement)
        } else {
            return Move.from(encoding: encoding - Placement.total).map(Action.move)
        }
    }

    public func encoded() -> Int {
        switch self {
        case .placement(let placement):
            placement.encoded()
        case .move(let move):
            Placement.total + move.encoded()
        }
    }
}

extension Placement {
    fileprivate static let total = 5 * 5 // 5 rows * 5 columns
    // Placements are encoded like:
    // 0: (0, 0)
    // 1: (0, 1)
    // 2: (0, 2)
    // ...
    // 5: (1, 0)
    // ...
    // 24: (4, 4)

    public static func from(encoding: Int) -> Placement? {
        guard encoding >= 0 && encoding < total else {
            return nil
        }
        let row = encoding / 5
        let column = encoding % 5
        return Position(row: row, column: column).map(Placement.init)
    }

    public func encoded() -> Int {
        position.row * 5 + position.column
    }
}

extension Move {
    fileprivate static let total = 2 * 8 * 8 // 2 workers * 8 move directions * 8 build directions
    // Moves are encoded like:
    // 0: worker 1, move nw, build nw
    // 1: worker 1, move nw, build n
    // 2: worker 1, move nw, build ne
    // ...
    // 8: worker 1, move n, build nw
    // ...
    // 63: worker 1, move se, build se
    // 64: worker 2, move nw, build nw
    // ...

    public static func from(encoding: Int) -> Move? {
        guard encoding >= 0 && encoding < total else {
            return nil
        }
        let workerID = encoding < 64 ? WorkerID.one : WorkerID.two
        let offset = workerID == .one ? 0 : -8
        let moveRawValue = encoding / 8 + offset
        let buildRawValue = encoding % 8
        guard let moveDirection = Direction(rawValue: moveRawValue),
              let buildDirection = Direction(rawValue: buildRawValue) else {
            return nil
        }
        return Move(
            id: workerID,
            moveDirection: moveDirection,
            buildDirection: buildDirection
        )
    }

    public func encoded() -> Int {
        let workerOffset = self.id == .one ? 0 : 64
        let moveOffset = 8 * moveDirection.rawValue
        let buildOffset = buildDirection.rawValue
        return workerOffset + moveOffset + buildOffset
    }
}
