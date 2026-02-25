import XCTest
@testable import Santorini

final class EncodingTests: XCTestCase {
    func testGameStateEncodingShapeAndHeightChannels() {
        let state = GameState()
        let encoded = state.encoded()
        XCTAssertEqual(encoded.count, 5)
        XCTAssertEqual(encoded.first?.count, 5)
        XCTAssertEqual(encoded.first?.first?.count, 9)

        for row in 0..<5 {
            for col in 0..<5 {
                XCTAssertEqual(encoded[row][col][0], 1.0)
                for channel in 1..<9 {
                    XCTAssertEqual(encoded[row][col][channel], 0.0)
                }
            }
        }
    }

    func testWorkerEncodingChannels() {
        var state = GameState()
        state.turn = .one
        state.workers = [
            Worker(id: .one, player: .one, position: Position(row: 0, column: 0)!),
            Worker(id: .two, player: .one, position: Position(row: 0, column: 1)!),
            Worker(id: .one, player: .two, position: Position(row: 4, column: 4)!),
            Worker(id: .two, player: .two, position: Position(row: 4, column: 3)!),
        ]
        state.phase = .play

        let encoded = state.encoded()
        XCTAssertEqual(encoded[0][0][5], 1.0)
        XCTAssertEqual(encoded[0][1][6], 1.0)
        XCTAssertEqual(encoded[4][4][7], 1.0)
        XCTAssertEqual(encoded[4][3][8], 1.0)
    }

    func testActionEncodingRoundTrip() {
        for row in 0..<5 {
            for col in 0..<5 {
                let placement = Placement(position: Position(row: row, column: col)!)
                let action = Action.placement(placement)
                let encoded = action.encoded()
                let decoded = Action.from(encoding: encoded)
                XCTAssertEqual(decoded, action)
            }
        }

        for workerId in [WorkerID.one, WorkerID.two] {
            for moveDirection in Direction.allCases {
                for buildDirection in Direction.allCases {
                    let move = Move(id: workerId, moveDirection: moveDirection, buildDirection: buildDirection)
                    let action = Action.move(move)
                    let encoded = action.encoded()
                    let decoded = Action.from(encoding: encoded)
                    XCTAssertEqual(decoded, action)
                }
            }
        }
    }
}
