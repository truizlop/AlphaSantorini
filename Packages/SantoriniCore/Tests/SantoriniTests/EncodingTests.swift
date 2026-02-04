import XCTest
@testable import Santorini

final class EncodingTests: XCTestCase {
    func testGameStateEncodingShapeAndHeightChannels() {
        let state = GameState()
        let encoded = state.encoded()
        XCTAssertEqual(encoded.count, 200)

        let height0 = encoded[0..<25]
        let otherHeights = encoded[25..<125]
        XCTAssertTrue(height0.allSatisfy { $0 == 1.0 })
        XCTAssertTrue(otherHeights.allSatisfy { $0 == 0.0 })
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
        let indexCurrentOne = 125 + 0
        let indexCurrentTwo = 150 + 1
        let indexOtherOne = 175 + (4 * 5 + 4)
        let indexOtherTwo = 175 + (4 * 5 + 3)

        XCTAssertEqual(encoded[indexCurrentOne], 1.0)
        XCTAssertEqual(encoded[indexCurrentTwo], 1.0)
        XCTAssertEqual(encoded[indexOtherOne], 1.0)
        XCTAssertEqual(encoded[indexOtherTwo], 1.0)
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
