import XCTest
@testable import Santorini

final class SymmetryTests: XCTestCase {
    func testPositionAndDirectionRoundTripWithInverseSymmetry() {
        for symmetry in BoardSymmetry.allCases {
            let inverse = symmetry.inverse
            for row in 0..<5 {
                for column in 0..<5 {
                    let position = Position(row: row, column: column)!
                    let roundTrip = position
                        .transformed(by: symmetry)
                        .transformed(by: inverse)
                    XCTAssertEqual(roundTrip, position)
                }
            }

            for direction in Direction.allCases {
                let roundTrip = direction
                    .transformed(by: symmetry)
                    .transformed(by: inverse)
                XCTAssertEqual(roundTrip, direction)
            }
        }
    }

    func testActionTransformIsBijectionForEachSymmetry() {
        for symmetry in BoardSymmetry.allCases {
            var mapped = Set<Int>()
            mapped.reserveCapacity(Action.total)

            for encoding in 0..<Action.total {
                guard let action = Action.from(encoding: encoding) else {
                    XCTFail("Expected valid action for encoding \(encoding)")
                    return
                }
                let transformed = action.transformed(by: symmetry)
                mapped.insert(transformed.encoded())
            }

            XCTAssertEqual(mapped.count, Action.total)
        }
    }

    func testLegalActionsAreEquivariantUnderSymmetry() {
        var board = Board()
        board.board[0][0] = .height1
        board.board[0][1] = .height2
        board.board[1][3] = .height3
        board.board[2][2] = .dome
        board.board[4][1] = .height1

        let workers = [
            Worker(id: .one, player: .one, position: Position(row: 1, column: 1)!),
            Worker(id: .two, player: .one, position: Position(row: 3, column: 0)!),
            Worker(id: .one, player: .two, position: Position(row: 4, column: 4)!),
            Worker(id: .two, player: .two, position: Position(row: 0, column: 3)!),
        ]
        let state = GameState(board: board, turn: .one, workers: workers, phase: .play)
        let legalActions = Set(state.legalActions)
        XCTAssertFalse(legalActions.isEmpty)

        for symmetry in BoardSymmetry.allCases {
            let transformedState = state.transformed(by: symmetry)
            let transformedLegalActions = Set(transformedState.legalActions)
            let expected = Set(legalActions.map { $0.transformed(by: symmetry) })
            XCTAssertEqual(transformedLegalActions, expected)
        }
    }
}
