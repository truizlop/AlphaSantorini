import XCTest
@testable import Santorini

final class SantoriniRulesTests: XCTestCase {
    func testPlacementAlternatesAndPhaseTransition() {
        var state = GameState()
        XCTAssertEqual(state.phase, .placement)
        XCTAssertEqual(state.turn, .one)
        XCTAssertEqual(state.workers.count, 0)

        state.placement(Placement(position: Position(row: 0, column: 0)!))
        XCTAssertEqual(state.workers.count, 1)
        XCTAssertEqual(state.turn, .two)

        state.placement(Placement(position: Position(row: 0, column: 1)!))
        XCTAssertEqual(state.workers.count, 2)
        XCTAssertEqual(state.turn, .one)

        state.placement(Placement(position: Position(row: 4, column: 4)!))
        XCTAssertEqual(state.workers.count, 3)
        XCTAssertEqual(state.turn, .two)

        state.placement(Placement(position: Position(row: 4, column: 3)!))
        XCTAssertEqual(state.workers.count, 4)
        XCTAssertEqual(state.phase, .play)
        XCTAssertEqual(state.turn, .one)

        state.placement(Placement(position: Position(row: 2, column: 2)!))
        XCTAssertEqual(state.workers.count, 4)
    }

    func testLegalPlacementsExcludeOccupied() {
        var state = GameState()
        state.placement(Placement(position: Position(row: 2, column: 2)!))
        state.placement(Placement(position: Position(row: 0, column: 0)!))
        let placements = state.legalPlacements
        XCTAssertFalse(placements.contains { $0.position == Position(row: 2, column: 2)! })
        XCTAssertFalse(placements.contains { $0.position == Position(row: 0, column: 0)! })
    }

    func testLegalMovesRejectDomesAndTooHigh() {
        var board = Board()
        board.board[2][2] = .height0
        board.board[2][3] = .dome
        board.board[3][2] = .height2
        board.board[1][2] = .height0

        let worker = Worker(id: .one, player: .one, position: Position(row: 2, column: 2)!)
        let opponent = Worker(id: .one, player: .two, position: Position(row: 4, column: 4)!)

        let state = GameState(board: board, turn: .one, workers: [worker, opponent], phase: .play)
        let moves = state.legalMoves

        let domeMove = Move(id: .one, moveDirection: .e, buildDirection: .n)
        XCTAssertFalse(moves.contains(domeMove))

        let tooHighMove = Move(id: .one, moveDirection: .s, buildDirection: .n)
        XCTAssertFalse(moves.contains(tooHighMove))

        let legalMove = Move(id: .one, moveDirection: .n, buildDirection: .n)
        XCTAssertTrue(moves.contains(legalMove))
    }

    func testWinnerOnHeight3() {
        var board = Board()
        board.board[1][1] = .height3
        let worker = Worker(id: .one, player: .one, position: Position(row: 1, column: 1)!)
        let state = GameState(board: board, turn: .two, workers: [worker], phase: .play)
        XCTAssertEqual(state.winner, .one)
        XCTAssertTrue(state.isOver)
    }

    func testWinnerWhenNoLegalMoves() {
        var board = Board()
        for row in 0..<5 {
            for col in 0..<5 {
                board.board[row][col] = .dome
            }
        }
        board.board[0][0] = .height0
        board.board[4][4] = .height0

        let workerOne = Worker(id: .one, player: .one, position: Position(row: 0, column: 0)!)
        let workerTwo = Worker(id: .two, player: .one, position: Position(row: 4, column: 4)!)
        let state = GameState(board: board, turn: .one, workers: [workerOne, workerTwo], phase: .play)

        XCTAssertTrue(state.legalMoves.isEmpty)
        XCTAssertEqual(state.winner, .two)
    }

    func testBuildingMovementRules() {
        XCTAssertTrue(Building.height0.canMoveTowards(building: .height0))
        XCTAssertTrue(Building.height0.canMoveTowards(building: .height1))
        XCTAssertFalse(Building.height0.canMoveTowards(building: .height2))
        XCTAssertFalse(Building.height2.canMoveTowards(building: .dome))
    }

    func testPositionMovementBounds() {
        let corner = Position(row: 0, column: 0)!
        XCTAssertFalse(corner.canMove(direction: .nw))
        XCTAssertTrue(corner.canMove(direction: .se))

        let moved = corner.move(direction: .nw)
        XCTAssertEqual(moved, Position(row: 0, column: 0)!)
    }
}
