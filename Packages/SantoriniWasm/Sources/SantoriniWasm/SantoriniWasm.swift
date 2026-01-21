import JavaScriptKit
import Santorini

private final class GameStore {
    private var nextHandle: Int = 1
    private var states: [Int: GameState] = [:]

    func create() -> Int {
        let handle = allocate(state: GameState())
        return handle
    }

    func clone(handle: Int) -> Int {
        guard let state = states[handle] else { return -1 }
        return allocate(state: state)
    }

    func state(for handle: Int) -> GameState? {
        states[handle]
    }

    func replace(handle: Int, state: GameState) {
        states[handle] = state
    }

    func add(state: GameState) -> Int {
        allocate(state: state)
    }

    func release(handle: Int) {
        states.removeValue(forKey: handle)
    }

    private func allocate(state: GameState) -> Int {
        let handle = nextHandle
        nextHandle += 1
        states[handle] = state
        return handle
    }
}

@MainActor
@main
enum SantoriniWasm {
    private static let store = GameStore()
    private static var closures: [JSClosure] = []

    static func main() {
        let exports = JSObject.global.Object.function!.new()

        exportFunction(named: "createGame", on: exports) { _ in
            JSValue.number(Double(store.create()))
        }

        exportFunction(named: "cloneState", on: exports) { args in
            guard let handle = args.first?.number else { return JSValue.number(-1) }
            return JSValue.number(Double(store.clone(handle: Int(handle))))
        }

        exportFunction(named: "releaseState", on: exports) { args in
            guard let handle = args.first?.number else { return .undefined }
            store.release(handle: Int(handle))
            return .undefined
        }

        exportFunction(named: "isTerminal", on: exports) { args in
            guard let handle = args.first?.number,
                  let state = store.state(for: Int(handle)) else {
                return JSValue.boolean(false)
            }
            return JSValue.boolean(state.isOver)
        }

        exportFunction(named: "winner", on: exports) { args in
            guard let handle = args.first?.number,
                  let state = store.state(for: Int(handle)) else {
                return JSValue.number(0)
            }
            guard let winner = state.winner else { return JSValue.number(0) }
            return JSValue.number(winner == .one ? 1 : -1)
        }

        exportFunction(named: "legalActions", on: exports) { args in
            guard let handle = args.first?.number,
                  let state = store.state(for: Int(handle)) else {
                return [Double]().jsValue
            }
            let actions = state.legalActions.map { Double($0.encoded()) }
            return actions.jsValue
        }

        exportFunction(named: "applyAction", on: exports) { args in
            guard args.count >= 2,
                  let handle = args[0].number,
                  let actionID = args[1].number,
                  let state = store.state(for: Int(handle)),
                  let action = Action.from(encoding: Int(actionID)) else {
                return JSValue.number(-1)
            }
            var newState = state
            switch action {
            case .placement(let placement):
                newState.placement(placement)
            case .move(let move):
                newState.play(move: move)
            }
            return JSValue.number(Double(store.add(state: newState)))
        }

        exportFunction(named: "getStateSummary", on: exports) { args in
            guard let handle = args.first?.number,
                  let state = store.state(for: Int(handle)) else {
                return JSValue.undefined
            }
            return makeSummary(for: state)
        }

        exportFunction(named: "encodeState", on: exports) { args in
            guard let handle = args.first?.number,
                  let state = store.state(for: Int(handle)) else {
                return [Double]().jsValue
            }
            return state.encoded().map(Double.init).jsValue
        }

        JSObject.global.SantoriniWasm = exports.jsValue
    }

    private static func exportFunction(
        named name: String,
        on object: JSObject,
        _ body: @escaping ([JSValue]) -> JSValue
    ) {
        let closure = JSClosure { arguments in
            body(arguments)
        }
        closures.append(closure)
        object[name] = closure.jsValue
    }

    private static func makeSummary(for state: GameState) -> JSValue {
        let summary = JSObject.global.Object.function!.new()
        summary["phase"] = JSValue.string(state.phase == .placement ? "placement" : "play")
        summary["turn"] = JSValue.string(state.turn == .one ? "one" : "two")

        let boardHeights: [Double] = (0..<5).flatMap { row in
            (0..<5).map { col in
                let position = Position(row: row, column: col)!
                return Double(state.board.building(at: position).rawValue)
            }
        }
        summary["boardHeights"] = boardHeights.jsValue

        let workerValues: [JSValue] = state.workers.map { worker in
            let item = JSObject.global.Object.function!.new()
            item["player"] = JSValue.string(worker.player == .one ? "one" : "two")
            item["id"] = JSValue.string(worker.id == .one ? "one" : "two")
            item["row"] = JSValue.number(Double(worker.position.row))
            item["col"] = JSValue.number(Double(worker.position.column))
            return item.jsValue
        }
        summary["workers"] = workerValues.jsValue

        return summary.jsValue
    }
}
