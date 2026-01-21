// swift-tools-version: 6.2

import PackageDescription

let package = Package(
    name: "SantoriniCore",
    platforms: [
        .macOS("13.3"),
    ],
    products: [
        .library(name: "Santorini", targets: ["Santorini"]),
        .library(name: "MCTS", targets: ["MCTS"]),
    ],
    targets: [
        .target(
            name: "Santorini",
            path: "Sources/Santorini"
        ),
        .target(
            name: "MCTS",
            dependencies: [],
            path: "Sources/MCTS"
        ),
    ]
)
