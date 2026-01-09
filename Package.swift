// swift-tools-version: 6.2
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "AlphaSantorini",
    platforms: [
        .macOS("13.3"),
    ],
    dependencies: [
        .package(url: "https://github.com/apple/swift-argument-parser.git", from: "1.2.0"),
        .package(url: "https://github.com/ml-explore/mlx-swift", from: "0.0.0"),
    ],
    targets: [
        // Targets are the basic building blocks of a package, defining a module or a test suite.
        // Targets can depend on other targets in this package and products from dependencies.
        .executableTarget(
            name: "AlphaSantorini",
            dependencies: [
                "GameEngine",
                .product(name: "ArgumentParser", package: "swift-argument-parser"),
            ]
        ),
        .target(name: "Santorini"),
        .target(name: "MCTS"),
        .target(
            name: "NeuralNetwork",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
            ]
        ),
        .target(
            name: "GameEngine",
            dependencies: [
                "Santorini",
                "MCTS",
                "NeuralNetwork",
            ]
        ),
    ]
)
