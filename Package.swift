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
        .package(path: "Packages/SantoriniCore"),
    ],
    targets: [
        // Targets are the basic building blocks of a package, defining a module or a test suite.
        // Targets can depend on other targets in this package and products from dependencies.
        .executableTarget(
            name: "AlphaSantorini",
            dependencies: [
                "Training",
                .product(name: "ArgumentParser", package: "swift-argument-parser"),
            ]
        ),
        .target(
            name: "NeuralNetwork",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
            ]
        ),
        .target(
            name: "Training",
            dependencies: [
                .product(name: "Santorini", package: "SantoriniCore"),
                .product(name: "MCTS", package: "SantoriniCore"),
                "NeuralNetwork",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXOptimizers", package: "mlx-swift"),
            ]
        ),
    ]
)
