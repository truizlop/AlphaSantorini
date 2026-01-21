// swift-tools-version: 6.2

import PackageDescription

let package = Package(
    name: "SantoriniWasm",
    platforms: [
        .macOS("13.3"),
    ],
    products: [
        .executable(name: "SantoriniWasm", targets: ["SantoriniWasm"]),
    ],
    dependencies: [
        .package(path: "../SantoriniCore"),
        .package(url: "https://github.com/swiftwasm/JavaScriptKit.git", from: "0.18.0"),
    ],
    targets: [
        .executableTarget(
            name: "SantoriniWasm",
            dependencies: [
                .product(name: "Santorini", package: "SantoriniCore"),
                .product(name: "MCTS", package: "SantoriniCore"),
                "JavaScriptKit",
            ],
            path: "Sources/SantoriniWasm"
        ),
    ]
)
