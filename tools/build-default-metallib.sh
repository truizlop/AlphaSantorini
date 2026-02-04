#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="$ROOT_DIR/.build/mlx-metal"
MLX_SRC="$ROOT_DIR/.build/checkouts/mlx-swift/Source/Cmlx/mlx"
OUTPUT="$ROOT_DIR/default.metallib"
JOBS="${JOBS:-2}"

if [[ ! -f "$MLX_SRC/CMakeLists.txt" ]]; then
  echo "mlx-swift checkout not found at $MLX_SRC" >&2
  echo "Run 'swift package resolve' or 'swift build' to fetch dependencies first." >&2
  exit 1
fi

cmake -S "$MLX_SRC" -B "$BUILD_DIR" \
  -DMLX_BUILD_TESTS=OFF \
  -DMLX_BUILD_EXAMPLES=OFF \
  -DMLX_BUILD_BENCHMARKS=OFF \
  -DMLX_BUILD_PYTHON_BINDINGS=OFF \
  -DMLX_BUILD_SAFETENSORS=OFF \
  -DMLX_BUILD_GGUF=OFF

cmake --build "$BUILD_DIR" --target mlx-metallib -j "$JOBS"

cp "$BUILD_DIR/mlx/backend/metal/kernels/mlx.metallib" "$OUTPUT"

echo "Wrote $OUTPUT"
