.PHONY: help build run-release wasm onnx onnx-verify web-install web-build web-dev clean

SWIFT ?= xcrun --toolchain $(XCODE_TOOLCHAIN) swift
DEVELOPER_DIR ?= /Applications/Xcode.app/Contents/Developer
XCODE_TOOLCHAIN ?= XcodeDefault
TOOLCHAINS ?=
PYTHON ?= python3
NPM ?= npm

WASM_SDK ?=
WASM_OUTPUT ?= web/public/wasm
MODEL_OUT ?= web/public/models/santorini.onnx
CHECKPOINT ?=

SWIFT_CMD = $(if $(TOOLCHAINS),env TOOLCHAINS=$(TOOLCHAINS) DEVELOPER_DIR=$(DEVELOPER_DIR) $(SWIFT),env DEVELOPER_DIR=$(DEVELOPER_DIR) $(SWIFT))

help:
	@echo "Targets:"
	@echo "  build        Build native Swift package"
	@echo "  run-release  Build and run Release executable"
	@echo "  wasm         Build SwiftWasm bundle (auto-detects WASM_SDK)"
	@echo "  onnx         Export ONNX model (requires CHECKPOINT=...)" 
	@echo "  onnx-verify  Verify ONNX model with onnxruntime-node"
	@echo "  web-install  Install web dependencies"
	@echo "  web-build    Build web UI (Vite)"
	@echo "  web-dev      Run web dev server (Vite)"
	@echo "  clean        Remove web build output"
	@echo ""
	@echo "Optional variables:"
	@echo "  TOOLCHAINS=<swift.org toolchain id> (for WASM builds)"

build:
	$(SWIFT_CMD) build

run-release:
	$(SWIFT_CMD) run -c release AlphaSantorini

wasm:
	@TOOLCHAINS_ENV="$(TOOLCHAINS)"; \
	TOOLCHAIN_DIR=""; \
	if [ -n "$$TOOLCHAINS_ENV" ]; then \
		if [ -d "$$TOOLCHAINS_ENV" ]; then \
			TOOLCHAIN_DIR="$$TOOLCHAINS_ENV"; \
		else \
			for DIR in /Library/Developer/Toolchains "$$HOME/Library/Developer/Toolchains"; do \
				if [ -d "$$DIR/$${TOOLCHAINS_ENV}.xctoolchain" ]; then \
					TOOLCHAIN_DIR="$$DIR/$${TOOLCHAINS_ENV}.xctoolchain"; \
					break; \
				fi; \
			done; \
		fi; \
	fi; \
	if [ -z "$$TOOLCHAIN_DIR" ]; then \
		for DIR in /Library/Developer/Toolchains "$$HOME/Library/Developer/Toolchains"; do \
			if [ -d "$$DIR" ]; then \
				DEV_CANDIDATES=$$(ls "$$DIR" | grep -E 'DEVELOPMENT-SNAPSHOT' || true); \
				if [ -n "$$DEV_CANDIDATES" ]; then \
					PICK=$$(printf "%s\n" $$DEV_CANDIDATES | sort | tail -n 1); \
					TOOLCHAIN_DIR="$$DIR/$$PICK"; \
					break; \
				fi; \
				CANDIDATES=$$(ls "$$DIR" | grep -E 'swift-latest|swift-' || true); \
				if [ -n "$$CANDIDATES" ]; then \
					PICK=$$(printf "%s\n" $$CANDIDATES | sort | tail -n 1); \
					TOOLCHAIN_DIR="$$DIR/$$PICK"; \
					break; \
				fi; \
			fi; \
		done; \
	fi; \
	if [ -z "$$TOOLCHAIN_DIR" ]; then \
		echo "No Swift.org toolchain found. Install one from https://www.swift.org/download/ or set TOOLCHAINS=..."; \
		exit 1; \
	fi; \
	SWIFT_CMD="$$TOOLCHAIN_DIR/usr/bin/swift"; \
	if [ ! -x "$$SWIFT_CMD" ]; then \
		echo "Swift binary not found at $$SWIFT_CMD"; \
		exit 1; \
	fi; \
	TOOLCHAIN_NAME=$$(basename "$$TOOLCHAIN_DIR"); \
	TOOLCHAIN_TAG=$$(printf "%s\n" "$$TOOLCHAIN_NAME" | sed 's/\\.xctoolchain$$//' | sed -n 's/^swift-//p'); \
	WASM_SDK_ENV="$(WASM_SDK)"; \
	if [ -z "$$WASM_SDK_ENV" ]; then \
		SDK_LIST=$$($$SWIFT_CMD sdk list 2>/dev/null | tr -d '\r'); \
		if [ -n "$$TOOLCHAIN_TAG" ]; then \
			WASM_SDK_ENV=$$(printf "%s\n" "$$SDK_LIST" | grep -E "^$${TOOLCHAIN_TAG}-wasm32-unknown-wasip1-threads$$" | head -n 1); \
		fi; \
		if [ -z "$$WASM_SDK_ENV" ]; then \
			WASM_SDK_ENV=$$(printf "%s\n" "$$SDK_LIST" | grep -E 'wasm32-unknown-wasip1-threads$$' | tail -n 1); \
		fi; \
	fi; \
	if [ -z "$$WASM_SDK_ENV" ]; then \
		echo "No WASM SDK found. Run 'swift sdk list' with a Swift.org toolchain or set WASM_SDK=..."; \
		exit 1; \
	fi; \
	echo "Using toolchain: $$TOOLCHAIN_DIR"; \
	echo "Using WASM SDK: $$WASM_SDK_ENV"; \
	cd Packages/SantoriniWasm && \
	$$SWIFT_CMD package --swift-sdk $$WASM_SDK_ENV \
		--allow-writing-to-directory ../../$(WASM_OUTPUT) \
		js --product SantoriniWasm --output ../../$(WASM_OUTPUT)

onnx:
	@if [ -z "$(CHECKPOINT)" ]; then \
		echo "Set CHECKPOINT to a .safetensors checkpoint path."; \
		exit 1; \
	fi
	$(PYTHON) tools/export_onnx.py --checkpoint $(CHECKPOINT) --output $(MODEL_OUT)

onnx-verify:
	node tools/verify_onnx.js $(MODEL_OUT)

web-install:
	cd web && $(NPM) install

web-build:
	cd web && $(NPM) run build

web-dev:
	cd web && $(NPM) run dev

clean:
	rm -rf web/dist web/.vite
