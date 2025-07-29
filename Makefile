# Makefile for Go AI Chat Application

# Project and Binary Variables
APP_NAME := imaginarium-amicum
GO_BIN := main
WEB_DIR := web

# Go build flags for setting gpuLayersStr in main.go
LD_FLAGS_CPU := -ldflags="-X main.gpuLayersStr=0"
LD_FLAGS_GPU := -ldflags="-X main.gpuLayersStr=-1" # -1 often means all layers

# --- Common Helper Targets ---

.PHONY: ensure-go-modules
ensure-go-modules:
	@echo "Ensuring Go modules are tidy and dependencies downloaded..."
	@go mod tidy
	@go mod download # This fetches seasonjs/stable-diffusion and its CGO dependencies

.PHONY: ensure-llama-binding
ensure-llama-binding:
	@echo "Ensuring go-llama.cpp binding is available and built..."
	@if [ ! -d "binding/go-llama.cpp" ]; then \
		echo "Cloning go-llama.cpp..."; \
		git clone --recurse-submodules https://github.com/go-skynet/go-llama.cpp.git binding/go-llama.cpp; \
	else \
		echo "go-llama.cpp already cloned."; \
	fi
	# libbinding.a will be built with platform-specific flags later.

# --- Build Targets per OS/Architecture ---

.PHONY: build all
all: build-linux-cpu # Default target builds for Linux CPU

# Linux (AMD64 CPU)
.PHONY: build-linux-cpu run-linux-cpu
build-linux-cpu: ensure-go-modules ensure-llama-binding
	@echo "Building for Linux (CPU)..."
	@cd binding/go-llama.cpp && make libbinding.a BUILD_TYPE=cpu CGO_LDFLAGS=""
	GOOS=linux GOARCH=amd64 CGO_ENABLED=1 \
	go build $(LD_FLAGS_CPU) -o $(GO_BIN)_linux_amd64 .
run-linux-cpu: build-linux-cpu
	@echo "Running Linux (CPU) build..."
	./$(GO_BIN)_linux_amd64

# Linux (NVIDIA GPU) - Assumes CUDA toolkit is installed on the build machine
.PHONY: build-linux-nvidia run-linux-nvidia
build-linux-nvidia: ensure-go-modules ensure-llama-binding
	@echo "Building for Linux (NVIDIA GPU)..."
	@cd binding/go-llama.cpp && make libbinding.a BUILD_TYPE=cublas CGO_LDFLAGS="-lcublas -lcudart -L/usr/local/cuda/lib64"
	GOOS=linux GOARCH=amd64 CGO_ENABLED=1 \
	CGO_LDFLAGS="-L/usr/local/cuda/lib64 -lcublas -lcudart" \
	go build $(LD_FLAGS_GPU) -o $(GO_BIN)_linux_nvidia .
run-linux-nvidia: build-linux-nvidia
	@echo "Running Linux (NVIDIA GPU) build (requires CUDA drivers/toolkit on host)..."
	LD_LIBRARY_PATH=/usr/local/cuda/lib64 ./$(GO_BIN)_linux_nvidia

# Linux (AMD GPU - ROCm) - Assumes ROCm toolchain is installed on the build machine
.PHONY: build-linux-ati run-linux-ati
build-linux-ati: ensure-go-modules ensure-llama-binding
	@echo "Building for Linux (AMD GPU - ROCm)..."
	@cd binding/go-llama.cpp && CC=/opt/rocm/llvm/bin/clang CXX=/opt/rocm/llvm/bin/clang++ make libbinding.a BUILD_TYPE=hipblas CGO_LDFLAGS="-O3 --hip-link --rtlib=compiler-rt -lrocblas -lhipblas -L/opt/rocm/lib"
	GOOS=linux GOARCH=amd64 CGO_ENABLED=1 \
	CC=/opt/rocm/llvm/bin/clang CXX=/opt/rocm/llvm/bin/clang++ \
	CGO_LDFLAGS="-L/opt/rocm/lib -lrocblas -lhipblas -O3 --hip-link --rtlib=compiler-rt" \
	go build $(LD_FLAGS_GPU) -o $(GO_BIN)_linux_ati .
run-linux-ati: build-linux-ati
	@echo "Running Linux (AMD GPU - ROCm) build (requires ROCm drivers on host)..."
	LD_LIBRARY_PATH=/opt/rocm/lib ./$(GO_BIN)_linux_ati

# macOS (Apple Silicon - Metal GPU)
.PHONY: build-macos-apple run-macos-apple
build-macos-apple: ensure-go-modules ensure-llama-binding
	@echo "Building for macOS (Apple Silicon - Metal GPU)..."
	# Build go-llama.cpp's internal libbinding.a with Metal flags
	@cd binding/go-llama.cpp && make libbinding.a BUILD_TYPE=metal CGO_LDFLAGS="-framework Foundation -framework Metal -framework MetalKit -framework MetalPerformanceShaders"
	GOOS=darwin GOARCH=arm64 CGO_ENABLED=1 \
	CGO_LDFLAGS="-framework Foundation -framework Metal -framework MetalKit -framework MetalPerformanceShaders" \
	go build $(LD_FLAGS_GPU) -o $(GO_BIN)_macos_arm64 .
run-macos-apple: build-macos-apple
	@echo "Running macOS (Apple Silicon - Metal GPU) build..."
	./$(GO_BIN)_macos_arm64

# Windows (AMD64 CPU)
.PHONY: build-windows-cpu run-windows-cpu
build-windows-cpu: ensure-go-modules ensure-llama-binding
	@echo "Building for Windows (CPU)..."
	@cd binding/go-llama.cpp && make libbinding.a BUILD_TYPE=cpu CGO_LDFLAGS=""
	GOOS=windows GOARCH=amd64 CGO_ENABLED=1 \
	go build $(LD_FLAGS_CPU) -o $(GO_BIN)_windows_amd64.exe .
run-windows-cpu: build-windows-cpu
	@echo "Running Windows (CPU) build..."
	.\\$(GO_BIN)_windows_amd64.exe

# Windows (NVIDIA GPU) - Experimental and requires specific setup on Windows
.PHONY: build-windows-nvidia run-windows-nvidia
build-windows-nvidia: ensure-go-modules ensure-llama-binding
	@echo "Building for Windows (NVIDIA GPU) - EXPERIMENTAL. Requires MSYS2/MinGW and CUDA SDK."
	@echo "You might need to adjust CGO_LDFLAGS to point to your CUDA installation (e.g., -L\"C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/vX.Y/lib/x64\")"
	@echo "Also ensure go-llama.cpp's Makefile 'make libbinding.a' works with your Windows compiler."
	@cd binding/go-llama.cpp && make libbinding.a BUILD_TYPE=cublas CGO_LDFLAGS="-lcublas -lcudart -L/path/to/cuda/lib/x64" # Placeholder: Update to your CUDA lib path
	GOOS=windows GOARCH=amd64 CGO_ENABLED=1 \
	CGO_LDFLAGS="-L\"C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/vX.Y/lib/x64\" -lcublas -lcudart" \
	go build $(LD_FLAGS_GPU) -o $(GO_BIN)_windows_nvidia.exe .
run-windows-nvidia: build-windows-nvidia
	@echo "Running Windows (NVIDIA GPU) build (requires CUDA drivers/toolkit on host).."
	@echo "Ensure CUDA libraries are in your PATH or copy them next to the executable."
	.\\$(GO_BIN)_windows_nvidia.exe

# Windows (AMD GPU - ROCm) - Highly experimental for native Windows, typically via WSL2
.PHONY: build-windows-ati run-windows-ati
build-windows-ati: ensure-go-modules ensure-llama-binding
	@echo "Building for Windows (AMD GPU - ROCm) - HIGHLY EXPERIMENTAL. Best in WSL2."
	@echo "Requires MSYS2/MinGW and ROCm SDK. You will need to set compiler (CC/CXX) paths to ROCm-enabled clang."
	@echo "Adjust CGO_LDFLAGS for ROCm libraries and ensure go-llama.cpp's Makefile works with your Windows compiler."
	@cd binding/go-llama.cpp && CC=/opt/rocm/llvm/bin/clang CXX=/opt/rocm/llvm/bin/clang++ make libbinding.a BUILD_TYPE=hipblas CGO_LDFLAGS="-O3 --hip-link --rtlib=compiler-rt -lrocblas -lhipblas -L/path/to/rocm/lib" # Placeholder: Update to your ROCm lib path
	GOOS=windows GOARCH=amd64 CGO_ENABLED=1 \
	CC=/path/to/rocm/llvm/bin/clang CXX=/path/to/rocm/llvm/bin/clang++ \
	CGO_LDFLAGS="-L/path/to/rocm/lib -lrocblas -lhipblas -O3 --hip-link --rtlib=compiler-rt" \
	go build $(LD_FLAGS_GPU) -o $(GO_BIN)_windows_ati.exe .
run-windows-ati: build-windows-ati
	@echo "Running Windows (AMD GPU - ROCm) build (requires ROCm drivers on host)."
	@echo "Ensure ROCm libraries are in your PATH or copy them next to the executable."
	.\\$(GO_BIN)_windows_ati.exe


# --- Clean Target ---

.PHONY: clean
clean:
	@echo "Cleaning up..."
	@rm -f $(GO_BIN) \
		$(GO_BIN)_linux_amd64 \
		$(GO_BIN)_linux_nvidia \
		$(GO_BIN)_linux_ati \
		$(GO_BIN)_macos_arm64 \
		$(GO_BIN)_windows_amd64.exe \
		$(GO_BIN)_windows_nvidia.exe \
		$(GO_BIN)_windows_ati.exe
	@rm -rf binding/go-llama.cpp # Clean go-llama.cpp binding directory
	@echo "Cleanup complete."

