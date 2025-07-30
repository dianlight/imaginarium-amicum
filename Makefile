# Makefile for Go AI Chat Application

# Project and Binary Variables
APP_NAME := imaginarium-amicum
GO_BIN := main
WEB_DIR := web
BUILD_DIR := build

# Go build flags for setting gpuLayersStr in main.go
LD_FLAGS_CPU := -ldflags="-X main.gpuLayersStr=0"
LD_FLAGS_GPU := -ldflags="-X main.gpuLayersStr=-1" # -1 often means all layers

# --- Common Helper Targets ---

.PHONY: ensure-go-modules
ensure-go-modules:
	@echo "Ensuring Go modules are tidy and dependencies downloaded..."
	@go mod tidy
	@go mod download # This fetches GoStableDiffusion and its CGO dependencies

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

.PHONY: ensure-stablediffusion-binding
ensure-stablediffusion-binding:
	@echo "Ensuring GoStableDiffusion binding is available and built..."
	@if [ ! -d "binding/gostablediffusion" ]; then \
		echo "Cloning GoStableDiffusion..."; \
		git clone https://github.com/Binozo/GoStableDiffusion.git binding/gostablediffusion; \
	else \
		echo "GoStableDiffusion already cloned."; \
	fi
	# stable-diffusion library will be built with platform-specific flags later.

# --- Build Targets per OS/Architecture ---

.PHONY: build all
build: build-macos-apple # Default target builds for current platform (macOS Apple Silicon)
all: build-linux-cpu # Default target builds for Linux CPU

# Create build directory if it doesn't exist
$(BUILD_DIR):
	@mkdir -p $(BUILD_DIR)

# Linux (AMD64 CPU)
.PHONY: build-linux-cpu run-linux-cpu
build-linux-cpu: ensure-llama-binding ensure-stablediffusion-binding ensure-go-modules $(BUILD_DIR)
	@echo "Building for Linux (CPU)..."
	@cd binding/go-llama.cpp && make libbinding.a BUILD_TYPE=cpu CGO_LDFLAGS=""
	@echo "Building GoStableDiffusion for Linux (CPU)..."
	@cd binding/gostablediffusion && go generate
	GOOS=linux GOARCH=amd64 CGO_ENABLED=1 \
	go build $(LD_FLAGS_CPU) -o $(BUILD_DIR)/$(GO_BIN)_linux_amd64 .
run-linux-cpu: build-linux-cpu
	@echo "Running Linux (CPU) build..."
	./$(BUILD_DIR)/$(GO_BIN)_linux_amd64

# Linux (NVIDIA GPU) - Assumes CUDA toolkit is installed on the build machine
.PHONY: build-linux-nvidia run-linux-nvidia
build-linux-nvidia: ensure-llama-binding ensure-stablediffusion-binding ensure-go-modules $(BUILD_DIR)
	@echo "Building for Linux (NVIDIA GPU)..."
	@cd binding/go-llama.cpp && make libbinding.a BUILD_TYPE=cublas CGO_LDFLAGS="-lcublas -lcudart -L/usr/local/cuda/lib64"
	@echo "Building GoStableDiffusion for Linux (NVIDIA GPU)..."
	@cd binding/gostablediffusion && CUDA=1 go generate
	GOOS=linux GOARCH=amd64 CGO_ENABLED=1 \
	CGO_LDFLAGS="-L/usr/local/cuda/lib64 -lcublas -lcudart" \
	go build $(LD_FLAGS_GPU) -o $(BUILD_DIR)/$(GO_BIN)_linux_nvidia .
run-linux-nvidia: build-linux-nvidia
	@echo "Running Linux (NVIDIA GPU) build (requires CUDA drivers/toolkit on host)..."
	LD_LIBRARY_PATH=/usr/local/cuda/lib64 ./$(BUILD_DIR)/$(GO_BIN)_linux_nvidia

# Linux (AMD GPU - ROCm) - Assumes ROCm toolchain is installed on the build machine
.PHONY: build-linux-ati run-linux-ati
build-linux-ati: ensure-llama-binding ensure-stablediffusion-binding ensure-go-modules $(BUILD_DIR)
	@echo "Building for Linux (AMD GPU - ROCm)..."
	@cd binding/go-llama.cpp && CC=/opt/rocm/llvm/bin/clang CXX=/opt/rocm/llvm/bin/clang++ make libbinding.a BUILD_TYPE=hipblas CGO_LDFLAGS="-O3 --hip-link --rtlib=compiler-rt -lrocblas -lhipblas -L/opt/rocm/lib"
	@echo "Building GoStableDiffusion for Linux (AMD GPU - ROCm)..."
	@cd binding/gostablediffusion && VULKAN=1 go generate
	GOOS=linux GOARCH=amd64 CGO_ENABLED=1 \
	CC=/opt/rocm/llvm/bin/clang CXX=/opt/rocm/llvm/bin/clang++ \
	CGO_LDFLAGS="-L/opt/rocm/lib -lrocblas -lhipblas -O3 --hip-link --rtlib=compiler-rt" \
	go build $(LD_FLAGS_GPU) -o $(BUILD_DIR)/$(GO_BIN)_linux_ati .
run-linux-ati: build-linux-ati
	@echo "Running Linux (AMD GPU - ROCm) build (requires ROCm drivers on host)..."
	LD_LIBRARY_PATH=/opt/rocm/lib ./$(BUILD_DIR)/$(GO_BIN)_linux_ati

# macOS (Apple Silicon - Metal GPU)
.PHONY: build-macos-apple run-macos-apple
build-macos-apple: ensure-llama-binding ensure-stablediffusion-binding ensure-go-modules $(BUILD_DIR)
	@echo "Building for macOS (Apple Silicon - Metal GPU)..."
	# Build go-llama.cpp's internal libbinding.a with Metal flags
	@cd binding/go-llama.cpp && make libbinding.a BUILD_TYPE=metal CGO_LDFLAGS="-framework Foundation -framework Metal -framework MetalKit -framework MetalPerformanceShaders"
	@echo "Building GoStableDiffusion for macOS (Apple Silicon - Metal GPU)..."
	@cd binding/gostablediffusion && METAL=1 go generate
	GOOS=darwin GOARCH=arm64 CGO_ENABLED=1 \
	CGO_LDFLAGS="-framework Foundation -framework Metal -framework MetalKit -framework MetalPerformanceShaders" \
	go build $(LD_FLAGS_GPU) -o $(BUILD_DIR)/$(GO_BIN)_macos_arm64 .
run-macos-apple: build-macos-apple
	@echo "Running macOS (Apple Silicon - Metal GPU) build..."
	./$(BUILD_DIR)/$(GO_BIN)_macos_arm64

# Windows (AMD64 CPU)
.PHONY: build-windows-cpu run-windows-cpu
build-windows-cpu: ensure-llama-binding ensure-stablediffusion-binding ensure-go-modules $(BUILD_DIR)
	@echo "Building for Windows (CPU)..."
	@cd binding/go-llama.cpp && make libbinding.a BUILD_TYPE=cpu CGO_LDFLAGS=""
	@echo "Building GoStableDiffusion for Windows (CPU)..."
	@cd binding/gostablediffusion && go generate
	GOOS=windows GOARCH=amd64 CGO_ENABLED=1 \
	go build $(LD_FLAGS_CPU) -o $(BUILD_DIR)/$(GO_BIN)_windows_amd64.exe .
run-windows-cpu: build-windows-cpu
	@echo "Running Windows (CPU) build..."
	.\\$(BUILD_DIR)/$(GO_BIN)_windows_amd64.exe

# Windows (NVIDIA GPU) - Experimental and requires specific setup on Windows
.PHONY: build-windows-nvidia run-windows-nvidia
build-windows-nvidia: ensure-llama-binding ensure-stablediffusion-binding ensure-go-modules $(BUILD_DIR)
	@echo "Building for Windows (NVIDIA GPU) - EXPERIMENTAL. Requires MSYS2/MinGW and CUDA SDK."
	@echo "You might need to adjust CGO_LDFLAGS to point to your CUDA installation (e.g., -L\"C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/vX.Y/lib/x64\")"
	@echo "Also ensure go-llama.cpp's Makefile 'make libbinding.a' works with your Windows compiler."
	@cd binding/go-llama.cpp && make libbinding.a BUILD_TYPE=cublas CGO_LDFLAGS="-lcublas -lcudart -L/path/to/cuda/lib/x64" # Placeholder: Update to your CUDA lib path
	@echo "Building GoStableDiffusion for Windows (NVIDIA GPU)..."
	@cd binding/gostablediffusion && CUDA=1 go generate
	GOOS=windows GOARCH=amd64 CGO_ENABLED=1 \
	CGO_LDFLAGS="-L\"C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/vX.Y/lib/x64\" -lcublas -lcudart" \
	go build $(LD_FLAGS_GPU) -o $(BUILD_DIR)/$(GO_BIN)_windows_nvidia.exe .
run-windows-nvidia: build-windows-nvidia
	@echo "Running Windows (NVIDIA GPU) build (requires CUDA drivers/toolkit on host).."
	@echo "Ensure CUDA libraries are in your PATH or copy them next to the executable."
	.\\$(BUILD_DIR)/$(GO_BIN)_windows_nvidia.exe

# Windows (AMD GPU - ROCm) - Highly experimental for native Windows, typically via WSL2
.PHONY: build-windows-ati run-windows-ati
build-windows-ati: ensure-llama-binding ensure-stablediffusion-binding ensure-go-modules $(BUILD_DIR)
	@echo "Building for Windows (AMD GPU - ROCm) - HIGHLY EXPERIMENTAL. Best in WSL2."
	@echo "Requires MSYS2/MinGW and ROCm SDK. You will need to set compiler (CC/CXX) paths to ROCm-enabled clang."
	@echo "Adjust CGO_LDFLAGS for ROCm libraries and ensure go-llama.cpp's Makefile works with your Windows compiler."
	@cd binding/go-llama.cpp && CC=/opt/rocm/llvm/bin/clang CXX=/opt/rocm/llvm/bin/clang++ make libbinding.a BUILD_TYPE=hipblas CGO_LDFLAGS="-O3 --hip-link --rtlib=compiler-rt -lrocblas -lhipblas -L/path/to/rocm/lib" # Placeholder: Update to your ROCm lib path
	@echo "Building GoStableDiffusion for Windows (AMD GPU - ROCm)..."
	@cd binding/gostablediffusion && VULKAN=1 go generate
	GOOS=windows GOARCH=amd64 CGO_ENABLED=1 \
	CC=/path/to/rocm/llvm/bin/clang CXX=/path/to/rocm/llvm/bin/clang++ \
	CGO_LDFLAGS="-L/path/to/rocm/lib -lrocblas -lhipblas -O3 --hip-link --rtlib=compiler-rt" \
	go build $(LD_FLAGS_GPU) -o $(BUILD_DIR)/$(GO_BIN)_windows_ati.exe .
run-windows-ati: build-windows-ati
	@echo "Running Windows (AMD GPU - ROCm) build (requires ROCm drivers on host)."
	@echo "Ensure ROCm libraries are in your PATH or copy them next to the executable."
	.\\$(BUILD_DIR)/$(GO_BIN)_windows_ati.exe


# --- Clean Target ---

.PHONY: clean
clean:
	@echo "Cleaning up..."
	@rm -rf $(BUILD_DIR) # Remove build directory and all its contents
	@rm -rf binding/go-llama.cpp # Clean go-llama.cpp binding directory
	@rm -rf binding/gostablediffusion # Clean GoStableDiffusion binding directory
	@echo "Cleanup complete."

