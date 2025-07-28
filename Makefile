# Makefile for Go AI Chat Application

# Project and Docker Variables
APP_NAME := go-ai-chat
GO_BIN := main
WEB_DIR := web
MODELS_DIR := models

# Go build flags for setting gpuLayersStr in main.go
LD_FLAGS_CPU := -ldflags="-X main.gpuLayersStr=0"
LD_FLAGS_NVIDIA := -ldflags="-X main.gpuLayersStr=-1" # -1 often means all layers
LD_FLAGS_ATI := -ldflags="-X main.gpuLayersStr=-1"    # -1 often means all layers
LD_FLAGS_APPLE := -ldflags="-X main.gpuLayersStr=-1"  # -1 for Metal offloading

# --- Build Targets ---

.PHONY: all build clean docker-build-cpu docker-build-nvidia docker-build-ati run-cpu run-nvidia run-ati pull-models build-apple run-apple build-bindings-for-docker

all: build docker-build-cpu

# Build the Go application locally (CPU-only)
build:
	@echo "Building Go application locally (CPU-only)..."
	@go mod tidy
	@CGO_ENABLED=1 go build $(LD_FLAGS_CPU) -o $(GO_BIN) .
	@echo "Local build complete: ./"$(GO_BIN)

# Build the Go application locally for Apple Silicon (Metal acceleration)
build-apple: pull-models
	@echo "Building Go application locally for Apple Silicon (Metal acceleration)..."
	@go mod tidy
	@echo "Building go-llama.cpp with Metal support..."
	@if [ ! -d "binding/go-llama.cpp" ]; then git clone --recurse-submodules https://github.com/go-skynet/go-llama.cpp.git binding/go-llama.cpp; fi
	@cd binding/go-llama.cpp && BUILD_TYPE=metal make libbinding.a CGO_LDFLAGS="-framework Foundation -framework Metal -framework MetalKit -framework MetalPerformanceShaders"
	@echo "Building go-sd.cpp with Metal support..."
	@if [ ! -d "binding/go-sd.cpp" ]; then git clone --recurse-submodules https://github.com/go-skynet/go-sd.cpp.git binding/go-sd.cpp; fi
	@cd binding/go-sd.cpp && BUILD_TYPE=metal make libbinding.a CGO_LDFLAGS="-framework Foundation -framework Metal -framework MetalKit -framework MetalPerformanceShaders"
	@echo "Building main Go application with Metal bindings..."
	@CGO_LDFLAGS="-L$(PWD)/binding/go-llama.cpp -L$(PWD)/binding/go-sd.cpp -framework Foundation -framework Metal -framework MetalKit -framework MetalPerformanceShaders" \
	C_INCLUDE_PATH="$(PWD)/binding/go-llama.cpp:$(PWD)/binding/go-sd.cpp" \
	CGO_ENABLED=1 go build $(LD_FLAGS_APPLE) -o $(GO_BIN) .
	@echo "Apple Silicon local build complete: ./"$(GO_BIN)

# Pulls required AI models
pull-models:
	@echo "Checking and pulling AI models..."
	@mkdir -p $(MODELS_DIR)
	@if [ ! -f "$(MODELS_DIR)/llama-2-7b-chat.Q4_K_M.gguf" ]; then \
		echo "Downloading Llama 2 7B Chat GGUF model..."; \
		wget -O "$(MODELS_DIR)/llama-2-7b-chat.Q4_K_M.gguf" https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/resolve/main/llama-2-7b-chat.Q4_K_M.gguf; \
	else \
		echo "Llama 2 model already exists."; \
	fi
	@if [ ! -f "$(MODELS_DIR)/v1-5-pruned-emaonly.safetensors" ]; then \
		echo "Downloading Stable Diffusion v1.5 model..."; \
		wget -O "$(MODELS_DIR)/v1-5-pruned-emaonly.safetensors" https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/v1-5-pruned-emaonly.safetensors; \
	else \
		echo "Stable Diffusion model already exists."; \
	fi
	@echo "Model download/check complete."

# Target to build bindings specifically for Docker stages
# This target should be invoked from within a Dockerfile RUN command.
# It takes BUILD_TYPE and CGO_LDFLAGS as environment variables from the Dockerfile context.
# CC and CXX should be set in the Dockerfile's RUN command if non-default compilers are needed.
build-bindings-for-docker:
	@echo "Building go-llama.cpp bindings with BUILD_TYPE=$(BUILD_TYPE)..."
	@if [ ! -d "binding/go-llama.cpp" ]; then git clone --recurse-submodules https://github.com/go-skynet/go-llama.cpp.git binding/go-llama.cpp; fi
	@cd binding/go-llama.cpp && BUILD_TYPE="$(BUILD_TYPE)" make libbinding.a CGO_LDFLAGS="$(CGO_LDFLAGS)"
	@echo "Building go-sd.cpp bindings with BUILD_TYPE=$(BUILD_TYPE)..."
	@if [ ! -d "binding/go-sd.cpp" ]; then git clone --recurse-submodules https://github.com/go-skynet/go-sd.cpp.git binding/go-sd.cpp; fi
	@cd binding/go-sd.cpp && BUILD_TYPE="$(BUILD_TYPE)" make libbinding.a CGO_LDFLAGS="$(CGO_LDFLAGS)"

# --- Docker Build Targets ---

# Build Docker image for CPU (AMD64 / ARM64 compatible)
docker-build-cpu: pull-models
	@echo "Building Docker image for CPU (AMD64 / ARM64 compatible)..."
	@docker build --build-arg GPU_LAYERS=0 -f Dockerfile.cpu -t $(APP_NAME):cpu .
	@echo "Docker image $(APP_NAME):cpu built successfully."

# Build Docker image for Nvidia GPU (requires CUDA toolkit on host for building, NVIDIA Container Toolkit for running)
# This will use an Nvidia CUDA base image.
docker-build-nvidia: pull-models
	@echo "Building Docker image for Nvidia GPU..."
	@docker build --build-arg GPU_LAYERS=-1 -f Dockerfile.nvidia -t $(APP_NAME):nvidia .
	@echo "Docker image $(APP_NAME):nvidia built successfully."

# Build Docker image for AMD GPU (requires ROCm on host for building, AMD ROCm support for Docker for running)
# This will use an AMD ROCm base image.
docker-build-ati: pull-models
	@echo "Building Docker image for AMD GPU (ROCm)..."
	@docker build --build-arg GPU_LAYERS=-1 -f Dockerfile.ati -t $(APP_NAME):ati .
	@echo "Docker image $(APP_NAME):ati built successfully."

# --- Docker Run Targets ---

# Run the CPU Docker container
run-cpu: docker-build-cpu
	@echo "Running CPU Docker container on port 8080..."
	@docker run --rm -p 8080:8080 $(APP_NAME):cpu

# Run the Nvidia GPU Docker container
# Requires NVIDIA Container Toolkit installed on your Docker host.
run-nvidia: docker-build-nvidia
	@echo "Running Nvidia GPU Docker container on port 8080..."
	@docker run --rm --gpus all -p 8080:8080 $(APP_NAME):nvidia

# Run the AMD GPU Docker container
# Requires ROCm enabled Docker runtime on your host.
# The `--device` flags provide access to the necessary ROCm devices.
run-ati: docker-build-ati
	@echo "Running AMD GPU Docker container on port 8080 (ROCm)..."
	@docker run --rm --device=/dev/kfd --device=/dev/dri -p 8080:8080 $(APP_NAME):ati

# Run the locally built Apple Silicon optimized application
run-apple: build-apple
	@echo "Running Apple Silicon optimized application locally on port 8080..."
	@./$(GO_BIN)

# --- Clean Target ---

clean:
	@echo "Cleaning up..."
	@rm -f $(GO_BIN)
	@rm -rf $(MODELS_DIR) # Also remove downloaded models
	@rm -rf binding/go-llama.cpp binding/go-sd.cpp # Remove cloned binding repos
	@docker rmi $(APP_NAME):cpu $(APP_NAME):nvidia $(APP_NAME):ati 2>/dev/null || true
	@echo "Cleanup complete."

