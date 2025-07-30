---
description: Repository Information Overview
alwaysApply: true
---

# Imaginarium Amicum Information

## Summary
An AI chatbot working fully local to be an imaginary friend. The application uses LLaMA language models and Stable Diffusion for image generation, providing a conversational AI experience that runs entirely on the user's machine without requiring cloud services.

## Structure
- **binding/**: Contains Go bindings for LLaMA.cpp
- **build/**: Compiled binaries for different platforms
- **web/**: Frontend web interface files
- **.github/**: GitHub workflows and configuration
- **.zencoder/**: Project rules and configuration

## Language & Runtime
**Language**: Go
**Version**: Go 1.24.5
**Build System**: Make
**Package Manager**: Go Modules

## Dependencies
**Main Dependencies**:
- github.com/go-skynet/go-llama.cpp - LLaMA language model binding
- github.com/gorilla/websocket v1.5.3 - WebSocket implementation
- github.com/seasonjs/stable-diffusion v0.2.0 - Stable Diffusion binding

**Development Dependencies**:
- github.com/ebitengine/purego - Go bindings for C libraries
- golang.org/x/sys - System-level Go packages

## Build & Installation
```bash
# CPU-only build
make build

# For Apple Silicon with Metal GPU acceleration
make build-macos-apple

# For NVIDIA GPU acceleration
make build-linux-nvidia

# For AMD GPU with ROCm
make build-linux-ati
```

## Main Files & Resources
**Entry Point**: main.go
**Web Interface**: web/index.html
**Model Bindings**: 
- binding/go-llama.cpp/ - LLaMA binding
- github.com/seasonjs/stable-diffusion - Stable Diffusion binding

**Model Sources**:
- LLaMA model: llama-2-7b-chat.Q4_K_M.gguf from Hugging Face
- Stable Diffusion model: v1-5-pruned-emaonly.safetensors from Hugging Face

## Docker Configuration
Docker support is implemented through Makefile targets:
```bash
# Build CPU Docker image
make docker-build-cpu

# Build NVIDIA GPU Docker image
make docker-build-nvidia

# Build AMD GPU Docker image
make docker-build-ati

# Run containers
make run-cpu
make run-nvidia
make run-ati
```

## Project Features
- Multi-platform support (Linux, macOS, Windows)
- GPU acceleration options (NVIDIA CUDA, AMD ROCm, Apple Metal)
- WebSocket-based chat interface
- Embedded web server and frontend
- Context-aware conversation with history management
- Image generation based on conversation context
- Fully local operation without cloud dependencies