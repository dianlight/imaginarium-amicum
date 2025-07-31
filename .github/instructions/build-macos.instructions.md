# Building and Running Instructions

## macOS (Apple Silicon)

To build and run the project on macOS with Apple Silicon, use these Make targets:

```bash
# Build only
make build-macos-apple

# Run (will build first if needed)
make run-macos-apple
```

This configuration uses Metal GPU acceleration for:
- LLM inference (via llama.cpp Metal backend)
- Stable Diffusion image generation (via Metal compute shaders)

The build process will:
- Configure Metal frameworks
- Build native arm64 binaries
- Enable GPU acceleration layers
- Set up proper DYLD paths for libraries

💡 **Tip**: You can use just `make run-macos-apple` since it includes the build step automatically.