# imaginarium-amicum
An AI chatbot working full local to be an imaginary friend

# Development Stage
Very very early stage!

# USE

How to Use the Makefile:

Local CPU Build:

```Bash
make build
```

This will compile the Go application using CPU inference for go-llama.cpp and go-sd.cpp and create an executable ./main in your project root.

Build CPU Docker Image:

```Bash
make docker-build-cpu
```

This will create a Docker image tagged go-ai-chat:cpu that is optimized for CPU execution and should run on AMD64 and ARM64 (e.g., Apple Silicon) machines.

Build Nvidia GPU Docker Image:

```Bash
make docker-build-nvidia
```

Prerequisites: You need an Nvidia GPU and a Docker setup with NVIDIA Container Toolkit installed on your host system. This command will build an image tagged go-ai-chat:nvidia that utilizes your Nvidia GPU.


Build the AMD GPU Docker Image:

```Bash
make docker-build-ati
```
This will build the Docker image optimized for AMD GPUs using ROCm.

Run the AMD GPU Docker Container:

```Bash
make run-ati
```
This command includes the necessary --device=/dev/kfd --device=/dev/dri flags to provide the container access to your AMD GPU.

Access the application at http://localhost:8080.

Run Nvidia GPU Docker Container:

```Bash
make run-nvidia
```
Prerequisites: Requires the NVIDIA Container Toolkit. This command will run the container with GPU access. Access the application at http://localhost:8080.

Run CPU Docker Container:

```Bash
make run-cpu
```

Clean Project Files and Docker Images:

```Bash
make clean
```

This removes the locally built executable and the Docker images created by the Makefile.

Important Considerations:

__Model Downloads__: The pull-models target will download the large AI models (several GBs) to your host's models directory. These are then copied into the Docker images during the build process.

__Performance__: CPU inference for large models like Llama 2 and Stable Diffusion is very slow. Nvidia GPU acceleration will significantly improve performance if you have the compatible hardware and software setup.

__Apple Silicon (M1/M2/M3)__: While a dedicated "Metal" Docker build target isn't practical within a Linux Docker context, the docker-build-cpu image will run efficiently on your Apple Silicon Mac thanks to Docker's Rosetta emulation for x86_64 images, or natively if you build for linux/arm64. For true Metal acceleration, you would typically compile llama.cpp and stable-diffusion.cpp natively on macOS and run the Go application directly without Docker.
