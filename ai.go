package main

import (
	"log"
	"strconv"
	"unsafe"

	"github.com/binozo/gostablediffusion/pkg/sd"
	"github.com/go-skynet/go-llama.cpp"
)

// AIModels holds the initialized AI models
type AIModels struct {
	LLM   *llama.LLama
	SDCtx *sd.Context
}

// InitializeAIModels initializes both LLM and Stable Diffusion models
func InitializeAIModels() (*AIModels, error) {
	// Parse gpuLayersStr set at build time
	var gpuLayers int
	if gpuLayersStr != "" {
		val, err := strconv.Atoi(gpuLayersStr)
		if err != nil {
			log.Printf("Warning: Could not parse GPULayers from build flag '%s'. Defaulting to CPU (0 layers). Error: %v", gpuLayersStr, err)
			gpuLayers = 0 // Default to CPU if parsing fails
		} else {
			gpuLayers = val
		}
	} else {
		gpuLayers = 0 // Default to CPU if no build flag is set
	}
	log.Printf("Initializing models with GPU layers: %d", gpuLayers)

	// Initialize LLM
	llm, err := initializeLLM(gpuLayers)
	if err != nil {
		return nil, err
	}

	// Initialize Stable Diffusion
	sdCtx, err := initializeStableDiffusion(gpuLayers)
	if err != nil {
		llm.Free()
		return nil, err
	}

	return &AIModels{
		LLM:   llm,
		SDCtx: sdCtx,
	}, nil
}

// initializeLLM initializes the LLaMA language model
func initializeLLM(gpuLayers int) (*llama.LLama, error) {
	// Try to initialize with Metal first, if it fails, fall back to CPU
	log.Printf("Attempting to initialize Llama LLM with Metal GPU acceleration (layers: %d)", gpuLayers)
	llm, err := llama.New(llamaModelPath, llama.SetContext(maxTokens), llama.SetGPULayers(gpuLayers))
	if err != nil {
		log.Printf("Warning: Metal GPU initialization failed: %v", err)
		log.Printf("Falling back to CPU-only mode for Llama LLM")

		// Fall back to CPU-only mode (0 GPU layers)
		llm, err = llama.New(llamaModelPath, llama.SetContext(maxTokens), llama.SetGPULayers(0))
		if err != nil {
			return nil, err
		}
		log.Printf("Successfully initialized Llama LLM in CPU-only mode")
	} else {
		log.Printf("Successfully initialized Llama LLM with Metal GPU acceleration")
	}

	return llm, nil
}

// initializeStableDiffusion initializes the Stable Diffusion model
func initializeStableDiffusion(gpuLayers int) (*sd.Context, error) {
	log.Printf("Initializing Stable Diffusion with GoStableDiffusion")

	// Set up logging callback
	sd.SetLogCallback(func(level sd.LogLevel, text string, data unsafe.Pointer) {
		switch level {
		case sd.LogDebug:
			log.Printf("SD DEBUG: %s", text)
		case sd.LogInfo:
			log.Printf("SD INFO: %s", text)
		case sd.LogWarn:
			log.Printf("SD WARN: %s", text)
		case sd.LogError:
			log.Printf("SD ERROR: %s", text)
		default:
			log.Printf("SD: %s", text)
		}
	})

	// Initialize SD context with model
	sdBuilder := sd.New().SetModel(sdModelPath)

	// Try with GPU acceleration if requested
	if gpuLayers != 0 {
		log.Printf("Attempting to initialize Stable Diffusion with GPU acceleration")
		// Note: GoStableDiffusion uses different GPU acceleration methods
		// We'll use Flash Attention for better performance if available
		sdBuilder = sdBuilder.UseFlashAttn()
	}

	sdCtx, err := sdBuilder.Load()
	if err != nil {
		return nil, err
	}

	log.Printf("Successfully loaded Stable Diffusion model")
	return sdCtx, nil
}

// FreeAIModels properly frees the AI models
func (models *AIModels) Free() {
	if models.LLM != nil {
		models.LLM.Free()
	}
	if models.SDCtx != nil {
		models.SDCtx.Free()
	}
}
