package main

import (
	"embed"
	"log"
	"net/http"
	"os"
	"path/filepath"
)

//go:embed web/*
var staticFiles embed.FS

func main() {
	log.Println("Starting Imaginarium Amicum application...")

	// Direct check for Stable Diffusion model file (for debugging)
	performDirectModelCheck()

	// Create and start the hub
	hub := NewHub()
	log.Println("Starting hub...")
	go hub.Run()

	// Start model setup in background
	log.Println("Starting model setup in background...")
	startModelSetup(hub)

	// Setup HTTP server
	setupHTTPServer(hub, http.FS(staticFiles))

	// Start the server
	port := ":8080"
	if err := startServer(port); err != nil {
		log.Fatalf("Server failed to start: %v", err)
	}
}

// performDirectModelCheck performs a direct check for model files (for debugging purposes)
func performDirectModelCheck() {
	modelDir, err := getModelDir()
	if err != nil {
		log.Printf("Error getting model directory: %v", err)
		return
	}

	sdModelPath := filepath.Join(modelDir, "v1-5-pruned-emaonly.safetensors")
	log.Printf("DIRECT CHECK: Checking for Stable Diffusion model at %s", sdModelPath)

	if fileInfo, err := os.Stat(sdModelPath); err == nil {
		log.Printf("DIRECT CHECK: Stable Diffusion model exists with size %d bytes", fileInfo.Size())
		minSDSize := int64(2 * 1024 * 1024 * 1024) // 2GB minimum
		if fileInfo.Size() >= minSDSize {
			log.Printf("DIRECT CHECK: Stable Diffusion model size is sufficient (>= %d bytes)", minSDSize)
		} else {
			log.Printf("DIRECT CHECK: Stable Diffusion model size is too small (< %d bytes)", minSDSize)
		}
	} else {
		log.Printf("DIRECT CHECK: Stable Diffusion model does not exist: %v", err)
	}
}
