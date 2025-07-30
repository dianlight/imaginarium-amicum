package main

import (
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"strconv"
	"time"
)

// Write implements io.Writer for ProgressWriter
func (pw *ProgressWriter) Write(p []byte) (n int, err error) {
	n, err = pw.Writer.Write(p)
	pw.Downloaded += int64(n)

	// Throttle updates to avoid spamming the channel/clients
	if time.Since(pw.lastUpdate) > 100*time.Millisecond || pw.Downloaded == pw.Total {
		pw.lastUpdate = time.Now()
		percent := 0.0
		if pw.Total > 0 {
			percent = float64(pw.Downloaded) / float64(pw.Total) * 100.0
		}
		pw.OnProgress(DownloadProgress{
			ModelName:  pw.ModelName,
			Downloaded: pw.Downloaded,
			Total:      pw.Total,
			Percent:    percent,
			Completed:  pw.Downloaded == pw.Total,
		})
	}
	return
}

// getModelDir determines the OS-specific directory for storing models
func getModelDir() (string, error) {
	// For user-specific application data that is persistent
	dataDir, err := os.UserConfigDir()
	if err != nil {
		return "", fmt.Errorf("failed to get user config directory: %w", err)
	}
	appDataDir := filepath.Join(dataDir, "imaginarium-amicum", "models")
	return appDataDir, nil
}

// downloadFile downloads a file from a URL to a specified path, reporting progress
func downloadFile(url, filepath, modelName string, notifyProgress func(progress DownloadProgress)) error {
	log.Printf("TRACE: Inside downloadFile function for %s model", modelName)

	// First, make a HEAD request to get the expected file size
	var expectedSize int64 = -1 // Default to unknown size
	client := &http.Client{
		CheckRedirect: func(req *http.Request, via []*http.Request) error {
			return nil // Allow redirects
		},
	}

	log.Printf("TRACE: Making HEAD request to %s for %s model", url, modelName)
	headResp, err := client.Head(url)
	if err != nil {
		log.Printf("Warning: Failed to get file size with HEAD request: %v. Will proceed with download anyway.", err)
	} else {
		defer headResp.Body.Close()
		expectedSize = headResp.ContentLength
		log.Printf("TRACE: Got content length %d for %s model", expectedSize, modelName)
		if linkedSize := headResp.Header.Get("X-Linked-Size"); linkedSize != "" {
			if size, err := strconv.ParseInt(linkedSize, 10, 64); err == nil && size > 0 {
				expectedSize = size
				log.Printf("Using X-Linked-Size header for expected file size: %d", expectedSize)
			}
		}
	}

	// Check if file already exists and has the correct size
	log.Printf("TRACE: Checking if %s model file exists at %s", modelName, filepath)
	if stat, err := os.Stat(filepath); err == nil && stat.Size() > 0 {
		log.Printf("TRACE: %s model file exists with size %d bytes", modelName, stat.Size())

		if expectedSize == -1 || stat.Size() == expectedSize {
			log.Printf("TRACE: %s model file size check - expectedSize: %d, actualSize: %d", modelName, expectedSize, stat.Size())

			if expectedSize == -1 {
				// Check minimum sizes for different model types
				minSize := getMinimumModelSize(modelName)

				if stat.Size() < minSize {
					log.Printf("%s model exists at %s but is too small (%d bytes). Minimum size: %d bytes. Redownloading.",
						modelName, filepath, stat.Size(), minSize)
					os.Remove(filepath)
				} else {
					log.Printf("%s model already exists at %s with sufficient size. Skipping download.", modelName, filepath)
					notifyProgress(DownloadProgress{
						ModelName:  modelName,
						Downloaded: stat.Size(),
						Total:      stat.Size(),
						Percent:    100.0,
						Completed:  true,
					})
					return nil
				}
			} else {
				// We know the expected size and it matches
				log.Printf("%s model already exists at %s with correct size. Skipping download.", modelName, filepath)
				notifyProgress(DownloadProgress{
					ModelName:  modelName,
					Downloaded: stat.Size(),
					Total:      stat.Size(),
					Percent:    100.0,
					Completed:  true,
				})
				return nil
			}
		} else {
			// File exists but has incorrect size, remove it and redownload
			log.Printf("%s model exists at %s but has incorrect size (%d vs expected %d). Redownloading.",
				modelName, filepath, stat.Size(), expectedSize)
			os.Remove(filepath)
		}
	} else {
		log.Printf("TRACE: %s model file does not exist or has zero size, will download", modelName)
	}

	return performDownload(client, url, filepath, modelName, expectedSize, notifyProgress)
}

// getMinimumModelSize returns the minimum expected size for different model types
func getMinimumModelSize(modelName string) int64 {
	switch modelName {
	case "Llama":
		return int64(1 * 1024 * 1024 * 1024) // 1GB minimum for Llama
	case "StableDiffusion":
		return int64(2 * 1024 * 1024 * 1024) // 2GB minimum for Stable Diffusion
	default:
		return int64(100 * 1024 * 1024) // 100MB default minimum
	}
}

// performDownload performs the actual file download
func performDownload(client *http.Client, url, filepath, modelName string, expectedSize int64, notifyProgress func(progress DownloadProgress)) error {
	log.Printf("Downloading %s model from %s to %s...", modelName, url, filepath)

	// Create a temporary file for downloading
	tmpFilePath := filepath + ".tmp"

	resp, err := client.Get(url)
	if err != nil {
		return fmt.Errorf("failed to send GET request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("bad status: %s", resp.Status)
	}

	// Update expected size if we didn't get it from HEAD request
	if expectedSize == -1 && resp.ContentLength > 0 {
		expectedSize = resp.ContentLength
		log.Printf("Got file size from GET request: %d bytes", expectedSize)
	}

	// Create a temporary file for downloading
	out, err := os.Create(tmpFilePath)
	if err != nil {
		return fmt.Errorf("failed to create temporary file %s: %w", tmpFilePath, err)
	}
	defer out.Close()

	writer := &ProgressWriter{
		Writer:     out,
		Total:      expectedSize,
		ModelName:  modelName,
		OnProgress: notifyProgress,
	}

	_, err = io.Copy(writer, resp.Body)
	if err != nil {
		// Clean up partial download if error occurs
		os.Remove(tmpFilePath)
		return fmt.Errorf("failed to write file: %w", err)
	}

	// Close the file before renaming
	out.Close()

	// Verify the downloaded file size
	if stat, err := os.Stat(tmpFilePath); err == nil {
		if expectedSize > 0 && stat.Size() != expectedSize {
			os.Remove(tmpFilePath)
			return fmt.Errorf("downloaded file size (%d) does not match expected size (%d)", stat.Size(), expectedSize)
		}
	}

	// Rename the temporary file to the final filename
	if err := os.Rename(tmpFilePath, filepath); err != nil {
		os.Remove(tmpFilePath) // Clean up temp file if rename fails
		return fmt.Errorf("failed to rename temporary file to final file: %w", err)
	}

	log.Printf("%s model downloaded successfully to %s", modelName, filepath)
	return nil
}

// setupModels handles creating the model directory and downloading models
func setupModels(hub *Hub) error {
	log.Println("Setting up models...")
	log.Println("TRACE: Inside setupModels function")

	modelDir, err := getModelDir()
	if err != nil {
		log.Printf("TRACE: Error getting model directory: %v", err)
		initialDownloadError = err
		return fmt.Errorf("failed to get model directory: %w", err)
	}

	log.Printf("Using model directory: %s", modelDir)

	// Ensure the model directory exists
	if err := os.MkdirAll(modelDir, 0755); err != nil {
		initialDownloadError = err
		return fmt.Errorf("failed to create model directory %s: %w", modelDir, err)
	}

	// Initialize global download status
	llamaDownloadStatus = DownloadProgress{ModelName: "Llama"}
	sdDownloadStatus = DownloadProgress{ModelName: "StableDiffusion"}

	// Set dynamic model paths
	llamaModelPath = filepath.Join(modelDir, "llama-2-7b-chat.Q4_K_M.gguf")
	sdModelPath = filepath.Join(modelDir, "v1-5-pruned-emaonly.safetensors")

	log.Printf("Llama model path: %s", llamaModelPath)
	log.Printf("Stable Diffusion model path: %s", sdModelPath)

	// Callback function to update global status and broadcast
	notifyFn := createProgressNotifier(hub)

	// Download models
	if err := downloadModels(notifyFn); err != nil {
		return err
	}

	log.Println("All models are ready.")
	downloadStatusMutex.Lock()
	serverIsReady = true
	downloadStatusMutex.Unlock()

	// Send final ready status
	hub.statusUpdateChan <- ServerStatus{
		Type:          "serverReady",
		Message:       "Server is ready! You can start chatting.",
		LlamaProgress: llamaDownloadStatus,
		SDProgress:    sdDownloadStatus,
		IsReady:       true,
	}

	return nil
}

// createProgressNotifier creates a progress notification function
func createProgressNotifier(hub *Hub) func(progress DownloadProgress) {
	return func(progress DownloadProgress) {
		// Update download status under mutex protection
		downloadStatusMutex.Lock()
		switch progress.ModelName {
		case "Llama":
			llamaDownloadStatus = progress
		case "StableDiffusion":
			sdDownloadStatus = progress
		}

		// Create status update while still holding mutex
		status := ServerStatus{
			Type:          "downloadProgress",
			Message:       fmt.Sprintf("Downloading Llama model: %.1f%%, Stable Diffusion model: %.1f%%", llamaDownloadStatus.Percent, sdDownloadStatus.Percent),
			LlamaProgress: llamaDownloadStatus,
			SDProgress:    sdDownloadStatus,
			IsReady:       false,
		}
		downloadStatusMutex.Unlock()

		// Send update to hub asynchronously to avoid blocking
		go func() {
			select {
			case hub.statusUpdateChan <- status:
				// Successfully sent update
			case <-time.After(100 * time.Millisecond):
				// Timeout if channel is blocked
				log.Println("Warning: Status update dropped due to blocked channel")
			}
		}()
	}
}

// downloadModels downloads both LLM and Stable Diffusion models
func downloadModels(notifyFn func(progress DownloadProgress)) error {
	log.Println("Starting Llama model download/check...")
	llamaErr := downloadFile(llamaModelURL, llamaModelPath, "Llama", notifyFn)
	if llamaErr != nil {
		log.Printf("Failed to download Llama model: %v", llamaErr)
		llamaDownloadStatus.ErrorMessage = llamaErr.Error()
		initialDownloadError = fmt.Errorf("llama download failed: %w", llamaErr)
	}

	log.Println("Starting Stable Diffusion model download/check...")
	sdErr := downloadFile(sdModelURL, sdModelPath, "StableDiffusion", notifyFn)
	if sdErr != nil {
		log.Printf("Failed to download Stable Diffusion model: %v", sdErr)
		sdDownloadStatus.ErrorMessage = sdErr.Error()
		if initialDownloadError == nil {
			initialDownloadError = fmt.Errorf("stable diffusion download failed: %w", sdErr)
		} else {
			initialDownloadError = fmt.Errorf("%w; stable diffusion download failed: %v", initialDownloadError, sdErr)
		}
	}

	if initialDownloadError != nil {
		log.Printf("Model download(s) failed. Server will not be fully functional. Error: %v", initialDownloadError)
		return initialDownloadError
	}

	return nil
}
