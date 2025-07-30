// main.go
package main

import (
	"bytes"
	"embed"
	"encoding/base64"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"runtime"
	"runtime/debug"
	"strconv"
	"sync"
	"time"

	"github.com/go-skynet/go-llama.cpp"
	"github.com/gorilla/websocket"
	sd "github.com/seasonjs/stable-diffusion" // Corrected Stable Diffusion binding
)

//go:embed web/*
var staticFiles embed.FS

const (
	defaultMessageContextLimit = 29
	maxTokens                  = 256
)

// These paths will now be determined at runtime based on the OS.
var llamaModelPath string
var sdModelPath string // Stable Diffusion model path re-added

// Model URLs for download
const (
	llamaModelURL = "https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/resolve/main/llama-2-7b-chat.Q4_K_M.gguf"
	sdModelURL    = "https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/v1-5-pruned-emaonly.safetensors" // SD model URL re-added
)

// gpuLayersStr will be set at build time using ldflags
var gpuLayersStr string

var upgrader = websocket.Upgrader{
	ReadBufferSize:  1024,
	WriteBufferSize: 1024,
	CheckOrigin: func(r *http.Request) bool {
		return true
	},
}

// ChatMessage represents a single message in the chat.
type ChatMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
	Image   string `json:"image,omitempty"` // Image field now will be populated by backend
}

// Chat stages for guiding the user through initial setup.
const (
	StageLanguage = iota
	StageSetting
	StageCharacter
	StageChatting
)

// ChatSession holds the context for a single user's chat.
type ChatSession struct {
	mu           sync.Mutex
	conn         *websocket.Conn
	history      []ChatMessage
	llm          *llama.LLama
	sdm          *sd.Model // Corrected Stable Diffusion instance type
	messageLimit int

	chatStage  int
	language   string
	setting    string
	characters string
}

// DownloadProgress represents the progress of a single model download.
type DownloadProgress struct {
	ModelName    string  `json:"modelName"`
	Downloaded   int64   `json:"downloadedBytes"`
	Total        int64   `json:"totalBytes"`
	Percent      float64 `json:"percent"`
	Completed    bool    `json:"completed"`
	ErrorMessage string  `json:"errorMessage,omitempty"`
}

// ServerStatus represents the overall status of the server for clients.
type ServerStatus struct {
	Type          string           `json:"type"` // "downloadProgress" or "serverReady"
	Message       string           `json:"message"`
	LlamaProgress DownloadProgress `json:"llamaProgress"`
	SDProgress    DownloadProgress `json:"sdProgress"` // SDProgress re-added
	IsReady       bool             `json:"isReady"`
}

// Global download status and mutex to protect it
var (
	llamaDownloadStatus  DownloadProgress
	sdDownloadStatus     DownloadProgress // sdDownloadStatus re-added
	downloadStatusMutex  sync.Mutex
	serverIsReady        bool // True once all models are downloaded/initialized
	initialDownloadError error
)

// MessageRequest encapsulates an incoming message with its session.
type MessageRequest struct {
	Session *ChatSession
	Message ChatMessage
}

// Hub maintains the set of active connections and broadcasts messages to the clients.
type Hub struct {
	sessions         map[*websocket.Conn]*ChatSession
	register         chan *websocket.Conn
	unregister       chan *websocket.Conn
	messageChan      chan *MessageRequest // Channel for incoming messages
	statusUpdateChan chan ServerStatus    // Channel for server-wide status updates
}

// NewHub creates and returns a new Hub instance.
func NewHub() *Hub {
	return &Hub{
		sessions:         make(map[*websocket.Conn]*ChatSession),
		register:         make(chan *websocket.Conn),
		unregister:       make(chan *websocket.Conn),
		messageChan:      make(chan *MessageRequest),
		statusUpdateChan: make(chan ServerStatus),
	}
}

// Run starts the hub, listening for connection changes and incoming messages.
// It also initializes the LLM and Stable Diffusion models once for all sessions.
func (h *Hub) Run() {
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

	var llm *llama.LLama
	var sdm *sd.Model // Corrected sdm variable type
	var err error

	// Wait until models are downloaded or an error occurs
	for {
		downloadStatusMutex.Lock()
		ready := serverIsReady
		dlErr := initialDownloadError
		downloadStatusMutex.Unlock()

		if dlErr != nil {
			log.Fatalf("Failed to initialize Hub due to model download error: %v", dlErr)
		}
		if ready {
			break
		}
		time.Sleep(500 * time.Millisecond) // Wait for downloads to complete
	}

	// Try to initialize with Metal first, if it fails, fall back to CPU
	log.Printf("Attempting to initialize Llama LLM with Metal GPU acceleration (layers: %d)", gpuLayers)
	llm, err = llama.New(llamaModelPath, llama.SetContext(maxTokens), llama.SetGPULayers(gpuLayers))
	if err != nil {
		log.Printf("Warning: Metal GPU initialization failed: %v", err)
		log.Printf("Falling back to CPU-only mode for Llama LLM")

		// Fall back to CPU-only mode (0 GPU layers)
		llm, err = llama.New(llamaModelPath, llama.SetContext(maxTokens), llama.SetGPULayers(0))
		if err != nil {
			log.Fatalf("Error initializing Llama LLM from %s even in CPU-only mode: %v", llamaModelPath, err)
		}
		log.Printf("Successfully initialized Llama LLM in CPU-only mode")
	} else {
		log.Printf("Successfully initialized Llama LLM with Metal GPU acceleration")
	}
	defer llm.Free()

	// Stable Diffusion Model initialization re-added
	options := sd.DefaultOptions // Corrected: sd.DefaultOptions is a function

	// Try with GPU first if requested
	if gpuLayers != 0 && runtime.GOOS != "darwin" {
		log.Printf("Attempting to initialize Stable Diffusion with GPU acceleration")
		options.GpuEnable = true

		sdm, err = sd.NewAutoModel(options)
		if err != nil {
			log.Printf("Warning: GPU initialization for Stable Diffusion failed: %v", err)
			log.Printf("Falling back to CPU-only mode for Stable Diffusion")

			// Fall back to CPU
			options.GpuEnable = false
		}
	} else {
		options.GpuEnable = false
	}

	// If we haven't successfully created the model yet (either because GPU was not requested or GPU init failed)
	if sdm == nil {
		sdm, err = sd.NewAutoModel(options)
		if err != nil {
			log.Fatalf("Error initializing Stable Diffusion model: %v", err)
		}
	}

	defer sdm.Close()

	// seasonjs/stable-diffusion does not expose SetLogCallback
	// sdm.SetLogCallback(func(level sd.LogLevel, msg string) {
	// 	log.Println(msg)
	// })

	log.Printf("Loading Stable Diffusion model from %s", sdModelPath)
	err = sdm.LoadFromFile(sdModelPath)
	if err != nil {
		log.Fatalf("Error loading Stable Diffusion model from %s: %v", sdModelPath, err)
	}

	log.Printf("Successfully loaded Stable Diffusion model (GPU enabled: %v)", options.GpuEnable)

	log.Println("LLM and Stable Diffusion models initialized successfully.") // Updated log message

	for {
		select {
		case conn := <-h.register:
			// Create a new chat session for each connected client
			session := &ChatSession{
				conn:         conn,
				history:      []ChatMessage{},
				llm:          llm, // Assign the shared LLM instance
				sdm:          sdm, // SD instance assignment re-added
				messageLimit: defaultMessageContextLimit,
				chatStage:    StageLanguage, // Start at the language selection stage
			}
			h.sessions[conn] = session
			log.Printf("Client connected: %s", conn.RemoteAddr())

			// Send current server status (including download progress or ready state)
			downloadStatusMutex.Lock()
			currentStatus := ServerStatus{
				Type:          "serverReady", // Assume ready if we got here
				Message:       "Server is ready! You can start chatting.",
				LlamaProgress: llamaDownloadStatus,
				SDProgress:    sdDownloadStatus, // SDProgress re-added
				IsReady:       serverIsReady,
			}
			if !serverIsReady { // If somehow a client connects before ready (shouldn't happen with current blocking logic)
				currentStatus.Type = "downloadProgress"
				currentStatus.Message = "Server is still downloading models."
			}
			downloadStatusMutex.Unlock()
			conn.WriteJSON(currentStatus)

			// Send the first chat prompt to the user only if the server is ready
			if serverIsReady {
				// Add the initial prompt to history so it persists
				initialPrompt := ChatMessage{Role: "assistant", Content: "Hello! What language would you like to use for our chat?"}
				session.history = append(session.history, initialPrompt)
				session.sendChatUpdate() // Send the history with the initial prompt
			}

		case conn := <-h.unregister:
			// Remove session when client disconnects
			if session, ok := h.sessions[conn]; ok {
				delete(h.sessions, conn)
				session.conn.Close()
				log.Printf("Client disconnected: %s", conn.RemoteAddr())
			}

		case req := <-h.messageChan:
			// Handle incoming chat messages in a new goroutine
			go h.handleChatMessage(req.Session, req.Message)

		case status := <-h.statusUpdateChan:
			// Broadcast server status updates (e.g., download progress) to all connected clients
			for conn := range h.sessions {
				conn.WriteJSON(status)
			}
		}
	}
}

// handleChatMessage processes an incoming chat message, generates an AI response,
// and then creates a contextualized image.
func (h *Hub) handleChatMessage(session *ChatSession, userMessage ChatMessage) {
	session.mu.Lock() // Ensure thread-safe access to session history
	defer session.mu.Unlock()

	// Handle initial chat setup stages
	switch session.chatStage {
	case StageLanguage:
		session.language = userMessage.Content
		session.history = append(session.history, userMessage) // Add user's language choice to history
		session.chatStage = StageSetting
		// Add the next prompt to history
		nextPrompt := ChatMessage{Role: "assistant", Content: "Great! Now, describe the general setting for our story/conversation (e.g., a futuristic city, a medieval kingdom, a quiet suburban house)."}
		session.history = append(session.history, nextPrompt)
		session.sendChatUpdate() // Send updated history to client
		return                   // Do not proceed to LLM generation yet
	case StageSetting:
		session.setting = userMessage.Content
		session.history = append(session.history, userMessage) // Add user's setting choice to history
		session.chatStage = StageCharacter
		// Add the next prompt to history - Updated message to include image generation context
		nextPrompt := ChatMessage{Role: "assistant", Content: "Finally, tell me about the main character(s) characteristics (e.g., a brave knight, a curious scientist, a mischievous cat). I'll also try to build a contextualized image based on my responses."}
		session.history = append(session.history, nextPrompt)
		session.sendChatUpdate() // Send updated history to client
		return                   // Do not proceed to LLM generation yet
	case StageCharacter:
		session.characters = userMessage.Content
		session.history = append(session.history, userMessage) // Add user's character choice to history
		session.chatStage = StageChatting
		nextPrompt := ChatMessage{Role: "assistant", Content: "Alright, let's start our chat! I'll generate responses and try to build contextualized images."}
		session.history = append(session.history, nextPrompt)
		session.sendChatUpdate()          // Send updated history to client
		session.history = []ChatMessage{} // Clear history to start fresh for chatting stage
		return                            // Send updated history to client
	}

	// For normal chatting stage:
	// 1. Add user message to history and send immediate update to client
	session.history = append(session.history, userMessage)
	session.sendChatUpdate()

	// 2. Notify client that assistant is thinking
	session.conn.WriteJSON(map[string]string{
		"type": "assistantThinking",
	})

	// 3. Summarize old messages if context limit is exceeded
	if len(session.history) > session.messageLimit {
		log.Printf("Chat history exceeding limit (%d). Summarizing...", session.messageLimit)
		summaryPrompt := "Summarize the following conversation concisely:\n"
		// Only summarize regular chat messages, not the initial context setup.
		// Start from the point where normal chatting began, or after the initial messages.
		startIndex := 0
		if session.language != "" && session.setting != "" && session.characters != "" {
			// Approximately skip the initial setup messages by user and assistant
			// This is a rough estimate; a more precise way would be to track their indices.
			startIndex = len(session.history) - session.messageLimit + 1
			if startIndex < 0 {
				startIndex = 0
			}
		}

		for i := startIndex; i < len(session.history); i++ {
			summaryPrompt += fmt.Sprintf("%s: %s\n", session.history[i].Role, session.history[i].Content)
		}

		summary, err := session.llm.Predict(summaryPrompt, llama.SetTokens(100), llama.SetTopK(40), llama.SetTopP(0.95), llama.SetTemperature(0.7))
		if err != nil {
			log.Printf("Error summarizing chat with LLM: %v. Keeping full history for now.", err)
			// Fallback: If summarization fails, just keep the last `messageLimit` messages.
			session.history = session.history[len(session.history)-session.messageLimit:]
		} else {
			log.Printf("Summary generated: %s", summary)
			newHistory := []ChatMessage{{Role: "system", Content: "Conversation Summary: " + summary}}
			// Append only the recent messages after the summary.
			newHistory = append(newHistory, session.history[startIndex:]...)
			session.history = newHistory
		}
		session.sendChatUpdate() // Send updated history with summary
	}

	// 4. Generate AI chat response using LLM
	var promptBuilder string

	// Always prepend the fixed context (language, setting, characters)
	fixedContext := fmt.Sprintf("The chat language is: %s. The setting is: %s. The main character(s) are: %s. The conversation follows. Reply as 'assistant:' followed by your reply. Make sure to follow the context. Do not repeat previous messages. Keep the tone consistent. Do not go off-topic. Only reply as 'assistant:', do not include any additional text.\n",
		session.language, session.setting, session.characters)
	promptBuilder += fixedContext

	for _, msg := range session.history {
		promptBuilder += fmt.Sprintf("%s: %s\n", msg.Role, msg.Content)
	}
	promptBuilder += "assistant:" // Indicate that assistant should respond

	log.Printf("Sending prompt to Llama LLM for chat response: %s", promptBuilder)

	var assistantResponse string
	prediction, err := session.llm.Predict(
		promptBuilder,
		llama.SetTokens(maxTokens),
		llama.SetTopK(40),
		llama.SetTopP(0.95),
		llama.SetTemperature(0.7),
		llama.SetSeed(int(time.Now().UnixNano())), // Use a new seed for each prediction
	)
	if err != nil {
		log.Printf("Llama LLM prediction error: %v", err)
		assistantResponse = "I'm sorry, I'm having trouble generating a response right now."
	} else {
		assistantResponse = prediction
	}

	assistantMessage := ChatMessage{
		Role:    "assistant",
		Content: assistantResponse,
	}
	// Add assistant's text response to history
	session.history = append(session.history, assistantMessage)

	log.Println("Starting image generation...")

	// Notify client that image generation has started
	session.conn.WriteJSON(map[string]string{
		"type": "imageGenerationStart",
	})

	imagePrompt := fmt.Sprintf("Generate a realistic image based on this description from an AI assistant, keeping the language, setting, and character context in mind. Focus on key visual elements. Description: \"%s\"", assistantResponse)
	// 5. Generate image based on AI response (re-added)
	log.Printf("Generating image based on AI response: %s", assistantResponse)
	// Optionally, add negative prompts or other SD parameters here
	sdOpts := sd.DefaultFullParams
	// NegativePrompt:   "out of frame, lowers, text, error, cropped, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, out of frame, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck, username, watermark, signature",
	//CfgScale:         7.0,
	//Width:            512,
	//Height:           512,
	//SampleMethod:     EULER_A,
	//SampleSteps:      20,
	sdOpts.SampleSteps = 1
	//Strength:         0.4,
	//Seed:             42,
	sdOpts.Seed = time.Now().UnixNano() // Use a new seed for each image
	//BatchCount:       1,
	//OutputsImageType: PNG,

	imgBuf := bytes.NewBuffer(make([]byte, 0, 1024*1024)) // Preallocate 1MB buffer for image data
	var imgs []io.Writer
	imgs = append(imgs, imgBuf)
	err = session.sdm.Predict(imagePrompt, sdOpts, imgs)
	if err != nil {
		log.Printf("Stable Diffusion image generation error: %v", err)
		// Set a placeholder image URL indicating an error
		assistantMessage.Image = "https://placehold.co/400x300/e5e7eb/6b7280?text=Image+Gen+Failed"
	} else {
		assistantMessage.Image = "data:image/png;base64," + base64.StdEncoding.EncodeToString(imgBuf.Bytes())
		log.Println("Image generated and base64 encoded successfully.")
	}

	session.history[len(session.history)-1].Image = assistantMessage.Image

	session.sendChatUpdate() // Send final updated history to client
}

// sendChatUpdate sends the current chat history to the client via WebSocket.
func (s *ChatSession) sendChatUpdate() {
	s.conn.WriteJSON(map[string]interface{}{
		"type":    "chatUpdate",
		"history": s.history,
	})
}

// sendMessage sends a specific control/status message to the client.
func (s *ChatSession) sendMessage(msgType, content string) {
	s.conn.WriteJSON(map[string]string{
		"type":    msgType,
		"content": content,
	})
}

// handleWebSocket upgrades the HTTP connection to a WebSocket connection
// and manages the client's session.
func (h *Hub) handleWebSocket(w http.ResponseWriter, r *http.Request) {
	conn, err := upgrader.Upgrade(w, r, nil)
	if err != nil {
		log.Printf("Failed to upgrade connection: %v", err)
		return
	}
	h.register <- conn // Register the new connection with the Hub

	defer func() {
		h.unregister <- conn // Unregister connection when function exits
	}()

	for {
		var msg ChatMessage
		err := conn.ReadJSON(&msg) // Read incoming JSON messages
		if err != nil {
			if websocket.IsCloseError(err, websocket.CloseNormalClosure, websocket.CloseGoingAway) {
				log.Printf("WebSocket closed by client: %v", err)
			} else {
				log.Printf("WebSocket read error: %v", err)
			}
			break // Exit loop on read error or close
		}

		// Handle specific commands like starting a new chat
		if msg.Content == "/newchat" {
			session := h.sessions[conn]
			session.mu.Lock()
			session.history = []ChatMessage{} // Clear history
			session.language = ""             // Reset fixed context
			session.setting = ""
			session.characters = ""
			session.chatStage = StageLanguage // Reset stage to start setup again
			// Add the initial prompt to history so it persists
			initialPrompt := ChatMessage{Role: "assistant", Content: "Hello! What language would you like to use for our chat?"}
			session.history = append(session.history, initialPrompt)
			session.mu.Unlock()
			session.sendChatUpdate() // This will now include the initial prompt in the history
			continue
		}

		// Send regular chat messages to the Hub's message channel
		h.messageChan <- &MessageRequest{
			Session: h.sessions[conn],
			Message: msg,
		}
	}
}

// getModelDir determines the OS-specific directory for storing models.
func getModelDir() (string, error) {
	// For user-specific application data that is persistent.
	dataDir, err := os.UserConfigDir()
	if err != nil {
		return "", fmt.Errorf("failed to get user config directory: %w", err)
	}
	appDataDir := filepath.Join(dataDir, "imaginarium-amicum", "models") // Updated app data dir name
	return appDataDir, nil
}

// ProgressWriter is an io.Writer that reports progress.
type ProgressWriter struct {
	io.Writer
	Total      int64
	Downloaded int64
	ModelName  string
	OnProgress func(progress DownloadProgress)
	lastUpdate time.Time
}

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

// downloadFile downloads a file from a URL to a specified path, reporting progress.
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
		// Continue with download even if HEAD fails
	} else {
		defer headResp.Body.Close()
		// Get expected size, checking for X-Linked-Size header which some CDNs use for the actual file size
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
		// If we couldn't determine expected size or the file size matches the expected size, consider it complete
		if expectedSize == -1 || stat.Size() == expectedSize {
			log.Printf("TRACE: %s model file size check - expectedSize: %d, actualSize: %d", modelName, expectedSize, stat.Size())
			// For unknown expected size, use a minimum size threshold (e.g., 100MB for LLM models)
			if expectedSize == -1 {
				// Minimum sizes for different model types (in bytes)
				minLlamaSize := int64(1 * 1024 * 1024 * 1024) // 1GB minimum for Llama
				minSDSize := int64(2 * 1024 * 1024 * 1024)    // 2GB minimum for Stable Diffusion

				var minSize int64
				if modelName == "Llama" {
					minSize = minLlamaSize
					log.Printf("TRACE: Using minimum size of %d bytes for Llama model", minSize)
				} else if modelName == "StableDiffusion" {
					minSize = minSDSize
					log.Printf("TRACE: Using minimum size of %d bytes for StableDiffusion model", minSize)
				} else {
					minSize = int64(100 * 1024 * 1024) // 100MB default minimum
					log.Printf("TRACE: Using default minimum size of %d bytes for unknown model type", minSize)
				}

				if stat.Size() < minSize {
					log.Printf("%s model exists at %s but is too small (%d bytes). Minimum size: %d bytes. Redownloading.",
						modelName, filepath, stat.Size(), minSize)
					os.Remove(filepath)
					log.Printf("TRACE: Removed %s model file due to insufficient size", modelName)
					// Continue to download
				} else {
					log.Printf("%s model already exists at %s with sufficient size. Skipping download.", modelName, filepath)
					log.Printf("TRACE: %s model file has sufficient size, setting download progress to complete", modelName)
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
				log.Printf("TRACE: %s model file has correct size, setting download progress to complete", modelName)
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
			log.Printf("TRACE: Removed %s model file due to incorrect size", modelName)
		}
	} else {
		log.Printf("TRACE: %s model file does not exist or has zero size, will download", modelName)
	}

	log.Printf("Downloading %s model from %s to %s...", modelName, url, filepath)
	// Create a temporary file for downloading
	tmpFilePath := filepath + ".tmp"

	// Use the same client with redirect support
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
		Total:      expectedSize, // Use our determined expected size
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

// setupModels handles creating the model directory and downloading models.
func setupModels(hub *Hub) error {
	log.Println("Setting up models...")
	log.Println("TRACE: Inside setupModels function")
	modelDir, err := getModelDir()
	if err != nil {
		log.Printf("TRACE: Error getting model directory: %v", err)
		initialDownloadError = err // Store error globally
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
	sdDownloadStatus = DownloadProgress{ModelName: "StableDiffusion"} // SD download status re-added

	// Set dynamic model paths
	llamaModelPath = filepath.Join(modelDir, "llama-2-7b-chat.Q4_K_M.gguf")
	sdModelPath = filepath.Join(modelDir, "v1-5-pruned-emaonly.safetensors") // SD model path setup re-added

	log.Printf("Llama model path: %s", llamaModelPath)
	log.Printf("Stable Diffusion model path: %s", sdModelPath)

	// Callback function to update global status and broadcast
	notifyFn := func(progress DownloadProgress) {
		// Update download status under mutex protection
		downloadStatusMutex.Lock()
		if progress.ModelName == "Llama" {
			llamaDownloadStatus = progress
		} else if progress.ModelName == "StableDiffusion" { // SD progress update logic re-added
			sdDownloadStatus = progress
		}

		// Create status update while still holding mutex
		status := ServerStatus{
			Type:          "downloadProgress",
			Message:       fmt.Sprintf("Downloading Llama model: %.1f%%, Stable Diffusion model: %.1f%%", llamaDownloadStatus.Percent, sdDownloadStatus.Percent),
			LlamaProgress: llamaDownloadStatus,
			SDProgress:    sdDownloadStatus, // SDProgress re-added
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

	log.Println("Starting Llama model download/check...")
	log.Println("TRACE: Before Llama model download")
	llamaErr := downloadFile(llamaModelURL, llamaModelPath, "Llama", notifyFn)
	log.Println("TRACE: After Llama model download")
	if llamaErr != nil {
		log.Printf("Failed to download Llama model: %v", llamaErr)
		llamaDownloadStatus.ErrorMessage = llamaErr.Error()
		initialDownloadError = fmt.Errorf("llama download failed: %w", llamaErr)
		log.Printf("TRACE: Set initialDownloadError to: %v", initialDownloadError)
	} else {
		log.Println("TRACE: Llama model download/check completed successfully")
	}

	log.Println("TRACE: Llama model download/check completed, now proceeding to Stable Diffusion model")
	log.Printf("TRACE: Llama download status: %+v", llamaDownloadStatus)
	log.Printf("TRACE: SD download status: %+v", sdDownloadStatus)
	log.Printf("TRACE: initialDownloadError: %v", initialDownloadError)

	// Stable Diffusion download logic re-added
	log.Println("Starting Stable Diffusion model download/check...")
	log.Println("TRACE: Before Stable Diffusion model check")
	log.Printf("TRACE: initialDownloadError before SD check: %v", initialDownloadError)

	// Check if SD model exists before download
	sdModelExists := false
	sdModelSizeSufficient := false

	if _, err := os.Stat(sdModelPath); err == nil {
		log.Println("TRACE: Stable Diffusion model file exists")
		sdModelExists = true
		fileInfo, err := os.Stat(sdModelPath)
		if err == nil && fileInfo.Size() > 0 {
			log.Printf("Stable Diffusion model already exists at %s with size %d bytes", sdModelPath, fileInfo.Size())
			// Check if size is at least 2GB (minimum size for SD model)
			minSDSize := int64(2 * 1024 * 1024 * 1024) // 2GB minimum
			if fileInfo.Size() >= minSDSize {
				log.Printf("Stable Diffusion model size (%d bytes) is sufficient (>= %d bytes)", fileInfo.Size(), minSDSize)
				log.Println("TRACE: Stable Diffusion model size is sufficient")
				sdModelSizeSufficient = true
			} else {
				log.Printf("Stable Diffusion model size (%d bytes) is too small (< %d bytes)", fileInfo.Size(), minSDSize)
				log.Println("TRACE: Stable Diffusion model size is too small")
			}
		} else {
			log.Printf("TRACE: Error getting Stable Diffusion model file info: %v", err)
		}
	} else {
		log.Printf("Stable Diffusion model does not exist at %s: %v", sdModelPath, err)
		log.Println("TRACE: Stable Diffusion model file does not exist")
	}

	log.Printf("TRACE: SD model exists: %v, SD model size sufficient: %v", sdModelExists, sdModelSizeSufficient)
	log.Printf("Attempting to download Stable Diffusion model from %s", sdModelURL)
	log.Printf("IMPORTANT: About to call downloadFile for Stable Diffusion model")
	log.Println("TRACE: Before Stable Diffusion model download")
	log.Printf("TRACE: initialDownloadError before SD download: %v", initialDownloadError)

	// If the model already exists with sufficient size, we might skip the actual download
	if sdModelExists && sdModelSizeSufficient {
		log.Println("TRACE: SD model exists with sufficient size, might skip actual download")
	}

	sdErr := downloadFile(sdModelURL, sdModelPath, "StableDiffusion", notifyFn)
	log.Println("TRACE: After Stable Diffusion model download")
	log.Printf("IMPORTANT: Returned from downloadFile for Stable Diffusion model with error: %v", sdErr)
	log.Printf("TRACE: SD download status after download attempt: %+v", sdDownloadStatus)

	if sdErr != nil {
		log.Printf("Failed to download Stable Diffusion model: %v", sdErr)
		sdDownloadStatus.ErrorMessage = sdErr.Error()
		if initialDownloadError == nil { // Only set if no prior error
			initialDownloadError = fmt.Errorf("stable diffusion download failed: %w", sdErr)
			log.Printf("TRACE: Set initialDownloadError to: %v", initialDownloadError)
		} else { // Append to existing error
			initialDownloadError = fmt.Errorf("%w; stable diffusion download failed: %v", initialDownloadError, sdErr)
			log.Printf("TRACE: Updated initialDownloadError to: %v", initialDownloadError)
		}
	} else {
		log.Printf("TRACE: Stable Diffusion model download/check completed successfully")
	}

	log.Printf("TRACE: Final SD download status: %+v", sdDownloadStatus)
	log.Printf("TRACE: Final initialDownloadError: %v", initialDownloadError)

	log.Println("TRACE: Checking for download errors")
	if initialDownloadError != nil {
		log.Printf("Model download(s) failed. Server will not be fully functional. Error: %v", initialDownloadError)
		// Update final status for clients
		log.Println("TRACE: Sending download error status update")
		hub.statusUpdateChan <- ServerStatus{
			Type:          "downloadError",
			Message:       fmt.Sprintf("Error during model download: %v", initialDownloadError),
			LlamaProgress: llamaDownloadStatus,
			SDProgress:    sdDownloadStatus, // SDProgress re-added
			IsReady:       false,
		}
		return initialDownloadError
	}

	log.Println("All models are ready.")
	downloadStatusMutex.Lock()
	serverIsReady = true // Mark server as ready after downloads
	downloadStatusMutex.Unlock()

	// Send final ready status
	log.Println("TRACE: Sending ready status update")
	hub.statusUpdateChan <- ServerStatus{
		Type:          "serverReady",
		Message:       "Server is ready! You can start chatting.",
		LlamaProgress: llamaDownloadStatus,
		SDProgress:    sdDownloadStatus, // SDProgress re-added
		IsReady:       true,
	}
	log.Println("TRACE: setupModels function completed successfully")
	return nil
}

func main() {
	log.Println("Starting application...")

	// Direct check for Stable Diffusion model file
	modelDir, err := getModelDir()
	if err != nil {
		log.Printf("Error getting model directory: %v", err)
	} else {
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

	hub := NewHub()
	// Start the Hub's goroutine
	log.Println("Starting hub...")
	go hub.Run()

	// Start model setup in a separate goroutine so main can proceed to serve static files
	// Hub.Run() will block until models are ready.
	log.Println("Starting model setup in background...")
	go func() {
		// Add panic recovery
		defer func() {
			if r := recover(); r != nil {
				log.Printf("PANIC in model setup goroutine: %v", r)
				debug.PrintStack()
			}
		}()

		log.Println("Background goroutine for model setup started")
		log.Println("IMPORTANT: About to call setupModels from goroutine")
		if err := setupModels(hub); err != nil {
			log.Printf("Error during model setup: %v", err)
			// Don't use Fatalf as it will terminate the program
			// initialDownloadError is already set and broadcasted by setupModels
		}
		log.Println("IMPORTANT: setupModels function completed in goroutine")
		log.Println("Model setup completed successfully")
	}()

	// Serve static files (HTML, CSS, JS) from the embedded 'web' directory
	fs := http.FileServer(http.FS(staticFiles))
	http.Handle("/", http.StripPrefix("/", fs))
	// Handle WebSocket connections
	http.HandleFunc("/ws", hub.handleWebSocket)

	port := ":8080"
	log.Printf("Server starting on port %s", port)
	serverErr := http.ListenAndServe(port, nil)
	if serverErr != nil {
		log.Fatalf("Server failed to start: %v", serverErr)
	}
}
