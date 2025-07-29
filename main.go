// main.go
package main

import (
	"bytes"
	"embed"
	"encoding/base64"
	"fmt"
	"image/png"
	"io"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"strconv"
	"sync"
	"time"

	"github.com/go-skynet/go-llama.cpp"
	sd "github.com/seasonjs/stable-diffusion" // Corrected Stable Diffusion binding
	"github.com/gorilla/websocket"
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
	llamaDownloadStatus   DownloadProgress
	sdDownloadStatus      DownloadProgress // sdDownloadStatus re-added
	downloadStatusMutex   sync.Mutex
	serverIsReady         bool // True once all models are downloaded/initialized
	initialDownloadError  error
)

// MessageRequest encapsulates an incoming message with its session.
type MessageRequest struct {
	Session *ChatSession
	Message ChatMessage
}


// Hub maintains the set of active connections and broadcasts messages to the clients.
type Hub struct {
	sessions       map[*websocket.Conn]*ChatSession
	register       chan *websocket.Conn
	unregister     chan *websocket.Conn
	messageChan    chan *MessageRequest // Channel for incoming messages
	statusUpdateChan chan ServerStatus // Channel for server-wide status updates
}

// NewHub creates and returns a new Hub instance.
func NewHub() *Hub {
	return &Hub{
		sessions:       make(map[*websocket.Conn]*ChatSession),
		register:       make(chan *websocket.Conn),
		unregister:     make(chan *websocket.Conn),
		messageChan:    make(chan *MessageRequest),
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

	llm, err = llama.New(llamaModelPath, llama.SetContext(defaultMessageContextLimit*2), llama.SetGPULayers(gpuLayers))
	if err != nil {
		log.Fatalf("Error initializing Llama LLM from %s: %v", llamaModelPath, err)
	}
	defer llm.Free()

	// Stable Diffusion Model initialization re-added
	options := sd.DefaultOptions    // Corrected: sd.DefaultOptions is a function
	if gpuLayers != 0 {
		options.GpuEnable = true             
	} else {
		options.GpuEnable = false            
	}

	sdm, err = sd.NewAutoModel(options)
	if err != nil {
		log.Fatalf("Error initializing Stable Diffusion model: %v", err)
	}
	defer sdm.Close()

	// seasonjs/stable-diffusion does not expose SetLogCallback
	// sdm.SetLogCallback(func(level sd.LogLevel, msg string) {
	// 	log.Println(msg)
	// })

	err = sdm.LoadFromFile(sdModelPath)
	if err != nil {
		log.Fatalf("Error loading Stable Diffusion model from %s: %v", sdModelPath, err)
	}


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
				session.sendMessage("initialPrompt", "Hello! What language would you like to use for our chat?")
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
	if session.chatStage == StageLanguage {
		session.language = userMessage.Content
		session.history = append(session.history, userMessage) // Add user's language choice to history
		session.chatStage = StageSetting
		session.sendMessage("initialPrompt", "Great! Now, describe the general setting for our story/conversation (e.g., a futuristic city, a medieval kingdom, a quiet suburban house).")
		session.sendChatUpdate() // Send updated history to client
		return // Do not proceed to LLM generation yet
	} else if session.chatStage == StageSetting {
		session.setting = userMessage.Content
		session.history = append(session.history, userMessage) // Add user's setting choice to history
		session.chatStage = StageCharacter
		// Updated message to include image generation context
		session.sendMessage("initialPrompt", "Finally, tell me about the main character(s) characteristics (e.g., a brave knight, a curious scientist, a mischievous cat). I'll also try to build a contextualized image based on my responses.")
		session.sendChatUpdate() // Send updated history to client
		return // Do not proceed to LLM generation yet
	} else if session.chatStage == StageCharacter {
		session.characters = userMessage.Content
		session.history = append(session.history, userMessage) // Add user's character choice to history
		session.chatStage = StageChatting
		session.sendMessage("status", "Alright, let's start our chat! I'll generate responses and try to build contextualized images.") // Updated message
		session.sendChatUpdate() // Send updated history to client
		// Fall through to normal chat processing now
	}

	// For normal chatting stage:
	// 1. Add user message to history and send immediate update to client
	session.history = append(session.history, userMessage)
	session.sendChatUpdate()

	// 2. Summarize old messages if context limit is exceeded
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

	// 3. Generate AI chat response using LLM
	var promptBuilder string

	// Always prepend the fixed context (language, setting, characters)
	fixedContext := fmt.Sprintf("The chat language is: %s. The setting is: %s. The main character(s) are: %s.\n",
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
	imagePrompt := fmt.Sprintf("Generate a realistic image based on this description from an AI assistant, keeping the language, setting, and character context in mind. Focus on key visual elements. Description: \"%s\"", assistantResponse)
	// 4. Generate image based on AI response (re-added)
	log.Printf("Generating image based on AI response: %s", assistantResponse)
	// Optionally, add negative prompts or other SD parameters here
	sdOpts := sd.DefaultOptions
	sdOpts.BatchCount = 1
	sdOpts.Width = 512
	sdOpts.Height = 512
	//sdOpts.SampleMethod = sd.SampleMethodEulerA
	sdOpts.SampleSteps = 20
	sdOpts.CfgScale = 7.0
	sdOpts.ClipSkip = 1 // Common setting for SD 1.5
	sdOpts.NegativePrompt = "ugly, deformed, disfigured, low quality, bad anatomy, bad art, blurry, out of focus"
	sdOpts.Seed = uint32(time.Now().UnixNano()) // Use a new seed for each image

	var imgBuf bytes.Buffer
	var imgs []io.Writer
	imgs = append(imgs, &imgBuf)
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
			session.mu.Unlock()
			session.sendChatUpdate()
			session.sendMessage("initialPrompt", "Hello! What language would you like to use for our chat?") // Send first prompt
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
	// Check if file already exists and is not zero-sized
	if stat, err := os.Stat(filepath); err == nil && stat.Size() > 0 {
		log.Printf("%s model already exists at %s. Skipping download.", modelName, filepath)
		notifyProgress(DownloadProgress{
			ModelName:  modelName,
			Downloaded: stat.Size(),
			Total:      stat.Size(),
			Percent:    100.0,
			Completed:  true,
		})
		return nil
	}

	log.Printf("Downloading %s model from %s to %s...", modelName, url, filepath)
	resp, err := http.Get(url)
	if err != nil {
		return fmt.Errorf("failed to send GET request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("bad status: %s", resp.Status)
	}

	out, err := os.Create(filepath)
	if err != nil {
		return fmt.Errorf("failed to create file %s: %w", filepath, err)
	}
	defer out.Close()

	writer := &ProgressWriter{
		Writer:     out,
		Total:      resp.ContentLength,
		ModelName:  modelName,
		OnProgress: notifyProgress,
	}

	_, err = io.Copy(writer, resp.Body)
	if err != nil {
		// Clean up partial download if error occurs
		os.Remove(filepath)
		return fmt.Errorf("failed to write file: %w", err)
	}

	return nil
}

// setupModels handles creating the model directory and downloading models.
func setupModels(hub *Hub) error {
	modelDir, err := getModelDir()
	if err != nil {
		initialDownloadError = err // Store error globally
		return fmt.Errorf("failed to get model directory: %w", err)
	}

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

	// Callback function to update global status and broadcast
	notifyFn := func(progress DownloadProgress) {
		downloadStatusMutex.Lock()
		defer downloadStatusMutex.Unlock()
		if progress.ModelName == "Llama" {
			llamaDownloadStatus = progress
		} else if progress.ModelName == "StableDiffusion" { // SD progress update logic re-added
			sdDownloadStatus = progress
		}

		// Send update to hub for broadcast
		hub.statusUpdateChan <- ServerStatus{
			Type:    "downloadProgress",
			Message: fmt.Sprintf("Downloading Llama model: %.1f%%, Stable Diffusion model: %.1f%%", llamaDownloadStatus.Percent, sdDownloadStatus.Percent),
			LlamaProgress: llamaDownloadStatus,
			SDProgress:    sdDownloadStatus, // SDProgress re-added
			IsReady: false,
		}
	}

	log.Println("Starting Llama model download/check...")
	llamaErr := downloadFile(llamaModelURL, llamaModelPath, "Llama", notifyFn)
	if llamaErr != nil {
		log.Printf("Failed to download Llama model: %v", llamaErr)
		llamaDownloadStatus.ErrorMessage = llamaErr.Error()
		initialDownloadError = fmt.Errorf("llama download failed: %w", llamaErr)
	}

	// Stable Diffusion download logic re-added
	log.Println("Starting Stable Diffusion model download/check...")
	sdErr := downloadFile(sdModelURL, sdModelPath, "StableDiffusion", notifyFn)
	if sdErr != nil {
		log.Printf("Failed to download Stable Diffusion model: %v", sdErr)
		sdDownloadStatus.ErrorMessage = sdErr.Error()
		if initialDownloadError == nil { // Only set if no prior error
			initialDownloadError = fmt.Errorf("stable diffusion download failed: %w", sdErr)
		} else { // Append to existing error
			initialDownloadError = fmt.Errorf("%w; stable diffusion download failed: %v", initialDownloadError, sdErr)
		}
	}

	if initialDownloadError != nil {
		log.Printf("Model download(s) failed. Server will not be fully functional. Error: %v", initialDownloadError)
		// Update final status for clients
		hub.statusUpdateChan <- ServerStatus{
			Type:    "downloadError",
			Message: fmt.Sprintf("Error during model download: %v", initialDownloadError),
			LlamaProgress: llamaDownloadStatus,
			SDProgress:    sdDownloadStatus, // SDProgress re-added
			IsReady: false,
		}
		return initialDownloadError
	}

	log.Println("All models are ready.")
	downloadStatusMutex.Lock()
	serverIsReady = true // Mark server as ready after downloads
	downloadStatusMutex.Unlock()

	// Send final ready status
	hub.statusUpdateChan <- ServerStatus{
		Type:    "serverReady",
		Message: "Server is ready! You can start chatting.",
		LlamaProgress: llamaDownloadStatus,
		SDProgress:    sdDownloadStatus, // SDProgress re-added
		IsReady: true,
	}
	return nil
}


func main() {
	hub := NewHub()
	// Start the Hub's goroutine
	go hub.Run()

	// Start model setup in a separate goroutine so main can proceed to serve static files
	// Hub.Run() will block until models are ready.
	go func() {
		if err := setupModels(hub); err != nil {
			log.Fatalf("Fatal error during model setup: %v", err)
			// At this point, initialDownloadError is already set and broadcasted by setupModels
			// If it's a fatal error, the server can't really function, so we might exit or go into a degraded mode.
			// For now, let's just log and let Hub.Run's check handle it.
		}
	}()


	// Serve static files (HTML, CSS, JS) from the embedded 'web' directory
	fs := http.FileServer(http.FS(staticFiles))
	http.Handle("/", http.StripPrefix("/", fs))
	// Handle WebSocket connections
	http.HandleFunc("/ws", hub.handleWebSocket)

	port := ":8080"
	log.Printf("Server starting on port %s", port)
	err := http.ListenAndServe(port, nil)
	if err != nil {
		log.Fatalf("Server failed to start: %v", err)
	}
}

