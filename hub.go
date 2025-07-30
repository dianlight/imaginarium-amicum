package main

import (
	"log"
	"time"

	"github.com/gorilla/websocket"
)

// NewHub creates and returns a new Hub instance
func NewHub() *Hub {
	return &Hub{
		sessions:         make(map[*websocket.Conn]*ChatSession),
		register:         make(chan *websocket.Conn),
		unregister:       make(chan *websocket.Conn),
		messageChan:      make(chan *MessageRequest),
		statusUpdateChan: make(chan ServerStatus),
	}
}

// Run starts the hub, listening for connection changes and incoming messages
// It also initializes the LLM and Stable Diffusion models once for all sessions
func (h *Hub) Run() {
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

	// Initialize AI models
	models, err := InitializeAIModels()
	if err != nil {
		log.Fatalf("Error initializing AI models: %v", err)
	}
	defer models.Free()

	log.Println("LLM and Stable Diffusion models initialized successfully.")

	for {
		select {
		case conn := <-h.register:
			h.handleClientConnection(conn, models)

		case conn := <-h.unregister:
			h.handleClientDisconnection(conn)

		case req := <-h.messageChan:
			// Handle incoming chat messages in a new goroutine
			go h.handleChatMessage(req.Session, req.Message)

		case status := <-h.statusUpdateChan:
			// Broadcast server status updates to all connected clients
			h.broadcastStatus(status)
		}
	}
}

// handleClientConnection handles new client connections
func (h *Hub) handleClientConnection(conn *websocket.Conn, models *AIModels) {
	// Create a new chat session for each connected client
	session := &ChatSession{
		conn:         conn,
		history:      []ChatMessage{},
		llm:          models.LLM,
		sdCtx:        models.SDCtx,
		messageLimit: defaultMessageContextLimit,
		chatStage:    StageLanguage, // Start at the language selection stage
	}
	h.sessions[conn] = session
	log.Printf("Client connected: %s", conn.RemoteAddr())

	// Send current server status
	h.sendServerStatus(conn)

	// Send the first chat prompt to the user only if the server is ready
	if serverIsReady {
		initialPrompt := ChatMessage{
			Role:    "assistant",
			Content: "Hello! What language would you like to use for our chat?",
		}
		session.history = append(session.history, initialPrompt)
		session.sendChatUpdate()
	}
}

// handleClientDisconnection handles client disconnections
func (h *Hub) handleClientDisconnection(conn *websocket.Conn) {
	if session, ok := h.sessions[conn]; ok {
		delete(h.sessions, conn)
		session.conn.Close()
		log.Printf("Client disconnected: %s", conn.RemoteAddr())
	}
}

// sendServerStatus sends the current server status to a specific connection
func (h *Hub) sendServerStatus(conn *websocket.Conn) {
	downloadStatusMutex.Lock()
	currentStatus := ServerStatus{
		Type:          "serverReady", // Assume ready if we got here
		Message:       "Server is ready! You can start chatting.",
		LlamaProgress: llamaDownloadStatus,
		SDProgress:    sdDownloadStatus,
		IsReady:       serverIsReady,
	}
	if !serverIsReady {
		currentStatus.Type = "downloadProgress"
		currentStatus.Message = "Server is still downloading models."
	}
	downloadStatusMutex.Unlock()
	conn.WriteJSON(currentStatus)
}

// broadcastStatus broadcasts server status updates to all connected clients
func (h *Hub) broadcastStatus(status ServerStatus) {
	for conn := range h.sessions {
		conn.WriteJSON(status)
	}
}

// handleChatMessage processes an incoming chat message, generates an AI response,
// and then creates a contextualized image
func (h *Hub) handleChatMessage(session *ChatSession, userMessage ChatMessage) {
	session.mu.Lock()
	defer session.mu.Unlock()

	// Handle initial chat setup stages
	if session.handleChatStages(userMessage) {
		return // Stage handling completed, no need to proceed to LLM generation
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
	session.summarizeHistory()

	// 4. Generate AI chat response using LLM
	assistantResponse := session.generateAIResponse()

	assistantMessage := ChatMessage{
		Role:    "assistant",
		Content: assistantResponse,
	}
	// Add assistant's text response to history
	session.history = append(session.history, assistantMessage)

	// 5. Generate image based on AI response
	imageData := session.generateImage(assistantResponse)
	session.history[len(session.history)-1].Image = imageData

	session.sendChatUpdate() // Send final updated history to client
}
