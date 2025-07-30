package main

import (
	"net/http"
	"sync"
	"time"

	"github.com/binozo/gostablediffusion/pkg/sd"
	"github.com/go-skynet/go-llama.cpp"
	"github.com/gorilla/websocket"
)

// Constants
const (
	defaultMessageContextLimit = 29
	maxTokens                  = 256
)

// Model URLs for download
const (
	llamaModelURL = "https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/resolve/main/llama-2-7b-chat.Q4_K_M.gguf"
	sdModelURL    = "https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/v1-5-pruned-emaonly.safetensors"
)

// Chat stages for guiding the user through initial setup
const (
	StageLanguage = iota
	StageSetting
	StageCharacter
	StageChatting
)

// ChatMessage represents a single message in the chat
type ChatMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
	Image   string `json:"image,omitempty"` // Image field populated by backend
}

// ChatSession holds the context for a single user's chat
type ChatSession struct {
	mu           sync.Mutex
	conn         *websocket.Conn
	history      []ChatMessage
	llm          *llama.LLama
	sdCtx        *sd.Context
	messageLimit int

	chatStage  int
	language   string
	setting    string
	characters string
}

// DownloadProgress represents the progress of a single model download
type DownloadProgress struct {
	ModelName    string  `json:"modelName"`
	Downloaded   int64   `json:"downloadedBytes"`
	Total        int64   `json:"totalBytes"`
	Percent      float64 `json:"percent"`
	Completed    bool    `json:"completed"`
	ErrorMessage string  `json:"errorMessage,omitempty"`
}

// ServerStatus represents the overall status of the server for clients
type ServerStatus struct {
	Type          string           `json:"type"` // "downloadProgress" or "serverReady"
	Message       string           `json:"message"`
	LlamaProgress DownloadProgress `json:"llamaProgress"`
	SDProgress    DownloadProgress `json:"sdProgress"`
	IsReady       bool             `json:"isReady"`
}

// MessageRequest encapsulates an incoming message with its session
type MessageRequest struct {
	Session *ChatSession
	Message ChatMessage
}

// Hub maintains the set of active connections and broadcasts messages to the clients
type Hub struct {
	sessions         map[*websocket.Conn]*ChatSession
	register         chan *websocket.Conn
	unregister       chan *websocket.Conn
	messageChan      chan *MessageRequest // Channel for incoming messages
	statusUpdateChan chan ServerStatus    // Channel for server-wide status updates
}

// ProgressWriter is an io.Writer that reports progress
type ProgressWriter struct {
	Writer     interface{ Write([]byte) (int, error) }
	Total      int64
	Downloaded int64
	ModelName  string
	OnProgress func(progress DownloadProgress)
	lastUpdate time.Time
}

// Global variables
var (
	// Model paths determined at runtime based on OS
	llamaModelPath string
	sdModelPath    string

	// GPU layers string set at build time using ldflags
	gpuLayersStr string

	// Global download status and mutex to protect it
	llamaDownloadStatus  DownloadProgress
	sdDownloadStatus     DownloadProgress
	downloadStatusMutex  sync.Mutex
	serverIsReady        bool // True once all models are downloaded/initialized
	initialDownloadError error

	// WebSocket upgrader
	upgrader = websocket.Upgrader{
		ReadBufferSize:  1024,
		WriteBufferSize: 1024,
		CheckOrigin: func(r *http.Request) bool {
			return true
		},
	}
)
