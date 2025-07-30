package main

import (
	"log"
	"net/http"
	"runtime/debug"

	"github.com/gorilla/websocket"
)

// handleWebSocket upgrades the HTTP connection to a WebSocket connection
// and manages the client's session
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
			session.resetSession()
			continue
		}

		// Send regular chat messages to the Hub's message channel
		h.messageChan <- &MessageRequest{
			Session: h.sessions[conn],
			Message: msg,
		}
	}
}

// startModelSetup starts the model setup process in a background goroutine
func startModelSetup(hub *Hub) {
	go func() {
		// Add panic recovery
		defer func() {
			if r := recover(); r != nil {
				log.Printf("PANIC in model setup goroutine: %v", r)
				debug.PrintStack()
			}
		}()

		log.Println("Background goroutine for model setup started")
		if err := setupModels(hub); err != nil {
			log.Printf("Error during model setup: %v", err)
			// Don't use Fatalf as it will terminate the program
			// initialDownloadError is already set and broadcasted by setupModels
		}
		log.Println("Model setup completed successfully")
	}()
}

// setupHTTPServer sets up the HTTP server with static file serving and WebSocket handling
func setupHTTPServer(hub *Hub, staticFiles http.FileSystem) {
	// Serve static files (HTML, CSS, JS) from the embedded 'web' directory
	fs := http.FileServer(staticFiles)
	http.Handle("/", http.StripPrefix("/", fs))

	// Handle WebSocket connections
	http.HandleFunc("/ws", hub.handleWebSocket)
}

// startServer starts the HTTP server on the specified port
func startServer(port string) error {
	log.Printf("Server starting on port %s", port)
	return http.ListenAndServe(port, nil)
}
