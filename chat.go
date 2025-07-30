package main

import (
	"bytes"
	"encoding/base64"
	"fmt"
	"image/png"
	"log"
	"time"

	"github.com/binozo/gostablediffusion/pkg/sd"
	"github.com/go-skynet/go-llama.cpp"
)

// sendChatUpdate sends the current chat history to the client via WebSocket
func (s *ChatSession) sendChatUpdate() {
	s.conn.WriteJSON(map[string]interface{}{
		"type":    "chatUpdate",
		"history": s.history,
	})
}

// sendMessage sends a specific control/status message to the client
func (s *ChatSession) sendMessage(msgType, content string) {
	s.conn.WriteJSON(map[string]string{
		"type":    msgType,
		"content": content,
	})
}

// handleChatStages processes the initial chat setup stages
func (s *ChatSession) handleChatStages(userMessage ChatMessage) bool {
	switch s.chatStage {
	case StageLanguage:
		s.language = userMessage.Content
		s.history = append(s.history, userMessage)
		s.chatStage = StageSetting
		nextPrompt := ChatMessage{
			Role:    "assistant",
			Content: "Great! Now, describe the general setting for our story/conversation (e.g., a futuristic city, a medieval kingdom, a quiet suburban house).",
		}
		s.history = append(s.history, nextPrompt)
		s.sendChatUpdate()
		return true

	case StageSetting:
		s.setting = userMessage.Content
		s.history = append(s.history, userMessage)
		s.chatStage = StageCharacter
		nextPrompt := ChatMessage{
			Role:    "assistant",
			Content: "Finally, tell me about the main character(s) characteristics (e.g., a brave knight, a curious scientist, a mischievous cat). I'll also try to build a contextualized image based on my responses.",
		}
		s.history = append(s.history, nextPrompt)
		s.sendChatUpdate()
		return true

	case StageCharacter:
		s.characters = userMessage.Content
		s.history = append(s.history, userMessage)
		s.chatStage = StageChatting
		nextPrompt := ChatMessage{
			Role:    "assistant",
			Content: "Alright, let's start our chat! I'll generate responses and try to build contextualized images.",
		}
		s.history = append(s.history, nextPrompt)
		s.sendChatUpdate()
		s.history = []ChatMessage{} // Clear history to start fresh for chatting stage
		return true
	}
	return false
}

// summarizeHistory summarizes old messages if context limit is exceeded
func (s *ChatSession) summarizeHistory() {
	if len(s.history) <= s.messageLimit {
		return
	}

	log.Printf("Chat history exceeding limit (%d). Summarizing...", s.messageLimit)
	summaryPrompt := "Summarize the following conversation concisely:\n"

	// Only summarize regular chat messages, not the initial context setup
	startIndex := 0
	if s.language != "" && s.setting != "" && s.characters != "" {
		startIndex = len(s.history) - s.messageLimit + 1
		if startIndex < 0 {
			startIndex = 0
		}
	}

	for i := startIndex; i < len(s.history); i++ {
		summaryPrompt += fmt.Sprintf("%s: %s\n", s.history[i].Role, s.history[i].Content)
	}

	summary, err := s.llm.Predict(summaryPrompt, llama.SetTokens(100), llama.SetTopK(40), llama.SetTopP(0.95), llama.SetTemperature(0.7))
	if err != nil {
		log.Printf("Error summarizing chat with LLM: %v. Keeping full history for now.", err)
		// Fallback: If summarization fails, just keep the last `messageLimit` messages
		s.history = s.history[len(s.history)-s.messageLimit:]
	} else {
		log.Printf("Summary generated: %s", summary)
		newHistory := []ChatMessage{{Role: "system", Content: "Conversation Summary: " + summary}}
		// Append only the recent messages after the summary
		newHistory = append(newHistory, s.history[startIndex:]...)
		s.history = newHistory
	}
	s.sendChatUpdate() // Send updated history with summary
}

// generateAIResponse generates an AI chat response using LLM
func (s *ChatSession) generateAIResponse() string {
	var promptBuilder string

	// Always prepend the fixed context (language, setting, characters)
	fixedContext := fmt.Sprintf("The chat language is: %s. The setting is: %s. The main character(s) are: %s."+
		"The conversation follows."+
		"Reply as any character(s) described above or as narrator '<character name>: <text>' format."+
		"Make sure to follow the context. Do not repeat previous messages. Keep the tone consistent. Do not go off-topic. Only reply do not include any additional text.\n",
		s.language, s.setting, s.characters)
	promptBuilder += fixedContext

	for _, msg := range s.history {
		promptBuilder += fmt.Sprintf("%s: %s\n", msg.Role, msg.Content)
	}
	promptBuilder += "<character name>:" // Indicate that assistant should respond

	log.Printf("Sending prompt to Llama LLM for chat response: %s", promptBuilder)

	prediction, err := s.llm.Predict(
		promptBuilder,
		llama.SetTokens(maxTokens),
		llama.SetTopK(40),
		llama.SetTopP(0.95),
		llama.SetTemperature(0.7),
		llama.SetSeed(int(time.Now().UnixNano())), // Use a new seed for each prediction
	)
	if err != nil {
		log.Printf("Llama LLM prediction error: %v", err)
		return "I'm sorry, I'm having trouble generating a response right now."
	}

	return prediction
}

// generateImage generates an image based on AI response using Stable Diffusion
func (s *ChatSession) generateImage(assistantResponse string) string {
	log.Println("Starting image generation...")

	// Notify client that image generation has started
	s.conn.WriteJSON(map[string]string{
		"type": "imageGenerationStart",
	})

	imagePrompt := fmt.Sprintf(
		"Generate a realistic image based on this description from an AI assistant, keeping the language, setting, and character context in mind. Focus on key visual elements."+
			" Description: \"%s\"", assistantResponse)

	log.Printf("Generating image based on AI response: %s", assistantResponse)

	// Create parameters for image generation
	params := sd.NewDefaultParams()
	params.CfgScale = 7.0
	params.SampleSteps = 5 // Increased from 1 for better quality
	params.SampleMethod = sd.Euler
	params.Height = 256
	params.Width = 256
	params.Seed = time.Now().UnixNano() // Use a new seed for each image
	params.Prompt = imagePrompt
	// Optional negative prompt for better quality
	params.NegativePrompt = "out of frame, text, error, cropped, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck, username, watermark, signature"

	// Generate the image
	result := s.sdCtx.Text2Img(params)

	// Convert image to base64
	imgBuf := bytes.NewBuffer(make([]byte, 0, 1024*1024))
	err := png.Encode(imgBuf, result.Image())
	if err != nil {
		log.Printf("Error encoding generated image to PNG: %v", err)
		return "https://placehold.co/400x300/e5e7eb/6b7280?text=Image+Encode+Failed"
	}

	log.Println("Image generated and base64 encoded successfully.")
	return "data:image/png;base64," + base64.StdEncoding.EncodeToString(imgBuf.Bytes())
}

// resetSession resets the chat session to initial state
func (s *ChatSession) resetSession() {
	s.mu.Lock()
	defer s.mu.Unlock()

	s.history = []ChatMessage{}
	s.language = ""
	s.setting = ""
	s.characters = ""
	s.chatStage = StageLanguage

	// Add the initial prompt to history
	initialPrompt := ChatMessage{
		Role:    "assistant",
		Content: "Hello! What language would you like to use for our chat?",
	}
	s.history = append(s.history, initialPrompt)
	s.sendChatUpdate()
}
