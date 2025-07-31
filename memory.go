package main

import (
	"context"
	"fmt"
	"log"

	"github.com/philippgille/chromem-go"
)

// initializeMemoryDB creates a new in-memory database with context and characters collections
func (s *ChatSession) initializeMemoryDB() error {
	// Create a new in-memory database
	db := chromem.NewDB()

	// Create context collection for storing language, setting, and characters info
	contextCollection, err := db.CreateCollection("context", nil, nil)
	if err != nil {
		return fmt.Errorf("failed to create context collection: %w", err)
	}

	// Create characters collection for storing character history
	charactersCollection, err := db.CreateCollection("characters", nil, nil)
	if err != nil {
		return fmt.Errorf("failed to create characters collection: %w", err)
	}

	s.memoryDB = db
	s.contextCollection = contextCollection
	s.charactersCollection = charactersCollection

	log.Println("Memory database initialized with context and characters collections")
	return nil
}

// storeContextInfo stores the context information (language, setting, characters) in the context collection
func (s *ChatSession) storeContextInfo() error {
	if s.contextCollection == nil {
		return fmt.Errorf("context collection not initialized")
	}

	ctx := context.Background()

	// Store language
	if s.language != "" {
		err := s.contextCollection.AddDocument(ctx, chromem.Document{
			ID:       "language",
			Content:  s.language,
			Metadata: map[string]string{"type": "language"},
		})
		if err != nil {
			return fmt.Errorf("failed to store language: %w", err)
		}
	}

	// Store setting
	if s.setting != "" {
		err := s.contextCollection.AddDocument(ctx, chromem.Document{
			ID:       "setting",
			Content:  s.setting,
			Metadata: map[string]string{"type": "setting"},
		})
		if err != nil {
			return fmt.Errorf("failed to store setting: %w", err)
		}
	}

	// Store characters
	if s.characters != "" {
		err := s.contextCollection.AddDocument(ctx, chromem.Document{
			ID:       "characters",
			Content:  s.characters,
			Metadata: map[string]string{"type": "characters"},
		})
		if err != nil {
			return fmt.Errorf("failed to store characters: %w", err)
		}
	}

	contextCount, charactersCount := s.getMemoryStats()
	log.Printf("Context information stored in memory database (Context: %d, Characters: %d)", contextCount, charactersCount)
	return nil
}

// storeCharacterMessage stores a character message in the characters collection
func (s *ChatSession) storeCharacterMessage(role, content string) error {
	if s.charactersCollection == nil {
		return fmt.Errorf("characters collection not initialized")
	}

	ctx := context.Background()

	// Generate a unique ID for this message
	messageID := fmt.Sprintf("%s_%d", role, len(s.history))

	err := s.charactersCollection.AddDocument(ctx, chromem.Document{
		ID:      messageID,
		Content: content,
		Metadata: map[string]string{
			"role":      role,
			"timestamp": fmt.Sprintf("%d", len(s.history)), // Use history length as simple timestamp
		},
	})
	if err != nil {
		return fmt.Errorf("failed to store character message: %w", err)
	}

	return nil
}

// retrieveContextInfo retrieves context information from the memory database
func (s *ChatSession) retrieveContextInfo() (language, setting, characters string, err error) {
	if s.contextCollection == nil {
		return "", "", "", fmt.Errorf("context collection not initialized")
	}

	ctx := context.Background()

	// Retrieve language
	langResults, err := s.contextCollection.Query(ctx, "language", 1, nil, nil)
	if err == nil && len(langResults) > 0 {
		language = langResults[0].Content
	}

	// Retrieve setting
	settingResults, err := s.contextCollection.Query(ctx, "setting", 1, nil, nil)
	if err == nil && len(settingResults) > 0 {
		setting = settingResults[0].Content
	}

	// Retrieve characters
	charactersResults, err := s.contextCollection.Query(ctx, "characters", 1, nil, nil)
	if err == nil && len(charactersResults) > 0 {
		characters = charactersResults[0].Content
	}

	return language, setting, characters, nil
}

// retrieveCharacterHistory retrieves character messages from the memory database
func (s *ChatSession) retrieveCharacterHistory(limit int) ([]chromem.Result, error) {
	if s.charactersCollection == nil {
		return nil, fmt.Errorf("characters collection not initialized")
	}

	ctx := context.Background()

	// Query for all character messages
	results, err := s.charactersCollection.Query(ctx, "", limit, nil, nil)
	if err != nil {
		return nil, fmt.Errorf("failed to retrieve character history: %w", err)
	}

	return results, nil
}

// clearMemoryDB clears all data from the memory database collections
func (s *ChatSession) clearMemoryDB() error {
	if s.memoryDB == nil {
		return nil // Nothing to clear
	}

	// Delete existing collections
	if s.contextCollection != nil {
		err := s.memoryDB.DeleteCollection("context")
		if err != nil {
			log.Printf("Warning: failed to delete context collection: %v", err)
		}
	}

	if s.charactersCollection != nil {
		err := s.memoryDB.DeleteCollection("characters")
		if err != nil {
			log.Printf("Warning: failed to delete characters collection: %v", err)
		}
	}

	// Reinitialize the collections
	return s.initializeMemoryDB()
}

// getEnhancedContext retrieves context and recent character history for AI response generation
func (s *ChatSession) getEnhancedContext() string {
	var contextBuilder string

	// Retrieve context information from memory database
	language, setting, characters, err := s.retrieveContextInfo()
	if err != nil {
		log.Printf("Failed to retrieve context from memory database: %v", err)
		// Fallback to session variables
		language = s.language
		setting = s.setting
		characters = s.characters
	}

	// Build enhanced context
	if language != "" || setting != "" || characters != "" {
		contextBuilder += "Context from memory:\n"
		if language != "" {
			contextBuilder += fmt.Sprintf("Language: %s\n", language)
		}
		if setting != "" {
			contextBuilder += fmt.Sprintf("Setting: %s\n", setting)
		}
		if characters != "" {
			contextBuilder += fmt.Sprintf("Characters: %s\n", characters)
		}
		contextBuilder += "\n"
	}

	// Retrieve recent character history (last 5 messages)
	recentHistory, err := s.retrieveCharacterHistory(5)
	if err != nil {
		log.Printf("Failed to retrieve character history from memory database: %v", err)
	} else if len(recentHistory) > 0 {
		contextBuilder += "Recent character interactions:\n"
		for _, msg := range recentHistory {
			role := msg.Metadata["role"]
			contextBuilder += fmt.Sprintf("%s: %s\n", role, msg.Content)
		}
		contextBuilder += "\n"
	}

	return contextBuilder
}

// getMemoryStats returns statistics about the memory database collections
func (s *ChatSession) getMemoryStats() (contextCount, charactersCount int) {
	if s.contextCollection == nil || s.charactersCollection == nil {
		return 0, 0
	}

	ctx := context.Background()

	// Count context documents
	contextResults, err := s.contextCollection.Query(ctx, "", 100, nil, nil)
	if err == nil {
		contextCount = len(contextResults)
	}

	// Count character messages
	charactersResults, err := s.charactersCollection.Query(ctx, "", 1000, nil, nil)
	if err == nil {
		charactersCount = len(charactersResults)
	}

	return contextCount, charactersCount
}
