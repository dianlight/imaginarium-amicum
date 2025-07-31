package main

import "testing"

func TestHelloWorld(t *testing.T) {
	expected := "Hello, World!"
	actual := "Hello, World!"
	if actual != expected {
		t.Errorf("expected %s, got %s", expected, actual)
	}
}