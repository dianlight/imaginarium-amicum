module dialight/imaginarium-amicum

go 1.24.5

require (
	github.com/go-skynet/go-llama.cpp v0.0.0-20240314183750-6a8041ef6b46
	github.com/gorilla/websocket v1.5.3
	github.com/seasonjs/stable-diffusion v0.2.0
)

require (
	github.com/ebitengine/purego v0.6.0-alpha.3.0.20240117154336-babd452e909b // indirect
	golang.org/x/sys v0.16.0 // indirect
)

replace github.com/go-skynet/go-llama.cpp => ./binding/go-llama.cpp
