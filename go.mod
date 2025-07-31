module dialight/imaginarium-amicum

go 1.24.5

require (
	github.com/binozo/gostablediffusion v0.0.0-20250323195332-e7f0f1b22988
	github.com/go-skynet/go-llama.cpp v0.0.0-20240314183750-6a8041ef6b46
	github.com/gorilla/websocket v1.5.3
	github.com/philippgille/chromem-go v0.7.0
	github.com/stretchr/testify v1.10.0
)

require (
	github.com/davecgh/go-spew v1.1.1 // indirect
	github.com/pmezard/go-difflib v1.0.0 // indirect
	golang.org/x/sys v0.16.0 // indirect
	gopkg.in/yaml.v3 v3.0.1 // indirect
)

replace github.com/go-skynet/go-llama.cpp => ./binding/go-llama.cpp

replace github.com/binozo/gostablediffusion => ./binding/gostablediffusion
