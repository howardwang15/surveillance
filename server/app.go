package main

import (
	"net/http"

	"github.com/howardwang15/surveillance/server/operations"

	"github.com/gorilla/mux"
)

func main() {
	r := mux.NewRouter()
	r.HandleFunc("/", operations.RenderHomePage)
	http.ListenAndServe(":8080", r)
}
