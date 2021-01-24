package main

import (
	"github.com/gorilla/mux"
	"github.com/howardwang15/surveillance/server/db"
	"github.com/howardwang15/surveillance/server/operations"
	"github.com/joho/godotenv"
	"net/http"
	"os"
)

func main() {
	godotenv.Load("../.env")
	r := mux.NewRouter()
	r.HandleFunc("/", operations.RenderHomePage)
	r.PathPrefix("/assets/").Handler(http.StripPrefix("/assets", http.FileServer(http.Dir("../files"))))
	dsn := os.Getenv("MYSQL_USER") + ":" + os.Getenv("MYSQL_ROOT_PASSWORD") + "@tcp(127.0.0.1:3306)/" + os.Getenv("MYSQL_DB") + "?parseTime=true"
	db.Init(dsn)
	http.ListenAndServe(":8080", r)
}
