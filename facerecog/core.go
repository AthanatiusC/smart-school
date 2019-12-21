package main

import (
	"github.com/AthanatiusC/smart-school/facerecog/controller/camera"
	"github.com/gorilla/mux"
	"log"
	"net/http"
)

func main() {
	go camera.InitCamera()
	router := mux.NewRouter()
	router.Headers("Content-Type", "Application/JSON")

	prefix := router.PathPrefix("/api/v1/").Subrouter()

	prefix.HandleFunc("/camera/register", camera.RegisterCamera).Methods("POST", "OPTIONS")
	prefix.HandleFunc("/camera", camera.GetAllCameras).Methods("GET", "OPTIONS")

	router.Use(mux.CORSMethodMiddleware(router))
	log.Println("Connection Successfull! Api running at http://localhost:9000")
	defer log.Println("Connection Closed")

	log.Fatal(http.ListenAndServe(":9000", router))
}
