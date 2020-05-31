package model

import (
	"go.mongodb.org/mongo-driver/bson/primitive"
	"gocv.io/x/gocv"
)

//Face Struct used for passing faces
type Face struct {
	ID        int
	Name      string
	FullImage gocv.Mat
	FaceImage gocv.Mat
}

//Payload struct used to move around arguments
type Payload struct {
	CaffeModelPath  string
	CaffeConfigPath string
	CaffeNet        gocv.Net
	Image           gocv.Mat
	Result          gocv.Mat
	Name            string
}

//Camera struct used to identify Camera Initialization
type Camera struct {
	ID          primitive.ObjectID `json:"id" bson:"_id,omitempty"`
	Description string
	DeviceID    int
	DeviceURL   string
	Camera      *gocv.VideoCapture
	Address     string
}

type Cameras struct {
	Cameras []Camera
}

type Embeddings struct {
	Value []float64
}

type Users struct {
	ID             primitive.ObjectID `json:"id" bson:"_id,omitempty"`
	Name           string
	Embedding_Data []Embeddings
}

type Recognition struct {
	ID       primitive.ObjectID `json:"id" bson:"_id,omitempty"`
	Name     string
	Accuracy float64
}
 