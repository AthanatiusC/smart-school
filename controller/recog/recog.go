package recog

import (
	"encoding/json"
	utils "github.com/AthanatiusC/smart-school/facerecog/controller"
	model "github.com/AthanatiusC/smart-school/facerecog/model"
	"gonum.org/v1/gonum/floats"
	// "github.com/dgrijalva/jwt-go/request"
	"github.com/kshedden/gonpy"
	"log"
	"math"
	"net/http"
)

var (
	UserList []model.Users
)

func main() {
	// r, err := gonpy.NewFileReader("controller\\core\\vector.npy")
	// r2, err := gonpy.NewFileReader("controller\\core\\vector2.npy")
	// utils.ErrorHandler(err)
	// data, err := r.GetFloat64()
	// data2, err := r2.GetFloat64()
	// utils.ErrorHandler(err)
	// log.Println(distance(data2))
}

type Reqq struct {
	Embedding []float64 `json:"Embedding"`
}

func HandleRequest(res http.ResponseWriter, req *http.Request) {
	// utils.HandleCORS(req)
	var request Reqq
	var user model.Users
	var user2 model.Users
	var embedding model.Embeddings
	r, err := gonpy.NewFileReader("controller\\core\\vector.npy")
	utils.ErrorHandler(err)
	r2, err := gonpy.NewFileReader("controller\\core\\vector2.npy")
	utils.ErrorHandler(err)
	data, err := r.GetFloat64()
	utils.ErrorHandler(err)
	data2, err := r2.GetFloat64()
	utils.ErrorHandler(err)
	// user.ID := 1
	user.Name = "Lexi Anugrah"
	embedding.Value = data
	user.Embedding_Data = append(user.Embedding_Data, embedding)

	user2.Name = "Ahmad Saugi"
	embedding.Value = data2
	user2.Embedding_Data = append(user2.Embedding_Data, embedding)

	UserList = append(UserList, user)
	UserList = append(UserList, user2)

	result := recognize(request.Embedding)
	if result.Accuracy < 0.2 {
		utils.WriteResult(req, res, result, "Nearest Face")
	} else {
		utils.WriteResult(req, res, nil, "Unrecognized Face")
	}
}

func init() {
	log.Println("Recognizer Initialized")
}

func recognize(embedding2 []float64) model.Recognition {
	// maximum := 0.5
	var result model.Recognition
	var recognitions []model.Recognition
	// var BestResult []float64
	for _, UserEmbeddingList := range UserList {
		// Index := index
		var val []float64
		for _, UserEmbeddings := range UserEmbeddingList.Embedding_Data {
			val = append(val, distance(UserEmbeddings.Value, embedding2))
		}
		result.Name = UserEmbeddingList.Name
		result.Accuracy = floats.Max(val)
		recognitions = append(recognitions, result)
		// log.Println(maximum)
	}
	var acculist []float64
	for _, value := range recognitions {
		acculist = append(acculist, value.Accuracy)
	}
	// log.Println(recognitions)
	return recognitions[floats.MinIdx(acculist)]
}

func distance(emb1, emb2 []float64) float64 {
	val := 0.0
	for i := range emb1 {
		val += math.Pow(emb1[i]-emb2[i], 2)
	}
	return val
}
