package utils

import (
	models "github.com/AthanatiusC/godir/models"
	"github.com/tomasen/realip"
	"go.mongodb.org/mongo-driver/bson"
	"go.mongodb.org/mongo-driver/bson/primitive"
	"go.mongodb.org/mongo-driver/mongo"
	"go.mongodb.org/mongo-driver/mongo/options"

	"context"
	"encoding/json"
	"fmt"
	"log"
	"net"
	"net/http"
	"os"
	// "strings"
)

func WriteLog(req *http.Request, message string) {
	// ip, Port := GetIPAdress(req)

	ip := realip.FromRequest(req)

	f, err := os.OpenFile("GoDir.log", os.O_RDWR|os.O_CREATE|os.O_APPEND, 777)
	if err != nil {
		log.Fatalf("error opening file: %v", err)
	}
	defer f.Close()

	log.SetOutput(f)
	log.Println(" [ " + ip + "] " + message)
	// log.Println(" [ " + ip + ":" + Port + " ] " + message)
}

func ConnectMongoDB() *mongo.Database {
	clientOptions := options.Client().ApplyURI("mongodb://localhost:27017")
	client, err := mongo.Connect(context.TODO(), clientOptions)
	iserror := ErrorHandler(err)
	if iserror {
		return nil
	}
	// Check theconnection
	err = client.Ping(context.TODO(), nil)
	iserror = ErrorHandler(err)
	if iserror {
		return nil
	}

	return client.Database("smart-school")
}

func ErrorHandler(err error) bool {
	if err != nil {
		fmt.Println(err)
		return true
	} else {
		return false
	}
}

func IsExists(path string) bool {
	_, err := os.Stat(path)
	ErrorHandler(err)
	if os.IsNotExist(err) {
		return false
	}
	return true
}

func VerifyOwnership(id primitive.ObjectID, auth_key string) bool {
	var model models.Users

	db := ConnectMongoDB()
	db.Collection("users").FindOne(context.TODO(), bson.M{"_id": id}).Decode(&model)

	if auth_key != "" {
		if auth_key != model.Auth {
			log.Println(auth_key + "	" + model.Auth)
			return false
		} else if auth_key == model.Auth {
			return true
		}
	}
	return false
}

type Payload struct {
	Message string      `json:"message"`
	Data    interface{} `json:"returned_data"`
}

func WriteResult(req *http.Request, res http.ResponseWriter, data interface{}, message string) {
	res.Header().Add("Access-Control-Allow-Origin", "*")
	(res).Header().Set("Access-Control-Allow-Headers", "*")
	(res).Header().Set("Access-Control-Allow-Methods", "*")
	res.Header().Set("Content-Type", "Application/JSON")

	var payload Payload
	payload.Message = message
	payload.Data = data
	result, _ := json.Marshal(payload)

	res.WriteHeader(http.StatusAccepted)
	res.Write([]byte(result))
	fmt.Println(message)
	WriteLog(req, message)
}

func GetIPAdress(req *http.Request) (ip string, port string) {
	ip, port, err := net.SplitHostPort(req.RemoteAddr)
	ErrorHandler(err)

	userIP := net.ParseIP(ip)
	if userIP == nil {
		return "Unknown", "0000"
	} else if ip == "::1" {
		ip = "localhost"
	}
	return ip, port
}
