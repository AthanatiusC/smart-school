package camera

import (
	"context"
	"encoding/json"
	"github.com/AthanatiusC/smart-school/facerecog/controller"
	"github.com/AthanatiusC/smart-school/facerecog/controller/face"
	"github.com/AthanatiusC/smart-school/facerecog/model"
	"github.com/hybridgroup/mjpeg"
	"go.mongodb.org/mongo-driver/bson"
	"log"
	"strconv"

	"gocv.io/x/gocv"
	"image"
	"image/color"
	"net/http"
	"os"
)

var (
	err    error
	webcam *gocv.VideoCapture
	stream *mjpeg.Stream
)

func streamCamera(camera model.Camera) {
	img := gocv.NewMat()
	defer img.Close()
	net := gocv.ReadNetFromCaffe("model\\deploy.prototxt", "model\\res10_300x300_ssd_iter_140000.caffemodel")

	for {
		if ok := webcam.Read(&img); !ok {
			log.Printf("Device closed: %v\n", camera.ID)
			return
		}
		if img.Empty() {
			continue
		}
		pt := image.Pt(10, 50)
		gocv.PutText(&img, "Description : "+camera.Description, pt, gocv.FontHersheyPlain, 1.2, color.RGBA{0, 0, 255, 0}, 2)
		img = face.Detect(img, net)
		buf, _ := gocv.IMEncode(".jpg", img)
		stream.UpdateJPEG(buf)
	}
}

func captureCamera(camera model.Camera) model.Camera {
	// webcam, err = gocv.OpenVideoCapture(int(camera.DeviceID))
	webcam, err = gocv.OpenVideoCapture("rtsp://admin:AWPZEO@192.168.1.64/h264_stream")
	if err != nil {
		log.Printf("Device Unavailable: %v\n", camera.ID)
	}
	camera.Camera = webcam
	return camera
}

func InitCamera() {
	host := "localhost:2020"

	cameras := getcamerasdetail()
	// cameras := []model.Camera{}

	for _, camera := range cameras {
		camera = captureCamera(camera)
		defer camera.Camera.Close()
		stream = mjpeg.NewStream()
		go streamCamera(camera)
		log.Println("Camera " + camera.ID.String() + " Started")
		http.Handle("/"+strconv.Itoa(0), stream)
	}
	// start http server
	log.Println("Camera Server : http://localhost:" + host)
	log.Fatal(http.ListenAndServe(host, nil))
	os.Exit(0)
}

func GetAllCameras(res http.ResponseWriter, req *http.Request) {
	switch req.Method {
	case "OPTIONS":
		utils.WriteResult(req, res, nil, "Access Allowed")
		return
	}
	cameras := getcamerasdetail()
	utils.WriteResult(req, res, cameras, "Success")
}

func getcamerasdetail() []model.Camera {
	var camera model.Camera
	var cameras []model.Camera
	db := utils.ConnectMongoDB()
	col, err := db.Collection("camera").Find(context.TODO(), bson.M{})
	utils.ErrorHandler(err)
	for col.Next(context.TODO()) {
		err := col.Decode(&camera)
		utils.ErrorHandler(err)
		cameras = append(cameras, camera)
	}
	return cameras
}

//CreateUsers insert one to DB
func RegisterCamera(res http.ResponseWriter, req *http.Request) {
	switch req.Method {
	case "OPTIONS":
		utils.WriteResult(req, res, nil, "Access Allowed")
		return
	}

	//Declare Variable
	var camera model.Camera
	var camera2 model.Camera

	//Decode Request
	err := json.NewDecoder(req.Body).Decode(&camera)

	//Connect DB
	db := utils.ConnectMongoDB()

	//Loop column
	db.Collection("camera").FindOne(context.TODO(), bson.M{"Address": camera.Address}).Decode(&camera2)
	if len(camera2.Address) != 0 {
		utils.WriteResult(req, res, nil, "Camera Already Exist!")
		return
	}

	_, err = db.Collection("camera").InsertOne(context.TODO(), camera)
	utils.ErrorHandler(err)

	//Return Res
	utils.ErrorHandler(err)
	utils.WriteResult(req, res, nil, "Camera Successfully Created!")
}
