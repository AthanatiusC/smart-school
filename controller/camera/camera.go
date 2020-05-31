package camera

import (
	"context"
	"encoding/json"
	"github.com/AthanatiusC/smart-school/facerecog/controller"
	// "github.com/AthanatiusC/smart-school/facerecog/controller/face"
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
	"time"
)

var (
	err        error
	webcam     *gocv.VideoCapture
	stream     *mjpeg.Stream
	facestream *mjpeg.Stream
	left       float32
	top        float32
	right      float32
	bottom     float32
	buf        []byte
)

func min(a, b float32) float32 {
	if a < b {
		return a
	}
	return b
}

func max(a, b float32) float32 {
	if a > b {
		return a
	}
	return b
}

func streamCamera(camera model.Camera) {
	img := gocv.NewMat()
	face := gocv.NewMat()
	defer img.Close()
	defer face.Close()
	net := gocv.ReadNetFromCaffe("model\\deploy.prototxt", "model\\recognizer.caffemodel")

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

		green := color.RGBA{0, 255, 0, 0}
		if img.Empty() {
			return
		}
		W := float32(img.Cols())
		H := float32(img.Rows())

		// convert image Mat to 96x128 blob that the detector can analyze
		// blob := gocv.BlobFromImage(img, 1.0, image.Pt(128, 96), gocv.NewScalar(104.0, 177.0, 123.0, 0), false, false)
		blob := gocv.BlobFromImage(img, 1.0, image.Pt(300, 300), gocv.NewScalar(104.0, 177.0, 123.0, 0), false, false)
		defer blob.Close()

		// feed the blob into the classifier
		net.SetInput(blob, "data")

		// run a forward pass through the network
		detBlob := net.Forward("detection_out")
		defer detBlob.Close()
		detections := gocv.GetBlobChannel(detBlob, 0, 0)

		defer detections.Close()
		face = gocv.NewMat()
		for r := 0; r < detections.Rows(); r++ {
			// you would want the classid for general object detection,
			// but we do not need it here.
			// classid := detections.GetFloatAt(r, 1)

			confidence := detections.GetFloatAt(r, 2)
			if confidence < 0.5 {
				continue
			}

			left = detections.GetFloatAt(r, 3) * W
			top = detections.GetFloatAt(r, 4) * H
			right = detections.GetFloatAt(r, 5) * W
			bottom = detections.GetFloatAt(r, 6) * H
			gocv.Circle(&img, image.Pt(int(right), int(top)), 5, green, 5)
			gocv.Circle(&img, image.Pt(int(left), int(top)), 5, green, 5)
			gocv.Circle(&img, image.Pt(int(right), int(bottom)), 5, green, 5)
			gocv.Circle(&img, image.Pt(int(left), int(bottom)), 5, green, 5)
			gocv.Circle(&img, image.Pt(int(left), int(bottom)), 5, green, 5)
			gocv.Circle(&img, image.Pt((int(left)+int(right))/2, (int(bottom)+int(top))/2), 5, green, 5)

			// scale to video size:
			left = min(max(0, left), W-1)
			right = min(max(0, right), W-1)
			bottom = min(max(0, bottom), H-1)
			top = min(max(0, top), H-1)

			// draw it
			// rect := image.Rect(int(left), int(top), int(right), int(bottom))
			// face = img.Region(rect)
			// face.Close()

			// gocv.Circle(&img, top, 5, color.RGBA{0, 0, 255, 0}, 2)
			// gocv.Rectangle(&img, rect, green, 3)
		}

		buf, _ = gocv.IMEncode(".jpg", img)
		stream.UpdateJPEG(buf)
	}
}

func captureCamera(camera model.Camera) model.Camera {
	webcam, err = gocv.OpenVideoCapture(int(camera.DeviceID))
	// webcam, err = gocv.OpenVideoCapture("rtsp://admin:AWPZEO@192.168.1.64/h264_stream")
	if err != nil {
		log.Printf("Device Unavailable: %v\n", camera.ID)
	}
	camera.Camera = webcam
	return camera
}

func InitCamera() {
	host := ":2020"

	for i := 0; i < 3; i++ {
		log.Println("Benchmaring...")
		start := time.Now()
		getcamerasdetail()
		log.Println("Fetching result : " + time.Since(start).String())
	}
	cameras := getcamerasdetail()

	// cameras := []model.Camera{}

	for _, camera := range cameras {
		camera = captureCamera(camera)
		defer camera.Camera.Close()
		stream = mjpeg.NewStream()
		facestream = mjpeg.NewStream()
		go streamCamera(camera)
		log.Println("Camera " + camera.ID.String() + " Started")
		http.Handle("/"+strconv.Itoa(camera.DeviceID), stream)
		http.Handle("/"+strconv.Itoa(camera.DeviceID)+"/face", stream)
	}
	// start http server
	log.Println("Camera Server : http://" + host)
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
