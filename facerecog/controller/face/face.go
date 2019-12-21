package face

import (
	"image"
	"image/color"

	"github.com/AthanatiusC/smart-school/facerecog/model"
	"gocv.io/x/gocv"
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

func Detect(packages model.Payload) model.Face {
	img := packages.Image
	net := packages.CaffeModel
	green := color.RGBA{0, 255, 0, 0}
	for {
		if img.Empty() {
			continue
		}
		W := float32(img.Cols())
		H := float32(img.Rows())

		// convert image Mat to 96x128 blob that the detector can analyze
		blob := gocv.BlobFromImage(img, 1.0, image.Pt(128, 96), gocv.NewScalar(104.0, 177.0, 123.0, 0), false, false)
		defer blob.Close()

		// feed the blob into the classifier
		net.SetInput(blob, "data")

		// run a forward pass through the network
		detBlob := net.Forward("detection_out")
		defer detBlob.Close()
		detections := gocv.GetBlobChannel(detBlob, 0, 0)

		defer detections.Close()

		for r := 0; r < detections.Rows(); r++ {
			// you would want the classid for general object detection,
			// but we do not need it here.
			// classid := detections.GetFloatAt(r, 1)

			confidence := detections.GetFloatAt(r, 2)
			if confidence < 0.5 {
				continue
			}

			left := detections.GetFloatAt(r, 3) * W
			top := detections.GetFloatAt(r, 4) * H
			right := detections.GetFloatAt(r, 5) * W
			bottom := detections.GetFloatAt(r, 6) * H

			// scale to video size:
			left = min(max(0, left), W-1)
			right = min(max(0, right), W-1)
			bottom = min(max(0, bottom), H-1)
			top = min(max(0, top), H-1)

			// draw it
			rect := image.Rect(int(left), int(top), int(right), int(bottom))
			gocv.Rectangle(&img, rect, green, 3)
		}
	}
	var face model.Face
	face.FullImage = img
	return face
}
