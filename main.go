package main

import (
	"fmt"
	"log"
	"sync"

	"gocv.io/x/gocv"
)

// ID for IR cameras
const leftEyeDeviceID = 0
const rightEyeDeviceID = 1

func processEyeStream(camID int, name string, wg *sync.WaitGroup) {
	defer wg.Done()

	log.Printf("Starting capture for %s (Device ID: %d)", name, camID)

	webcam, err := gocv.OpenVideoCapture(camID)
	if err != nil {
		log.Printf("Error opening video capture for %s (device %d): %v", name, camID, err)
		return
	}
	defer func(webcam *gocv.VideoCapture) {
		err := webcam.Close()
		if err != nil {
			log.Printf("Error closing webcam: %v", err)
		}
	}(webcam)

	windowName := fmt.Sprintf("%s Stream", name)
	window := gocv.NewWindow(windowName)
	defer func(window *gocv.Window) {
		err := window.Close()
		if err != nil {
			log.Printf("Error closing window: %v", err)
		}
	}(window)

	img := gocv.NewMat()
	defer func(img *gocv.Mat) {
		err := img.Close()
		if err != nil {
			log.Printf("Error closing image: %v", err)
		}
	}(&img)

	for {
		if ok := webcam.Read(&img); !ok || img.Empty() {
			log.Printf("Could not read frame from %s. Exiting loop.", name)
			break
		}

		err := window.IMShow(img)
		if err != nil {
			return
		}

		if window.WaitKey(1) == 27 {
			log.Printf("%s stream received exit signal.", name)
			break
		}
	}
	log.Printf("%s stream processing finished.", name)
}

func main() {
	var wg sync.WaitGroup

	wg.Add(1)
	go processEyeStream(leftEyeDeviceID, "Left Eye", &wg)

	wg.Add(1)
	go processEyeStream(rightEyeDeviceID, "Right Eye", &wg)

	wg.Wait()
	fmt.Println("All eye camera streams closed.")
}
