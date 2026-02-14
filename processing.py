import time
from threading import Thread

import cv2


def draw_eye_keypoints(img, keypoints, keypoints_conf, confidence_threshold=0.5):
    """Draw only left and right eye keypoints on image and return coordinates"""
    eye_coordinates = []  # Store coordinates for each person

    for person_kpts, person_conf in zip(keypoints, keypoints_conf):
        left_eye = person_kpts[1]
        right_eye = person_kpts[2]
        left_eye_conf = person_conf[1]
        right_eye_conf = person_conf[2]

        # Store coordinates for this person
        person_eyes = {
            'left_eye': None,
            'right_eye': None,
            'left_eye_conf': float(left_eye_conf),
            'right_eye_conf': float(right_eye_conf)
        }

        if left_eye_conf > confidence_threshold:
            left_x, left_y = int(left_eye[0]), int(left_eye[1])
            cv2.circle(img, (left_x, left_y), 5, (0, 255, 0), -1)
            person_eyes['left_eye'] = (left_x, left_y)

            # Draw coordinates as text
            cv2.putText(img, f'L({left_x},{left_y})', (left_x + 10, left_y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

        if right_eye_conf > confidence_threshold:
            right_x, right_y = int(right_eye[0]), int(right_eye[1])
            cv2.circle(img, (right_x, right_y), 5, (255, 0, 0), -1)
            person_eyes['right_eye'] = (right_x, right_y)

            # Draw coordinates as text
            cv2.putText(img, f'R({right_x},{right_y})', (right_x + 10, right_y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)

        eye_coordinates.append(person_eyes)

    return img, eye_coordinates


def detect_camera_properties(camera_id):
    """Detect camera type and optimal settings"""
    print(f"\nDetecting camera {camera_id} properties...")

    # Try different backends
    backends = [
        ('V4L2', cv2.CAP_V4L2),
        ('ANY', cv2.CAP_ANY),
    ]

    for backend_name, backend in backends:
        try:
            cap = cv2.VideoCapture(camera_id, backend)

            if not cap.isOpened():
                continue

            # Try to read a test frame
            ret, frame = cap.read()

            if not ret or frame is None:
                cap.release()
                continue

            # Get camera properties
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)

            # Check if it's grayscale (IR camera)
            is_grayscale = len(frame.shape) == 2 or (len(frame.shape) == 3 and frame.shape[2] == 1)

            print(f"✓ Camera {camera_id} detected ({backend_name}):")
            print(f"  Resolution: {width}x{height}")
            print(f"  FPS: {fps}")
            print(f"  Format: {'Grayscale/IR' if is_grayscale else 'Color'}")
            print(f"  Frame shape: {frame.shape}")

            cap.release()

            return {
                'backend': backend,
                'width': width,
                'height': height,
                'fps': fps,
                'is_grayscale': is_grayscale,
                'working': True
            }

        except Exception as e:
            print(f"  Backend {backend_name} failed: {e}")
            continue

    print(f"✗ Camera {camera_id} not accessible")
    return {'working': False}


class WebcamStream:
    """Threaded webcam capture with IR camera support"""

    def __init__(self, src=0, backend=cv2.CAP_V4L2, target_width=640, target_height=480):
        self.cap = cv2.VideoCapture(src, backend)

        if not self.cap.isOpened():
            raise ValueError(f"Cannot open camera {src}")

        # Try to set properties (may not work on all cameras)

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, target_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, target_height)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        # Verify we can read frames
        self.ret, self.frame = self.cap.read()
        if not self.ret or self.frame is None:
            self.cap.release()
            raise ValueError("Camera opened but cannot read frames")

        # Detect if grayscale/IR camera
        self.is_grayscale = len(self.frame.shape) == 2 or (len(self.frame.shape) == 3 and self.frame.shape[2] == 1)

        if self.is_grayscale:
            print("  IR/Grayscale camera detected - converting to 3-channel")

        self.stopped = False

    def start(self):
        """Start the thread to read frames"""
        Thread(target=self.update, daemon=True).start()
        return self

    def update(self):
        """Keep reading frames in background thread"""
        while not self.stopped:
            if not self.ret:
                self.stop()
            else:
                # Read frames
                self.ret, self.frame = self.cap.read()
                if self.ret:
                    # For IR cameras, convert grayscale to 3-channel
                    if self.is_grayscale and self.frame is not None:
                        if len(self.frame.shape) == 2:
                            self.frame = cv2.cvtColor(self.frame, cv2.COLOR_GRAY2BGR)

                    # Read again to clear buffer
                    temp_ret, temp_frame = self.cap.read()
                    if temp_ret:
                        self.ret, self.frame = temp_ret, temp_frame
                        if self.is_grayscale and self.frame is not None:
                            if len(self.frame.shape) == 2:
                                self.frame = cv2.cvtColor(self.frame, cv2.COLOR_GRAY2BGR)

    def read(self):
        """Return the current frame"""
        return self.frame

    def stop(self):
        """Stop the thread and release camera"""
        self.stopped = True
        if self.cap.isOpened():
            self.cap.release()


def process_webcam_threaded(model, camera_id=0, print_coordinates=True):
    """Optimized threaded webcam processing with IR camera support"""

    try:
        # Detect camera properties first
        cam_props = detect_camera_properties(camera_id)

        if not cam_props['working']:
            raise ValueError(f"Camera {camera_id} not working")

        # Start threaded stream with detected properties
        stream = WebcamStream(
            src=camera_id,
            backend=cam_props['backend'],
            target_width=640,
            target_height=480
        ).start()
        time.sleep(1.0)  # Let camera warm up

        print("✓ Webcam started")
        print("Press 's' to save screenshot")
        if print_coordinates:
            print("Eye coordinates will be printed to console")

        fps_time = time.time()
        fps_counter = 0
        fps = 0
        screenshot_count = 0

        while True:
            frame = stream.read()

            if frame is None:
                print("Warning: No frame received")
                time.sleep(0.1)
                continue

            # Ensure frame is 3-channel BGR
            if len(frame.shape) == 2:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            elif frame.shape[2] == 1:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

            # Run YOLO inference
            results = model(frame, verbose=False, imgsz=416, device='cpu', conf=0.5)

            eye_coordinates_all = []
            for result in results:
                if result.keypoints is not None and len(result.keypoints) > 0:
                    frame, eye_coordinates = draw_eye_keypoints(frame, result.keypoints.xy, result.keypoints.conf)
                    eye_coordinates_all.extend(eye_coordinates)

            # Print coordinates to console
            if print_coordinates and eye_coordinates_all:
                print("\n--- Eye Coordinates ---")
                for idx, person in enumerate(eye_coordinates_all):
                    print(f"Person {idx + 1}:")
                    if person['left_eye']:
                        print(
                            f"  Left Eye:  x={person['left_eye'][0]}, y={person['left_eye'][1]} (conf: {person['left_eye_conf']:.2f})")
                    else:
                        print(f"  Left Eye:  Not detected")

                    if person['right_eye']:
                        print(
                            f"  Right Eye: x={person['right_eye'][0]}, y={person['right_eye'][1]} (conf: {person['right_eye_conf']:.2f})")
                    else:
                        print(f"  Right Eye: Not detected")

            # Calculate FPS
            fps_counter += 1
            if (time.time() - fps_time) > 1:
                fps = fps_counter / (time.time() - fps_time)
                fps_counter = 0
                fps_time = time.time()

            # Display FPS and instructions
            camera_type = "IR/Grayscale" if cam_props['is_grayscale'] else "Color"
            cv2.putText(frame, f'FPS: {fps:.1f} | Camera {camera_id} ({camera_type})', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(frame, 'S: Screenshot', (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # Display coordinate count on frame
            if eye_coordinates_all:
                cv2.putText(frame, f'Detected: {len(eye_coordinates_all)} person(s)', (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

            cv2.imshow('Eye Detection', frame)

            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('s'):
                screenshot_path = f"./screenshots/screenshot_cam{camera_id}_{screenshot_count}.jpg"
                cv2.imwrite(screenshot_path, frame)
                print(f"✓ Screenshot saved: {screenshot_path}")

                # Also save coordinates to text file
                coord_file = f"./coordinates/coordinates_cam{camera_id}_{screenshot_count}.txt"
                with open(coord_file, 'w') as f:
                    f.write("Eye Coordinates\n")
                    f.write("=" * 50 + "\n")
                    for idx, person in enumerate(eye_coordinates_all):
                        f.write(f"Person {idx + 1}:\n")
                        if person['left_eye']:
                            f.write(
                                f"  Left Eye:  x={person['left_eye'][0]}, y={person['left_eye'][1]} (conf: {person['left_eye_conf']:.2f})\n")
                        else:
                            f.write(f"  Left Eye:  Not detected\n")

                        if person['right_eye']:
                            f.write(
                                f"  Right Eye: x={person['right_eye'][0]}, y={person['right_eye'][1]} (conf: {person['right_eye_conf']:.2f})\n")
                        else:
                            f.write(f"  Right Eye: Not detected\n")
                        f.write("\n")
                print(f"✓ Coordinates saved: {coord_file}")

                screenshot_count += 1

        stream.stop()
        cv2.destroyAllWindows()
        print("✓ Stopped")

    except ValueError as e:
        print(f"✗ Error: {e}")
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()


def list_available_cameras():
    """List all available cameras"""
    print("\n" + "=" * 50)
    print("Scanning for available cameras...")
    print("=" * 50)

    available = []

    for i in range(10):
        props = detect_camera_properties(i)
        if props['working']:
            available.append(i)

    print("\n" + "=" * 50)
    if available:
        print(f"✓ Found {len(available)} working camera(s): {available}")
    else:
        print("✗ No working cameras found")
    print("=" * 50 + "\n")

    return available