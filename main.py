import cv2
import sys


def main():
    # Load OpenCV's pre-trained face and eye detection models
    face_cascade_path = (
        cv2.data.haarcascades
        + "haarcascade_frontalface_default.xml"
    )

    eye_cascade_path = (
        cv2.data.haarcascades
        + "haarcascade_eye.xml"
    )

    face_cascade = cv2.CascadeClassifier(face_cascade_path)
    eye_cascade = cv2.CascadeClassifier(eye_cascade_path)

    # Check whether the models loaded correctly
    if face_cascade.empty():
        print("Error: Could not load face detection model.")
        sys.exit(1)

    if eye_cascade.empty():
        print("Error: Could not load eye detection model.")
        sys.exit(1)

    # Open the default webcam
    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        print("Error: Could not open webcam.")
        print("Check camera permissions or try another camera index.")
        sys.exit(1)

    print("Face and Eye Detection started!")
    print("Press Q or ESC to exit.")

    try:
        while True:
            # Read a frame from the webcam
            success, frame = camera.read()

            if not success:
                print("Error: Could not read webcam frame.")
                break

            # Mirror the video for a natural webcam experience
            frame = cv2.flip(frame, 1)

            # Convert the frame to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detect faces
            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.2,
                minNeighbors=5,
                minSize=(60, 60)
            )

            # Process each detected face
            for (x, y, w, h) in faces:

                # Draw a green rectangle around the face
                cv2.rectangle(
                    frame,
                    (x, y),
                    (x + w, y + h),
                    (0, 255, 0),
                    2
                )

                # Add a face label
                cv2.putText(
                    frame,
                    "Face",
                    (x, max(y - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )

                # Extract the detected face region
                face_gray = gray[y:y + h, x:x + w]
                face_color = frame[y:y + h, x:x + w]

                # Detect eyes within the face region
                eyes = eye_cascade.detectMultiScale(
                    face_gray,
                    scaleFactor=1.1,
                    minNeighbors=6,
                    minSize=(15, 15)
                )

                # Draw blue rectangles around detected eyes
                for (ex, ey, ew, eh) in eyes:
                    cv2.rectangle(
                        face_color,
                        (ex, ey),
                        (ex + ew, ey + eh),
                        (255, 0, 0),
                        2
                    )

                    cv2.putText(
                        frame,
                        "Eye",
                        (x + ex, max(y + ey - 5, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.45,
                        (255, 0, 0),
                        1
                    )

            # Display detection statistics
            cv2.putText(
                frame,
                f"Faces detected: {len(faces)}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2
            )

            cv2.putText(
                frame,
                "Press Q or ESC to exit",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )

            # Display the live webcam feed
            cv2.imshow("Face and Eye Detection - OpenCV", frame)

            # Exit when Q or ESC is pressed
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q") or key == 27:
                break

    except KeyboardInterrupt:
        print("\nProgram stopped by user.")

    finally:
        # Release webcam and close all OpenCV windows
        camera.release()
        cv2.destroyAllWindows()

        print("Camera closed. Project finished.")


if __name__ == "__main__":
    main()

