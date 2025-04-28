# Computer Vision Projects

This repository contains various computer vision projects developed using OpenCV, PyTorch, YOLO, MediaPipe, and Supervision.

## Projects

### ASL Character Prediction
*   Predicts American Sign Language (ASL) characters from a real-time video stream.
*   Uses OpenCV for video input, MediaPipe for hand tracking, and PyTorch for model training and inference.
*   See the `ASL/` directory and potentially `gesture_recognizer.ipynb`.

### Football Player and Ball Tracking
*   Analyzes the position of the football and players in a video.
*   Draws ellipses over players and tracks their movement.
*   Implemented using YOLO for object detection, OpenCV for video processing, and Supervision for tracking and visualization.
*   See the `Football/` directory.

### Object Detection (Fine-tuned YOLO)
*   Contains a fine-tuned YOLO model and potentially related datasets or scripts.
*   Located in the `Object Detection/` directory.

### Green Light, Red Light Game
*   A game inspired by "Squid Game".
*   Monitors player pose and movement using computer vision.
*   Detects movement during the "Red Light" phase and signals termination.
*   Uses pose estimation techniques (likely MediaPipe or similar).
*   See the `Green Light Game/` directory.

### Face Stylizer
*   Applies artistic styles to faces detected in images or video.
*   See the `face_stylizer.ipynb` notebook.

### Face Detector
*   Detects faces in images or video streams.
*   See the `face_detector.ipynb` notebook.

### Gesture Recognizer
*   Recognizes specific hand gestures from video input.
*   Likely related to the ASL project.
*   See the `gesture_recognizer.ipynb` notebook.

---

*This README provides a brief overview. Please refer to the individual project directories and notebooks for detailed information and code.*
