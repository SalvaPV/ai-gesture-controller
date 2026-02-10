# Hand Gesture Volume Controller

## Overview
This project implements a real-time computer vision system that controls the system volume based on hand gestures. It utilizes Google's MediaPipe for hand landmark detection and OpenCV for image processing.

The core logic measures the Euclidean distance between the thumb tip and the index finger tip to calculate a dynamic control ratio.

## Technical Implementation
The system captures video input and processes frames to detect hand landmarks. The volume control mechanic works as follows:

1.  **Landmark Detection:** Identifies coordinates for the Thumb Tip (ID 4) and Index Finger Tip (ID 8).
2.  **Geometry:** Calculates the Euclidean distance between these two points.
3.  **Interpolation:** Maps the distance range (measured in pixels) to a volume percentage (0-100%) using linear interpolation.
4.  **Feedback:** Renders a visual volume bar and percentage on the interface in real-time.

## Prerequisites
* Python 3.11 (Recommended for Apple Silicon compatibility)
* Webcam

## Installation

1.  Clone the repository:
    ```bash
    git clone [https://github.com/SalvaPV/ai-gesture-controller.git](https://github.com/SalvaPV/ai-gesture-controller.git)
    cd ai-gesture-controller
    ```

2.  Create a virtual environment:
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

    **Note for macOS (M1/M2/M3) users:**
    If you encounter issues with MediaPipe, ensure you are using Python 3.11 and install the specific compatible version:
    ```bash
    pip install "mediapipe==0.10.14"
    ```

## Usage
Run the main script:
```bash
python main.py