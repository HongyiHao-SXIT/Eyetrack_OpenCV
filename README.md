# FurrySuit Eye Tracking with OpenCV

## Project Overview
This project implements real-time eye tracking using OpenCV for FurrySuit applications. The system captures eye movements and displays tracking results, enhancing the interactive experience of the FurrySuit.

## Project development status
1. Config the development environment of the project. (Finished)
2. Write the basic version of the code. (Finished)
3. The besst detect code was debugges through experiments. (Finished)
4. Connect the coordingate of the pupil and the image experiment. (Finshed)
5. Find a way to make the images move smooth. (Finished)
6. Continue woth the program debugging. (Under Way)

## Core Features
- Real-time eye detection and tracking
- Pupil position identification
- Eye state monitoring (open/closed)
- Movement detection

## Requirements
- Python 3.6+
- OpenCV (cv2)
- NumPy

## Installation
```bash
pip install opencv-python numpy
```

## Usage Instructions
1. Run the script with Python 3:
```bash
python eye_tracker.py
```
2. Press 'q' to quit the tracking
3. View real-time results in the console and display window

## Key Parameters to Adjust
- `position_change_threshold`: Sensitivity to eye movement
- Threshold value in `cv2.threshold()`: Pupil detection sensitivity
- Frame cropping dimensions: Adjust based on camera position

## Notes
- Ensure proper lighting conditions for best results
- Camera should be positioned to clearly capture eyes
- For FurrySuit integration, consider mounting the camera inside the headpiece

This implementation provides a complete, self-contained OpenCV solution for FurrySuit eye tracking without external dependencies beyond OpenCV and NumPy.