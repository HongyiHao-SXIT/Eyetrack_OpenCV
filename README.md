# FurrySuit Eye Tracking with YOLOv11

## Features
- **Real-time eye movement tracking**: Utilize a camera to capture eye images in real-time and accurately track the position of the pupil.
- **Error handling and monitoring**: Equipped with error monitoring and automatic restart functions to ensure the stable operation of the system.
- **Object detection based on YOLOv11**: Employ the YOLOv11 model to quickly and accurately detect eye images.
- **Pupil coordinate tracking**: Calculate the relative position of the pupil center based on the output of the detection box.
- **Dynamic graphic control**: Drive the pupil animation of the furry suit through Pygame or other image rendering methods, including tracking and blinking effects.
- **Adaptation to embedded platforms**: Optimize the model inference performance to adapt to the operating environment of the Walnut Pi.
- **HDMI output display**: Display the detection results and animations in real-time on an externally connected HDMI screen.

## Environment Requirements
- **Python environment**: Python 3.8 or higher
- **Dependency libraries**:
  - OpenCV (cv2)
  - NumPy
  - Pygame
  - Ultralytics

## Usage Instructions
### Running the Python Program
Make sure you have installed the required Python dependency libraries, and then run the corresponding Python script. The following are the detailed steps:

#### 1. Install Dependencies
Ensure you are using a Python 3.8+ environment and install the following dependencies:
```bash
pip install opencv-python pygame ultralytics
```
> ⚠️ For devices like the Walnut Pi, it is recommended to use the lightweight YOLOv11-nano model version provided by Ultralytics to improve real-time performance.

#### 2. Run the Main Program
```bash
python Eyetrack_Fursuit/YOLOv11/main.py
```
The system will automatically open the camera, load the model for detection, and transmit the pupil coordinates to the Pygame animation logic.

#### 3. Hardware Connection Instructions
* **Walnut Pi**: Run the main program of this project and perform model inference.
* **Camera**: Use a USB or CSI interface to capture eye images.
* **HDMI display**: Display the pupil and blinking images through the Pygame animation.

## Project Directory Structure
```
Eyetrack_Fursuit/
├── YOLOv11/
│   ├── main.py                # Main program entry: Model loading and camera detection
│   ├── model/last.pt          # Trained YOLOv11 model
│   ├── pupil_anim.py          # Pygame animation logic control (pupil, blinking)
│   ├── assets/                # Animation image resources (eyeball images, blinking frames, etc.)
│   ├── utils.py               # Auxiliary functions such as coordinate conversion
│   └── README.md              # This instruction file
└── README.md                  # Overall project instruction file
```

## Model Training Instructions
The configuration of the dataset used during training is as follows:
```yaml
# Example of dataset.yaml
train:  /Dataset/images/train
val:    /Dataset/images/val
test:   /Dataset/images/test

nc: 2
names: ['eyeball', 'pupil']
```
Model training command (Ultralytics v8.x):
```bash
yolo task=detect mode=train model=yolov11n.pt data=dataset.yaml epochs=200 imgsz=640
```
The path of the training output file is:
```
Eyetrack_Fursuit/YOLOv11/runs/train/train-200epoch-v11n.yaml/weights/last.pt
```

## Project Developers
### Lanyi_adict
* Major: Computer Science
* Skills: Deep learning, database development, graphics system integration
* Responsibilities: Training and optimization of the YOLOv11 model, system architecture design, and implementation of the camera interface

### Joyang
* Major: Mechatronic Engineering
* Skills: JavaEE, embedded image processing, AI system integration
* Responsibilities: Platform deployment guidance, experiment design, overall project scheduling, and quality review

## Future Work
* ✅ Multi-threaded acceleration of camera frame reading and model inference
* ✅ Dynamic simulation of blinking frequency
* 🚧 Multi-eye recognition and linkage control
* 🚧 Support for controlling the eye movements of external mechanical structures via MQTT/serial port