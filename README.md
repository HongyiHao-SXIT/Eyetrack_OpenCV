# FurrySuit Eye Tracking Project

## Project Overview
This project aims to develop an eye tracking system based on OpenMV and OpenCV for use inside the skull of a FurrySuit. The system is capable of real-time identification of the wearer's eye movements and displays the results on a screen outside the FurrySuit, adding a more vivid interactive experience to the FurrySuit.

## Features
- **Real-time Eye Identification**: Utilize an OpenMV camera to capture eye images in real-time inside the FurrySuit.
- **Precise Eye Tracking**: Employ powerful computer vision algorithms of OpenCV to accurately identify and track eye movements.
- **External Visualization**: Display the eye tracking results on an external screen to enhance the visual effect.

## Hardware Requirements
- **OpenMV Camera**: Used to capture eye images inside the FurrySuit skull.
- **Computer (such as Raspberry Pi)**: Runs OpenCV algorithms for image processing and analysis.
- **External Display Screen**: Used to show the eye tracking results.
- **Power Supply**: Provides a stable power source for the OpenMV camera and the computer.
- **Wires and Connectors**: Used to connect various hardware components.

## Software Requirements
- **OpenMV IDE**: Used to write and upload the code for the OpenMV camera.
- **Python 3.x**: The programming language for running OpenCV algorithms.
- **OpenCV Library**: Used for image processing and eye identification.
- **Other Python Libraries**: Such as NumPy, etc., to assist OpenCV in data processing.

## Installation Steps

### OpenMV Camera Setup
1. Download and install the OpenMV IDE.
2. Open the OpenMV IDE and connect the OpenMV camera to the computer.
3. Write and upload the following code to the OpenMV camera in the OpenMV IDE:

```python
import sensor, image, time

# Initialize the camera
sensor.reset()
sensor.set_pixformat(sensor.GRAYSCALE)
sensor.set_framesize(sensor.QVGA)
sensor.skip_frames(time = 2000)
sensor.set_auto_gain(False)
sensor.set_auto_whitebal(False)

clock = time.clock()

while(True):
    clock.tick()
    img = sensor.snapshot()
    # You can add more image processing code here, such as face detection, etc.
    print(clock.fps())
```

### Computer-side Setup
1. Install Python 3.x.
2. Use the following command to install the required Python libraries:
```bash
pip install opencv-python numpy
```
3. Write the Python code on the computer side to receive data from the OpenMV camera and perform eye identification:

```python
import cv2
import numpy as np
import serial

# Open the serial port to connect to the OpenMV camera
ser = serial.Serial('COM3', 115200)  # Modify the serial port and baud rate according to the actual situation

while True:
    if ser.in_waiting:
        # Read the data sent by the OpenMV camera
        data = ser.readline().decode('utf-8').rstrip()
        # You can add more code here for data processing and eye identification
        print(data)

    # Display the result
    cv2.imshow('Eye Tracking', np.zeros((480, 640), dtype=np.uint8))  # Replace with the actual image display
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

ser.close()
cv2.destroyAllWindows()
```

## Usage
1. Install the OpenMV camera inside the FurrySuit skull, ensuring that it can clearly capture eye images.
2. Connect the OpenMV camera to the computer and ensure that the serial communication is normal.
3. Start the Python code on the computer side to start real-time eye identification and tracking.
4. Connect the external display screen to the computer, and you can see the eye tracking results on the screen.

## Notes
- Ensure that the installation position and angle of the OpenMV camera can accurately capture eye images.
- Adjust the parameters of the camera (such as brightness, contrast, etc.) to obtain the best image quality.
- During use, avoid strong vibrations and interference to prevent affecting the accuracy of eye tracking.

## Contribution Guidelines
If you are interested in this project, you can contribute in the following ways:
- Submit bug reports and feature requests.
- Submit code improvements and optimizations.
- Provide documentation and examples.

## License
This project is licensed under the [MIT License](LICENSE). Please refer to the LICENSE file for details.

## Contact Information
If you have any questions or suggestions, please contact Lanyi_adict@outlook.com. 