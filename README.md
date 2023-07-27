# Application for easy object detection with YOLOv7 via GUI
It's easy to run YOLOv7 with GUI. Furthermore, it graphs the results of the count of detected objects.
You can already analyze your image data in YOLOv7 without complicated options or additional code.

![yolov7-gui](https://github.com/tmori2918/yolov7-gui/assets/56751392/438090e4-5a2c-4a33-a65e-0be977fcdee4)

Most of the code in this repository is appropriated from [yolov7](https://github.com/WongKinYiu/yolov7).

# Prepare
## Download repository and create virtual environment
```
conda create -n yolov7 python=3.10
conda activate yolov7

git clone https://github.com/SpreadKnowledge/yolov7-gui.git
cd yolov7-gui
pip install -r requirements.txt
```
## Download yolov7 model

Download the yolov7 pt file of your choice from the "Testing" section of [this website](https://github.com/WongKinYiu/yolov7). 
Then save it to a directory of your choice.
You can also use your original yolov7 model.

## Prepare your images

Please put together the image data you want YOLOv7 to detect objects into one directory.
Then, copy the data to a location of your choice.
*Note: This GUI application can only read image directories. It cannot object detect videos.

# Run yolov7-gui application
```
python run_yolov7.py
```

# How to operate GUI

1. Press "Select Input Folder": The Explorer will start up and you select the image directory you have prepared.
2. Press "Select YOLOv7 Weights": Select the YOLOv7 model used for object detection from the Explorer.
3. Press "Select Output Folder": Select a directory to output object detected images, annotation files (.txt), graphs summarizing object detection results, and csv files. If you have not already created a directory, you may do so within Explorer.
4. Press "Run Detection":　Object detection begins. At the same time, the progress bar starts moving to the right. When the object detection is finished, the GUI displays a bar graph summarizing the count results and an object detection images.
5. Once object detection is complete, the Input Folder or YOLOv7 model can be changed again to detect the object.
6. Finish object detection: Click the × button in the upper right corner of the GUI screen to exit the application.

*Note: When displaying the second object detection result in the GUI, the screen is buggy. The object detection results are output to the Output Folder without any problem. We will fix it in the future.
