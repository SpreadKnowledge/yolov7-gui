# yolov7-gui
It's easy to run YOLOv7 with GUI. Furthermore, it graphs the results of the count of detected objects.
You can already analyze your image data in YOLOv7 without complicated options or additional code.

![yolov7-gui](https://github.com/tmori2918/yolov7-gui/assets/56751392/438090e4-5a2c-4a33-a65e-0be977fcdee4)

Most of the code in this repository is appropriated from [yolov7](https://github.com/WongKinYiu/yolov7).

# Prepare
## Download Repository and Create Virtual Environment
```
git clone https://github.com/SpreadKnowledge/yolov7-gui.git
conda create -n yolov7 python=3.10
conda activate yolov7
cd yolov7-gui
pip install -r requirements.txt
```
## Download yolov7 model

Download the yolov7 pt file of your choice from the "Testing" section of [this website](https://github.com/WongKinYiu/yolov7). 
Then save it to a directory of your choice.
You can also use your original yolov7 model.

## Prepare

Please put together the image data you want YOLOv7 to detect objects into one directory.
Then, copy the data to a location of your choice.
*Note: This GUI application can only read image directories. It cannot object detect videos.

# Run yolov7-gui application
```
python run_yolov7.py
```

# How to operate GUI
