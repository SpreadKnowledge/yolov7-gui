from cx_Freeze import setup, Executable
import sys

sys.setrecursionlimit(5000)  # or a higher number, depending on your needs

# Dependencies are automatically detected, but it might need fine tuning.
build_exe_options = {
    "packages": [
        "os", 
        "torch", 
        "models", 
        "utils", 
        "tkinter", 
        "ttkbootstrap", 
        "PIL", 
        "time", 
        "matplotlib", 
        "matplotlib.backends", 
        "cv2", 
        "numpy", 
        "pathlib", 
        "pandas", 
        "seaborn", 
        "collections", 
        "scipy.spatial.transform"
    ], 
    "includes": ["models.yolo"]
}

# GUI applications require a different base on Windows (the default is for a console application).
base = None

executables = [
    Executable('run_yolov7.py', base=base)
]

setup(name='yolov7_gui',
      version = '0.1.0',
      description = 'YOLOv7 GUI',
      options = {'build_exe': build_exe_options},
      executables = executables)