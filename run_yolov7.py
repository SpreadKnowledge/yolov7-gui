from tkinter import ttk
from ttkbootstrap import Style
from tkinter import filedialog, messagebox, Label, Canvas
import threading
from PIL import Image, ImageTk

import time
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import os
import cv2
import torch
from numpy import random
from models.experimental import attempt_load
from utils.datasets import LoadImages
from utils.general import (
    check_img_size, non_max_suppression, apply_classifier, scale_coords,
    xyxy2xywh, set_logging)
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized
from pathlib import Path
import pandas as pd
import seaborn as sns
from collections import defaultdict

import numpy as np

# Define global variables
input_folder = ''
output_folder = ''
yolov7_weights = ''
btn_font = ("Segoe UI", 20)
lbl_font = ("Segoe UI", 14)
background_color = '#ADD8E6'

# Set default window size
WINDOW_WIDTH = 1600
WINDOW_HEIGHT = 900

class ImageFrame:
    def __init__(self, parent, output_folder):
        self.parent = parent
        self.frame = ttk.Frame(parent, width=WINDOW_WIDTH//2, height=WINDOW_HEIGHT)
        self.frame.pack(side='right', fill='both', expand=True)
        self.output_folder = output_folder
        self.output_files = [file for file in os.listdir(output_folder) if file.endswith(('.jpg', '.png')) and file != 'count_result.png']
        self.index = 0
        self.canvas = Canvas(self.frame, width=(WINDOW_WIDTH//2-200), height=(WINDOW_HEIGHT-200))
        self.canvas.pack(side='left', fill='both', expand=True)
        self.prev_button = ttk.Button(self.frame, text="< Prev", command=self.prev_image)
        self.prev_button.pack(side='left')
        self.next_button = ttk.Button(self.frame, text="Next >", command=self.next_image)
        self.next_button.pack(side='left')
        self.show_image()

    def show_image(self):
        self.canvas.delete('all')
        img_path = os.path.join(self.output_folder, self.output_files[self.index])
        img = Image.open(img_path)
        img.thumbnail(((WINDOW_WIDTH//2-200), (WINDOW_HEIGHT-200)), Image.LANCZOS)
        self.canvas.image = ImageTk.PhotoImage(img)
        self.canvas.create_image(0, 0, anchor='nw', image=self.canvas.image)
        self.canvas.config(scrollregion=self.canvas.bbox('all'))

    def prev_image(self):
        if self.index > 0:
            self.index -= 1
            self.show_image()

    def next_image(self):
        if self.index < len(self.output_files) - 1:
            self.index += 1
            self.show_image()

def select_input_folder():
    global input_folder
    input_folder = filedialog.askdirectory(title="Select input folder")
    lbl_input_folder["text"] = f"Input Folder: {input_folder}"

def select_output_folder():
    global output_folder
    output_folder = filedialog.askdirectory(title="Select output folder")
    lbl_output_folder["text"] = f"Output Folder: {output_folder}"

def select_weights():
    global yolov7_weights
    yolov7_weights = filedialog.askopenfilename(title="Select YOLOv7 weights file", filetypes=[("YOLOv7 weights", "*.pt")])
    lbl_yolov7_weights["text"] = f"YOLOv7 Weights: {yolov7_weights}"

def detect(input_folder, output_folder, weights='yolov7.pt', imgsz=640, conf_thres=0.5, iou_thres=0.5):
    # Initialize
    set_logging()
    # GPU or cpu
    device = select_device('')
    #device = select_device('cpu')
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size
    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    dataset = LoadImages(input_folder, img_size=imgsz)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    t0 = time.time()
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once

    total_images = len(os.listdir(input_folder))
    current_image = 0

    for path, img, im0s, vid_cap in dataset:
        current_image += 1
        progress['value'] = (current_image / total_images) * 100
        window.update_idletasks()

        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=False)[0]

        # Apply NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes=None, agnostic=False)
        t2 = time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            p, s, im0 = path, '', im0s

            save_path = str(Path(output_folder) / Path(p).name)
            txt_path = str(Path(output_folder) / Path(p).stem) + ('_%g' % dataset.frame if dataset.mode == 'video' else '')
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += '%g %ss, ' % (n, names[int(c)])  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if True:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if False else (cls, *xywh)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if True:  # Add bbox to image
                        label = '%s %.2f' % (names[int(cls)], conf)
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=5)

            cv2.imwrite(save_path, im0)

    print('Done. (%.3fs)' % (time.time() - t0))

def count_and_plot(weights, output_folder):
    device = select_device('')
    model = attempt_load(weights, map_location=device)  # load FP32 model
    names = model.module.names if hasattr(model, 'module') else model.names
    class_ids = range(len(names))

    class_dict = {k:v for k,v in enumerate(names)}
    counts = defaultdict(int)

    for file in os.listdir(output_folder):
        if file.endswith('.txt'):
            with open(os.path.join(output_folder, file)) as f:
                for line in f:
                    class_id = int(line.split()[0])
                    counts[class_id] += 1

    class_counts = [counts[i] for i in class_ids]
    class_names = [class_dict[i] for i in class_ids]

    df = pd.DataFrame({'class': class_names, 'count': class_counts})
    df = df[df['count'] > 0]  # Exclude classes with zero counts
    plt.figure(figsize=(16, 11))
    sns.barplot(x='class', y='count', data=df, orient='v')
    plt.xlabel('Class Name', fontsize=24)
    plt.ylabel('Count Number', fontsize=24)
    plt.tick_params(axis='x', labelsize=26)
    plt.tick_params(axis='y', labelsize=26)
    plt.yticks(np.arange(min(class_counts), max(class_counts)+1, 1.0))  # add this line
    plt.grid(False)
    for i in range(len(df['count'])):
        plt.text(i, df['count'].iloc[i], df['count'].iloc[i], ha = 'center', fontsize=36)
    plt.savefig(os.path.join(output_folder, 'count_result.png'), bbox_inches='tight')

    # Save counts including zero counts to csv
    df_all = pd.DataFrame({'class': class_names, 'count': class_counts})
    df_all.to_csv(os.path.join(output_folder, 'count_result.csv'), index=False)
"""
def run_detection():
    global input_folder, output_folder, yolov7_weights
    if input_folder == "" or output_folder == "" or yolov7_weights == "":
        messagebox.showwarning("Warning", "Please select input folder, output folder and YOLOv7 weights.")
        return

    lbl_process["text"] = "Please wait a moment..."
    window.update()
    detect(input_folder, output_folder, yolov7_weights)
    count_and_plot(yolov7_weights, output_folder)
    lbl_process["text"] = ""
    show_detected_images()
"""
def run_detection():
    global input_folder, output_folder, yolov7_weights
    if input_folder == "" or output_folder == "" or yolov7_weights == "":
        messagebox.showwarning("Warning", "Please select input folder, output folder and YOLOv7 weights.")
        return

    lbl_process["text"] = "Please wait a moment..."
    window.update()
    
    # Create a new thread for the detection and plot functions
    detection_thread = threading.Thread(target=detect_and_plot, args=(input_folder, output_folder, yolov7_weights), daemon=True)
    detection_thread.start()

def detect_and_plot(input_folder, output_folder, yolov7_weights):
    detect(input_folder, output_folder, yolov7_weights)
    count_and_plot(yolov7_weights, output_folder)
    lbl_process["text"] = ""
    show_detected_images()

def show_detected_images():
    global output_folder
    image_files = [file for file in os.listdir(output_folder) if file.endswith(('.jpg', '.png')) and file != 'count_result.png']
    img_frame = ImageFrame(window, output_folder)
    img_path = os.path.join(output_folder, 'count_result.png')
    img = Image.open(img_path)
    img.thumbnail(((WINDOW_WIDTH//2-200), (WINDOW_HEIGHT-200)), Image.LANCZOS)
    tk_image = ImageTk.PhotoImage(img)
    lbl_graph = Label(window, image=tk_image)
    lbl_graph.image = tk_image  # keep a reference
    lbl_graph.pack(side='left', fill='both', expand=True)

if __name__ == '__main__':
    style = Style(theme='united')  # Use the united(blue&orange) theme
    style = ttk.Style()
    style.configure("green.Horizontal.TProgressbar", foreground='green', background='green')

    window = style.master
    window.title("YOLOv7 Object Detection App")
    window.geometry(f"{WINDOW_WIDTH}x{WINDOW_HEIGHT}")
    window.configure(bg=background_color)
    window.resizable(True, True)  # Make the window resizable

    s = ttk.Style()
    s.configure('TButton', font=btn_font)

    btn_input_folder = ttk.Button(window, text="Select Input Folder", command=select_input_folder, style="TButton", width=20, padding=7)
    btn_input_folder.pack()

    lbl_input_folder = ttk.Label(window, text="Input Folder: Not selected", font=lbl_font, background=background_color)
    lbl_input_folder.pack()

    btn_weights = ttk.Button(window, text="Select YOLOv7 Weights", command=select_weights, style="TButton", width=20, padding=7)
    btn_weights.pack()

    lbl_yolov7_weights = ttk.Label(window, text="YOLOv7 Weights: Not selected", font=lbl_font, background=background_color)
    lbl_yolov7_weights.pack()

    btn_output_folder = ttk.Button(window, text="Select Output Folder", command=select_output_folder, style="TButton", width=20, padding=7)
    btn_output_folder.pack()

    lbl_output_folder = ttk.Label(window, text="Output Folder: Not selected", font=lbl_font, background=background_color)
    lbl_output_folder.pack()

    btn_detect = ttk.Button(window, text="Run Detection", command=run_detection, style="TButton", width=25, padding=10)
    btn_detect.pack()

    lbl_process = ttk.Label(window, text="", font=lbl_font, background=background_color)
    lbl_process.pack()

    progress = ttk.Progressbar(window, orient='horizontal', length=400, mode='determinate')
    progress.pack()

    window.mainloop()