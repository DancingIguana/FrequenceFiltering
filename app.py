import matplotlib.pyplot as plt
import PySimpleGUI as sg
import cv2
import utils
import FT
from utils.array_to_data import array_to_data
import threading
import time
import json
import sys
import itertools

loading = False
filtered_img = None
def apply_filter(height,width, Fuv, graph):
    Huv = FT.get_Huv(centers,(height,width))
    global filtered_img

    filtered_img = FT.frequency_filter(Fuv, Huv)

    graph.erase()
    graph.set_size((width,height))
    graph.change_coordinates((-width/2, -height/2), (width/2, height/2))
    graph.draw_image(data = utils.array_to_data(filtered_img), location = (-width/2,height/2))
    global loading
    loading = False

cv2_image = utils.load_image_gray("opencv.png")
height,width = cv2_image.shape[:2]

layout = [
    [
        sg.Text("Image File"),
        sg.Input(size = (25,1), key = "-FILE-"),
        sg.FileBrowse(),
        sg.Button("Load Image")
    ],
    [
        sg.Text("Filter circles file"),
        sg.Input(size = (25,1), key = "-CIRCLES-"),
        sg.FileBrowse(),
        sg.Button("Load Circles")
    ],
    [
        sg.T("Choose what clicking a figure does", enable_events = True)
    ],
    [
        sg.R("Draw Circles", 1, key = "-CIRCLE-", enable_events = True),
        sg.R("Erase Circle", 1, key = "-ERASE-", enable_events = True)
    ],
    [
        sg.Graph(
            canvas_size = (width,height),
            key = "-GRAPH-",
            graph_bottom_left = (-width/2, -height/2),
            graph_top_right = (width/2, height/2)
        ),
        sg.Graph(
            canvas_size = (width, height),
            key = "-FTGRAPH-",
            enable_events = True,
            drag_submits = True,
            graph_bottom_left = (-width/2, -height/2),
            graph_top_right = (width/2, height/2)
        ),
    ],
    [
        sg.Button("Apply Filter"),
        sg.Button("Save Image"),
        sg.Button("Save circles")
    ]

]


window = sg.Window("Frequence Filtering", layout, finalize = True)

graph = window["-GRAPH-"]
graph.draw_image(data = utils.array_to_data(cv2_image), location = (-width/2,height/2))


Fuv, Fuv_norm = FT.get_Fuv_from_gray_image(cv2_image)

ft_graph = window["-FTGRAPH-"]
ft_graph.draw_image(data = utils.array_to_data(Fuv_norm), location = (-width/2,height/2))
dragging = False
start_point = end_point = prior_rect1 = prior_rect2 = None
filtered_img = None
centers = {}

while True:
    event, values = window.read(timeout = 50)
    if event in (sg.WINDOW_CLOSED, "Exit"):
        break

    if event == "Load Image" and values["-FILE-"]:
        cv2_image = utils.load_image_gray(values["-FILE-"])

        height, width = cv2_image.shape[:2]

        graph.erase()
        graph.set_size((width,height))
        graph.change_coordinates((-width/2, -height/2), (width/2, height/2))
        graph.draw_image(data = utils.array_to_data(cv2_image), location = (-width/2,height/2))

        Fuv, Fuv_norm = FT.get_Fuv_from_gray_image(cv2_image)
        ft_graph.erase()
        ft_graph.set_size((width,height))
        ft_graph.change_coordinates((-width/2, -height/2), (width/2, height/2))
        ft_graph.draw_image(data = utils.array_to_data(Fuv_norm), location = (-width/2,height/2))

        centers = {}

    if event == "Load Circles" and values["-CIRCLES-"]:
        with open(values["-CIRCLES-"],"r") as f:
            centers = json.load(f)
        ft_graph.erase()
        ft_graph.set_size((width,height))
        ft_graph.change_coordinates((-width/2, -height/2), (width/2, height/2))
        ft_graph.draw_image(data = utils.array_to_data(Fuv_norm), location = (-width/2,height/2))
        for circle in centers:
            c = centers[circle]["center"]
            center = (int(c[0]), int(c[1]))
            radius = int(centers[circle]["radius"])
            ft_graph.draw_circle(center, radius, fill_color = "black")
        
    mouse = values["-FTGRAPH-"]
    if event == "-FTGRAPH-" and not loading:
        if mouse == (None,None):
            continue
        
        x,y = mouse
        
        if not dragging:
            start_point = (x,y)
            dragging = True
            drag_figures = list(ft_graph.get_figures_at_location((x,y)))[1:]
            drag_figures += list(ft_graph.get_figures_at_location((-x,-y)))[1:]
            lastxy = x,y
        else:
            end_point = (x,y)
        
        if prior_rect1:
            ft_graph.delete_figure(prior_rect1)
        if prior_rect2:
            ft_graph.delete_figure(prior_rect2)
        
        if None not in (start_point,end_point):
            if values["-CIRCLE-"]:
                prior_rect1 = ft_graph.draw_circle(start_point,end_point[0]-start_point[0],fill_color = "black")
                prior_rect2 = ft_graph.draw_circle((-start_point[0],-start_point[1]),end_point[0]-start_point[0],fill_color = "black")

            if values["-ERASE-"]:
                for i,figure in enumerate(drag_figures):
                    ft_graph.delete_figure(figure)

    elif event.endswith("+UP") and not loading and start_point is not None and end_point is not None:
        centers[prior_rect1] = {
            "center": start_point, 
            "radius": end_point[0] - start_point[0]
        }
        centers[prior_rect2] = {
            "center": (-start_point[0], - start_point[1]), 
            "radius": end_point[0] - start_point[0]
        }
        start_point, end_point = None, None
        dragging = False
        prior_rect1 = prior_rect2 = None
    
    if event == "Apply Filter":
        loading = True
        threading.Thread(target = apply_filter, args = (height,width,Fuv,graph), daemon=True).start()
    
    if event == "Save Image":
        if not loading and filtered_img is not None:
            save_path = sg.popup_get_file("Save",save_as = True,no_window = True) + ".png"
            cv2.imwrite(save_path,filtered_img)

    if event == "Save circles":
        save_path = sg.popup_get_file("Save",save_as = True,no_window = True) + ".json"
        with open(save_path, "w") as f:
            json.dump(centers,f,indent = 6)
window.close()