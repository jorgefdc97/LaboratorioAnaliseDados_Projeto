import tkinter as tk
import ttkbootstrap as ttk
import os
from PIL import Image, ImageTk
import numpy as np
import matplotlib.pyplot as plt


SCRIPT_PATH = os.path.dirname(__file__)
APPLICATION_NAME = "Data Analysis Lab"

root = ttk.Window(themename="darkly")
root.title(APPLICATION_NAME)
root.geometry("1280x680")
root.iconbitmap("")

#CREATE FRAME FOR RADIO BUTTONS INSIDE MAIN_FRAME TO STAY FIXED ON TOP
#CREATE FRAME TO DISPLAY INFORMATION FOR EACH RADIO BUTTON
#RECREATE FRAME FOR INFORMATION EVERYTIME


def radio_selection():
    if var.get() == "Timeseries":
        print("Timeseries selected!")
        timeseries_template()
    else:
        print("Algorithm selected!")
        algorithm_template()


def timeseries_template():
    time_basis_template()
    ttk.Label(main_frame, text="Select the size of prediction", padding=20).pack()
    ttk.Spinbox(main_frame, from_=0, to=100).pack()


def algorithm_template():
    time_basis_template()
    ttk.Label(main_frame, text="Select the test size (%)", padding=20).pack()
    ttk.Spinbox(main_frame, from_=0, to=50).pack()


def time_basis_template():
    label_time = ttk.Label(main_frame, text="Select the analysis time basis: ", padding=20)
    label_time.pack()

    n = tk.StringVar()
    time = ttk.Combobox(main_frame, width=27, textvariable=n)

    # Adding combobox drop down list
    time['values'] = (' daily', ' weekly', ' monthly')

    time.pack()
    time.current()


main_frame = tk.Frame(root, width=1280, height=680)
main_frame.configure(padx="100")
main_frame.pack(side="left")

ttk.Label(main_frame,text="Select the algorithm: ", padding=20).pack()
var = tk.StringVar(root, "1")
radio_button1 = ttk.Radiobutton(main_frame, text="Timeseries", variable=var, value=1, command=radio_selection, padding=20)
radio_button1.pack()
radio_button2 =ttk.Radiobutton(main_frame, text="Algoritmo", variable=var, value=2,command=radio_selection)
radio_button2.pack()

root.mainloop()
