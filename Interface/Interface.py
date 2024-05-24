import tkinter as tk
import ttkbootstrap as ttk
import os
from PIL import Image, ImageTk
import numpy as np
import matplotlib.pyplot as plt

SCRIPT_PATH = os.path.dirname(__file__)
APPLICATION_NAME = "Data Analysis Lab"


class DataAnalysisLab:
    def __init__(self, root):
        self.WIDTH = 1280
        self.HEIGHT = 680
        self.root = root
        self.root.title(APPLICATION_NAME)
        self.root.geometry(f"{self.WIDTH}x{self.HEIGHT}")
        self.root.iconbitmap("")  # Add path to icon file if available

        self.ML_ALGORITHM = "Algorithm"
        self.ml_algorithm_var = tk.StringVar(root)

        # Variables to store the selected values
        self.time_basis_var = tk.StringVar()
        self.prediction_size_var = tk.IntVar()
        self.test_size_var = tk.IntVar()

        self.create_widgets()

    def create_widgets(self):
        # Frame for radio buttons
        self.radio_frame = ttk.Frame(self.root, width=400, height=100, padding=40)
        self.radio_frame.pack(side="top", anchor="nw")

        ttk.Label(self.radio_frame, text="Select the algorithm:", padding=20).pack()

        self.radio_button1 = ttk.Radiobutton(
            self.radio_frame, text="Timeseries", variable=self.ml_algorithm_var, value="Timeseries",
            command=self.radio_selection, padding=20
        )
        self.radio_button1.pack()

        self.radio_button2 = ttk.Radiobutton(
            self.radio_frame, text=self.ML_ALGORITHM, variable=self.ml_algorithm_var, value=self.ML_ALGORITHM,
            command=self.radio_selection, padding=20
        )
        self.radio_button2.pack()

        # Frame to fill with the algorithm options
        self.main_frame = ttk.Frame(self.root, width=400, height=250, padding=40)
        self.main_frame.pack(side="left", anchor="n")

        # Frame to fill with graph and title
        self.graph_frame = ttk.Frame(self.root, height=self.HEIGHT, padding=40)
        self.graph_frame.pack(side="top")
        self.title_graph_frame = ttk.Frame(self.graph_frame, height=15, padding=40)
        self.title_graph_frame.pack()
        self.graph_prediction_frame = ttk.Frame(self.graph_frame, height=self.HEIGHT)
        self.graph_prediction_frame.pack()

    def radio_selection(self):
        print(self.ml_algorithm_var.get())
        if self.ml_algorithm_var.get() == "Timeseries":
            print("Timeseries selected!")
            self.timeseries_template()
        else:
            print("Algorithm selected!")
            self.algorithm_template()

    def timeseries_template(self):
        self.time_basis_template()
        ttk.Label(self.main_frame, text="Select the size of prediction", padding=20).pack()
        spinbox = ttk.Spinbox(self.main_frame, from_=0, to=100, textvariable=self.prediction_size_var)
        spinbox.pack()
        ttk.Button(self.main_frame, text="OK", command=self.select_options).pack(pady=20)

    def algorithm_template(self):
        self.time_basis_template()
        ttk.Label(self.main_frame, text="Select the test size (%)", padding=20).pack()
        spinbox = ttk.Spinbox(self.main_frame, from_=0, to=50, textvariable=self.test_size_var)
        spinbox.pack()
        ttk.Button(self.main_frame, text="OK", command=self.select_options).pack(pady=20)

    def time_basis_template(self):
        self.clear_main_frame()
        ttk.Label(self.main_frame, text="Select the analysis time basis:", padding=20).pack()

        time_combobox = ttk.Combobox(self.main_frame, width=27, textvariable=self.time_basis_var)
        time_combobox['values'] = ('daily', 'weekly', 'monthly')
        time_combobox.pack()

    def select_options(self):
        print(f"Time Basis: {self.time_basis_var.get()}")
        print(f"Prediction Size: {self.prediction_size_var.get()}")
        print(f"Test Size: {self.test_size_var.get()}")
        self.clear_graph_title()
        if self.time_basis_var.get() != "" and (self.prediction_size_var.get() != 0 or self.test_size_var != 0):
            if self.ml_algorithm_var.get() != self.ML_ALGORITHM:
                ttk.Label(self.title_graph_frame,
                          text=f"Timeseries will be made for {self.prediction_size_var.get()} days "
                               "in a "f"{self.time_basis_var.get()} basis").pack(side="top")

                # -----------CALL SCRIPT TO GENERATE AND SHOW TIMESERIES HERE
                #
                #
                # ------------show inside graph_prediction_frame
                #
                #
                #
            else:
                ttk.Label(self.title_graph_frame, text=f"{self.ML_ALGORITHM} will be made with "
                                                       f"{self.test_size_var.get()}% test"
                                                       " with a "f"{self.time_basis_var.get()} basis").pack(side="top")

                # -----------CALL SCRIPT TO GENERATE AND SHOW PREDICTION ALGORITHM HERE
                #
                #
                # ------------show inside graph_prediction_frame
                #
                #
                #

    def clear_main_frame(self):
        for widget in self.main_frame.winfo_children():
            widget.destroy()

    def clear_graph_title(self):
        for widget in self.title_graph_frame.winfo_children():
            widget.destroy()


if __name__ == "__main__":
    root = ttk.Window(themename="darkly")
    app = DataAnalysisLab(root)
    root.mainloop()
