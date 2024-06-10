import tkinter as tk
import ttkbootstrap as ttk
import os
from PIL import Image, ImageTk
import numpy as np
import matplotlib.pyplot as plt
import data as data_module

SCRIPT_PATH = os.path.dirname(__file__)
APPLICATION_NAME = "Data Analysis Lab"
DAILY_CSV_PATH = "../GOOG.US_D1_cleaned.csv"
WEEKLY_CSV_PATH = "../GOOG.US_D1_cleaned.csv" #ALTERAR PARA PATH CSV WEEKLY
MONTHLY_CSV_PATH = "../GOOG.US_D1_cleaned.csv" #ALTERAR PARA PATH CSV MONTHLY

class DataAnalysisLab:
    def __init__(self, root):
        self.WIDTH = 1500
        self.HEIGHT = 780
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
        self.prediction_var = tk.StringVar()

        self.create_widgets()

    def create_widgets(self):
        self.notebook = ttk.Notebook(self.root, height=self.HEIGHT, width=self.WIDTH)

        self.tab1 = ttk.Frame(self.notebook)
        self.main_frame = ttk.Frame(self.tab1)
        self.main_frame.pack(side="left")

        # Frame for radio buttons
        self.algorithm_frame = ttk.Frame(self.main_frame)
        self.algorithm_frame.pack(anchor="n")

        ttk.Label(self.algorithm_frame, text="Select the algorithm:", padding=20).pack()

        self.radio_button1 = ttk.Radiobutton(
            self.algorithm_frame, text="Timeseries", variable=self.ml_algorithm_var, value="Timeseries",
            command=self.radio_selection, padding=20
        )
        self.radio_button1.pack()

        self.radio_button2 = ttk.Radiobutton(
            self.algorithm_frame, text=self.ML_ALGORITHM, variable=self.ml_algorithm_var, value=self.ML_ALGORITHM,
            command=self.radio_selection, padding=20
        )
        self.radio_button2.pack()

        # Frame to fill with the algorithm options
        self.optional_frame = ttk.Frame(self.main_frame, padding=40)
        self.optional_frame.pack()

        # Frame to fill with graph and title
        self.graph_frame = ttk.Frame(self.tab1, padding=60)
        self.graph_frame.pack(side="left")
        self.title_graph_frame = ttk.Frame(self.graph_frame, height=15, padding=40)
        self.title_graph_frame.pack()
        self.graph_prediction_frame = ttk.Frame(self.graph_frame, height=self.HEIGHT)
        self.graph_prediction_frame.pack()

        self.tab2 = ttk.Frame(self.notebook)
        self.tab3 = ttk.Frame(self.notebook)
        self.tab4 = ttk.Frame(self.notebook)

        self.notebook.add(self.tab1, text='Prediction')
        self.notebook.add(self.tab2, text='Algorithm 1')
        self.notebook.add(self.tab3, text='Algorithm 2')
        self.notebook.add(self.tab4, text='Algorithm 3')
        self.notebook.pack()

    def radio_selection(self):
        print(self.ml_algorithm_var.get())
        if self.ml_algorithm_var.get() == "Timeseries":
            print("Timeseries selected!")
            self.timeseries_template()
        else:
            print("Algorithm selected!")
            self.algorithm_template()

    def prediction_object_selection(self):
        ttk.Label(self.optional_frame, text="Select the prediction's object:", padding=20).pack()
        self.prediction_combobox = ttk.Combobox(self.optional_frame, width=27, textvariable=self.prediction_var)
        self.prediction_combobox['values'] = ('open', 'close', 'high', 'low')
        self.prediction_combobox.pack()

    def timeseries_template(self):
        self.time_basis_template()
        ttk.Label(self.optional_frame, text="Select the size of prediction (days)", padding=20).pack()
        spinbox = ttk.Spinbox(self.optional_frame, from_=0, to=100, textvariable=self.prediction_size_var)
        spinbox.pack()
        ttk.Button(self.optional_frame, text="OK", command=self.select_options).pack(pady=20)

    def algorithm_template(self):
        self.time_basis_template()
        self.prediction_object_selection()
        ttk.Label(self.optional_frame, text="Select the test size (%)", padding=20).pack()
        spinbox = ttk.Spinbox(self.optional_frame, from_=0, to=50, textvariable=self.test_size_var)
        spinbox.pack()
        ttk.Button(self.optional_frame, text="OK", command=self.select_options).pack(pady=20)

    def time_basis_template(self):
        self.clear_main_frame()
        ttk.Label(self.optional_frame, text="Select the analysis time basis", padding=20).pack()

        time_combobox = ttk.Combobox(self.optional_frame, width=27, textvariable=self.time_basis_var)
        time_combobox['values'] = ('daily', 'weekly', 'monthly')
        time_combobox.pack()

    def select_options(self):
        print(f"Object: {self.prediction_var.get()}")
        print(f"Time Basis: {self.time_basis_var.get()}")
        print(f"Prediction Size: {self.prediction_size_var.get()}")
        print(f"Test Size: {self.test_size_var.get()}")
        self.clear_graph_title()
        if self.time_basis_var.get() != "" and (self.prediction_size_var.get() != 0 or self.test_size_var != 0):
            if self.time_basis_var.get() == "daily":
                file_path = DAILY_CSV_PATH
            elif self.time_basis_var.get() == "weekly":
                file_path = WEEKLY_CSV_PATH
            else:
                file_path = MONTHLY_CSV_PATH

            df_all = data_module.read_and_preprocess(file_path)

            if self.ml_algorithm_var.get() != self.ML_ALGORITHM:
                ttk.Label(self.title_graph_frame,
                          text=f"Timeseries will be made for {self.prediction_size_var.get()} days "
                               "in a "f"{self.time_basis_var.get()} basis").pack(side="top")
            else:
                ttk.Label(self.title_graph_frame, text=f"{self.ML_ALGORITHM} will be made with "
                                                       f"{self.test_size_var.get()}% test"
                                                       " with a "f"{self.time_basis_var.get()} basis").pack(side="top")

                test_size = self.test_size_var.get() / 100
                data_module.mlp_regressor(df_all, self.prediction_var.get(), test_size)

    def clear_main_frame(self):
        for widget in self.optional_frame.winfo_children():
            widget.destroy()

    def clear_graph_title(self):
        for widget in self.title_graph_frame.winfo_children():
            widget.destroy()


if __name__ == "__main__":
    root = ttk.Window(themename="darkly")
    app = DataAnalysisLab(root)
    root.mainloop()
