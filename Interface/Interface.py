import time
import tkinter as tk
import ttkbootstrap as ttk
import os
from PIL import Image, ImageTk
import data as data_module

SCRIPT_PATH = os.path.dirname(__file__)
APPLICATION_NAME = "Data Analysis Lab"
DAILY_CSV_PATH = "../Resources/GOOG.US_D1_cleaned.csv"
WEEKLY_CSV_PATH = "../Resources/GOOG.US_W1_cleaned.csv"
MONTHLY_CSV_PATH = "../Resources/GOOG.US_MN1_cleaned.csv"
DAILY_MODEL_PATH = "../Models/sarima_model_daily.pkl"
WEEKLY_MODEL_PATH = "../Models/sarima_model_weekly.pkl"
MONTHLY_MODEL_PATH = "../Models/sarima_model_monthly.pkl"


class DataAnalysisLab:
    def __init__(self, root):
        self.WIDTH = 1920
        self.HEIGHT = 1080
        self.root = root
        self.root.title(APPLICATION_NAME)
        self.root.geometry(f"{self.WIDTH}x{self.HEIGHT}")
        self.root.iconbitmap("")  # Add path to icon file if available

        self.ML_ALGORITHM = "Support Vector Machine (SVR)"
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

        ttk.Label(self.algorithm_frame, text="Select the algorithm:", padding=20).pack(pady=20)

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
        self.graph_frame.pack()
        self.title_graph_frame = ttk.Frame(self.graph_frame, height=15, padding=40)
        self.title_graph_frame.pack()
        self.graph_prediction_frame = ttk.Frame(self.graph_frame)
        self.graph_prediction_frame.pack()


        self.tab2 = ttk.Frame(self.notebook)
        ttk.Label(self.tab2, text="Confusion Matrix", font=("Verdana", 20), padding=20).pack()
        self.naive_frame_superior = ttk.Frame(self.tab2, padding=0)
        self.naive_frame_inferior = ttk.Frame(self.tab2, padding=0)
        self.naive_frame_superior.pack()
        self.naive_frame_inferior.pack()
        self.generate_graph(self.naive_frame_superior, "confusion_matrix_daily.png", False)
        self.generate_graph(self.naive_frame_superior, "confusion_matrix_weekly.png", False)
        self.generate_graph(self.naive_frame_inferior, "confusion_matrix_monthly.png", False)

        self.tab3 = ttk.Frame(self.notebook)
        ttk.Label(self.tab3, text="K-Means", font=("Verdana", 20)).pack(pady=20)
        self.kmeans_frame = ttk.Frame(self.tab3, padding=80)
        self.kmeans_frame.pack()
        self.generate_graph(self.kmeans_frame, "kmeans_elbow.png", True)
        self.generate_graph(self.kmeans_frame, "kmeans_cluster.png", True)


        self.tab4 = ttk.Frame(self.notebook)
        ttk.Label(self.tab4, text="Principal Component Analysis (PCA)", font=("Verdana", 20)).pack()
        self.pca_frame_superior = ttk.Frame(self.tab4, padding=0)
        self.pca_frame_inferior = ttk.Frame(self.tab4, padding=0)
        self.pca_frame_superior.pack()
        self.pca_frame_inferior.pack()
        self.generate_graph(self.pca_frame_superior, "pca_daily.png", True)
        self.generate_graph(self.pca_frame_superior, "pca_weekly.png", True)
        self.generate_graph(self.pca_frame_inferior, "pca_monthly.png", True)


        self.tab5 = ttk.Frame(self.notebook)
        ttk.Label(self.tab5, text="Moving averages - Timeseries", font=("Verdana", 20)).pack(pady=20)
        self.averages_frame_superior = ttk.Frame(self.tab5, padding=0)
        self.averages_frame_inferior = ttk.Frame(self.tab5, padding=0)
        self.averages_frame_superior.pack()
        self.averages_frame_inferior.pack()
        self.generate_graph(self.averages_frame_superior, "moving_averages_daily.png", True)
        self.generate_graph(self.averages_frame_superior, "moving_averages_weekly.png", True)
        self.generate_graph(self.averages_frame_inferior, "moving_averages_monthly.png", True)

        self.tab6 = ttk.Frame(self.notebook)
        ttk.Label(self.tab6, text="Autoregressive Integrated Moving Average (ARIMA)", font=("Verdana", 20)).pack(pady=20)
        self.arima_frame_superior = ttk.Frame(self.tab6, padding=0)
        self.arima_frame_inferior = ttk.Frame(self.tab6, padding=0)
        self.arima_frame_superior.pack()
        self.arima_frame_inferior.pack()
        self.generate_graph(self.arima_frame_superior, "arima_day.png", True)
        self.generate_graph(self.arima_frame_superior, "arima_week.png", True)
        self.generate_graph(self.arima_frame_inferior, "arima_month.png", True)


        self.tab7 = ttk.Frame(self.notebook)
        ttk.Label(self.tab7, text="Seasonal Autoregressive Integrated Moving Average (SARIMA)", font=("Verdana", 20)).pack(pady=20)
        self.sarima_frame_superior = ttk.Frame(self.tab7)
        self.sarima_frame_superior.pack()
        self.sarima_frame_inferior = ttk.Frame(self.tab7)
        self.sarima_frame_inferior.pack()
        #self.generate_graph(self.sarima_frame_superior, "sarima_day.png", True)
        self.generate_graph(self.sarima_frame_superior, "sarima_week.png", True)
        self.generate_graph(self.sarima_frame_superior, "sarima_month.png", True)
        #self.generate_graph(self.sarima_frame_inferior, "sarima_month.png", True)

        self.tab8 = ttk.Frame(self.notebook)
        ttk.Label(self.tab8, text="Hierarchical Clustering", font=("Verdana", 20)).pack(pady=20)
        self.hclustering_frame = ttk.Frame(self.tab8, padding=80)
        self.hclustering_frame.pack()
        self.generate_graph(self.hclustering_frame, "hierarchical_clustering.png", False)

        self.tab9 = ttk.Frame(self.notebook)
        ttk.Label(self.tab9, text="Timeseries decomposition", font=("Verdana", 20)).pack(pady=20)
        self.decomposition_frame_superior = ttk.Frame(self.tab9, padding=0)
        self.decomposition_frame_inferior = ttk.Frame(self.tab9, padding=0)
        self.decomposition_frame_superior.pack()
        self.decomposition_frame_inferior.pack()
        self.generate_graph(self.decomposition_frame_superior, "timeseries_daily_decomposition.png", True)
        self.generate_graph(self.decomposition_frame_superior, "timeseries_weekly_decomposition.png", True)
        self.generate_graph(self.decomposition_frame_inferior, "timeseries_monthly_decomposition.png", True)


        self.notebook.add(self.tab1, text='Prediction')
        self.notebook.add(self.tab2, text='Classification')
        self.notebook.add(self.tab3, text='K-means')
        self.notebook.add(self.tab4, text='PCA')
        self.notebook.add(self.tab5, text='M. Averages')
        self.notebook.add(self.tab6, text='ARIMA')
        self.notebook.add(self.tab7, text='SARIMA')
        self.notebook.add(self.tab8, text='H. Clustering')
        self.notebook.add(self.tab9, text='Decomposition')
        self.notebook.pack()

    def radio_selection(self):
        print(self.ml_algorithm_var.get())
        if self.ml_algorithm_var.get() == "Timeseries":
            print("Timeseries selected!")
            self.timeseries_template()
        else:
            print("Algorithm selected!")
            self.algorithm_template()
            self.prediction_size_var.set(0)

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
        self.clear_graph_frame()
        if (self.time_basis_var.get() != "" or self.prediction_var.get() != ""
                and (self.prediction_size_var.get() != 0 or self.test_size_var.get() != 0)):
            if self.time_basis_var.get() == "daily":
                file_path = DAILY_CSV_PATH
                model_path = DAILY_MODEL_PATH
                data_frequency = "Days"

            elif self.time_basis_var.get() == "weekly":
                file_path = WEEKLY_CSV_PATH
                model_path = WEEKLY_MODEL_PATH
                data_frequency = "Weeks"
            else:
                file_path = MONTHLY_CSV_PATH
                model_path = MONTHLY_MODEL_PATH
                data_frequency = "Months"


            if self.ml_algorithm_var.get() != self.ML_ALGORITHM:
                ttk.Label(self.title_graph_frame,
                          text=f"Timeseries will be made for {self.prediction_size_var.get()} days "
                               "in a "f"{self.time_basis_var.get()} basis").pack(side="top")
                data_module.forecast(model_path, self.prediction_size_var.get(), data_frequency,
                                     "../Graphs/prediction.png")
                self.generate_graph(self.graph_prediction_frame, "prediction.png", True)
            else:
                df_all = data_module.read_and_preprocess(file_path)
                ttk.Label(self.title_graph_frame, text=f"{self.ML_ALGORITHM} prevision will be made with "
                                                       f"{self.test_size_var.get()}% test"
                                                       " in a "f"{self.time_basis_var.get()} basis").pack(side="top")

                test_size = self.test_size_var.get() / 100
                data_module.svm_regression(df_all, self.prediction_var.get(), test_size, "../Graphs/prediction.png")
                self.generate_graph(self.graph_prediction_frame, "prediction.png", True)


    def generate_graph(self, tab, graph_path, resize):
        path = os.path.join("../Graphs", graph_path)
        print(path)
        try:
            image = Image.open(path)
            if(resize):
                image2 = image.resize((900, 450))
            else:
                image2 = image
            photo = ImageTk.PhotoImage(image2)
            label = tk.Label(tab, image=photo)
            label.image = photo  # Keep a reference to avoid garbage collection
            label.pack(side="left")
        except Exception as e:
            print(f"Error loading graph image: {e}")

    def clear_main_frame(self):
        for widget in self.optional_frame.winfo_children():
            widget.destroy()

    def clear_graph_frame(self):
        for widget in self.title_graph_frame.winfo_children():
            widget.destroy()
        for widget in self.graph_prediction_frame.winfo_children():
            widget.destroy()

if __name__ == "__main__":
    root = ttk.Window(themename="darkly")
    app = DataAnalysisLab(root)
    root.mainloop()
