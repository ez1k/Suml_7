import tkinter
import csv
import tkinter.messagebox
import customtkinter
import pandas as pd
from tkinter import filedialog
from tkinter import ttk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from libs.model import train, predict

customtkinter.set_appearance_mode("System")
customtkinter.set_default_color_theme("blue")

class App(customtkinter.CTk):
    def __init__(self):
        super().__init__()

        self.title("ML LOGIT")
        self.geometry(f"{1100}x{500}")

        self.grid_columnconfigure(1, weight=1)
        self.grid_columnconfigure((2,3), weight=0)
        self.rowconfigure((0,1,2), weight=0) 

        self.sidebar_frame = customtkinter.CTkFrame(self, width=140, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, rowspan=5, sticky="nsew")
        self.sidebar_frame.grid_rowconfigure(4, weight=1)
        self.logo_label = customtkinter.CTkLabel(self.sidebar_frame, text="ML LOGIT 1.0", 
                                                 font=customtkinter.CTkFont(size=20, weight="bold"))
        self.logo_label.grid(row=0, column=0, padx=20, pady=(20,10))
        
        # Bind buttons to methods
        self.sidebar_button_1 = customtkinter.CTkButton(self.sidebar_frame, text="Train", command=self.model_train)
        self.sidebar_button_1.grid(row=1, column=0, padx=20, pady=10)
        self.sidebar_button_2 = customtkinter.CTkButton(self.sidebar_frame, text="Predict", command=self.get_prediction)
        self.sidebar_button_2.grid(row=2, column=0, padx=20, pady=10)
        self.sidebar_button_3 = customtkinter.CTkButton(self.sidebar_frame, text="Plot", command=self.plot)
        self.sidebar_button_3.grid(row=3, column=0, padx=20, pady=10)
        self.sidebar_button_4 = customtkinter.CTkButton(self.sidebar_frame, text="Load csv", command=self.load_csv)
        self.sidebar_button_4.grid(row=4, column=0, padx=20, pady=10)

        self.appearance_mode_label = customtkinter.CTkLabel(self.sidebar_frame, text="Appearance mode:", anchor="w")
        self.appearance_mode_label.grid(row=5, column=0, padx=20, pady=(10,0))
        self.appearance_mode_optionmenu = customtkinter.CTkOptionMenu(self.sidebar_frame, values=["Light", "Dark", "System"],
                                                                      command=self.change_appearance_mode_event)
        self.appearance_mode_optionmenu.grid(row=6, column=0, padx=20, pady=(10,10))

        self.scaling_label = customtkinter.CTkLabel(self.sidebar_frame, text="UI Scaling:", anchor="w")
        self.scaling_label.grid(row=7, column=0, padx=20, pady=(10,0))
        self.scaling_optionmenu = customtkinter.CTkOptionMenu(self.sidebar_frame, values=["80%", "90%", "100%", "110%", "120%"],
                                                              command=self.change_scaling_event)
        self.scaling_optionmenu.grid(row=8, column=0, padx=20, pady=(10,20))
                                    
        self.entry1 = customtkinter.CTkEntry(self, placeholder_text="PointX")
        self.entry1.grid(row=3, column=1, columnspan=1, padx=(20,0), pady=(20,20), sticky="nsew")

        self.entry2 = customtkinter.CTkEntry(self, placeholder_text="PointY")
        self.entry2.grid(row=4, column=1, columnspan=1, padx=(20,0), pady=(0,20), sticky="nsew")

        self.canvas_frame = customtkinter.CTkFrame(master=self)
        self.canvas_frame.grid(row=0, column=2, padx=(20,0), pady=(20,0), sticky="nsew")

        self.csvtable = ttk.Treeview(master=self, show="headings")
        self.csvtable.grid(row=0, column=1, padx=(20,0), pady=(20,0), sticky="nsew")

    def change_appearance_mode_event(self, new_appearance_mode:str):
        customtkinter.set_appearance_mode(new_appearance_mode)

    def change_scaling_event(self, new_scaling:str):
        new_scaling_float = int(new_scaling.replace("%", "")) / 100
        customtkinter.set_widget_scaling(new_scaling_float)

    def plot(self):
        data = pd.read_csv("data/10_points.csv")
        fig = Figure(figsize=(5,5), dpi=100)

        x = data["x"]
        y = data["y"]

        plot1 = fig.add_subplot(1,1,1)

        plot1.scatter(x,y)
        plot1.plot(x,y,color="red")

        canvas = FigureCanvasTkAgg(fig, master=self.canvas_frame)
        canvas.draw()

        canvas.get_tk_widget().pack(padx=(20,20), pady=(20,20))

    def model_train(self):
        model_path = "ml_models/our_model.pkl"
        file_path = "data/10_points.csv"
        train(float(self.entry1.get()), float(self.entry2.get()), file_path, model_path)

    def get_prediction(self):
        model_path = "ml_models/our_model.pkl"
        if model_path:
            self.entry2.delete(0, "end")
            self.entry2.insert(0, str(round(predict(float(self.entry1.get()), model_path)[0][0],2)))

    def load_csv(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if file_path:
            data = pd.read_csv(file_path)
            self.csvtable.delete(*self.csvtable.get_children())
            self.csvtable["columns"] = list(data.columns)
            for col in data.columns:
                self.csvtable.heading(col, text=col)
            for row in data.itertuples(index=False):
                self.csvtable.insert("", "end", values=row)

if __name__ == "__main__":
    app = App()
    app.mainloop()
