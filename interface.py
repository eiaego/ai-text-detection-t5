import sys
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QWidget,
    QLabel, QRadioButton, QLineEdit, QTextEdit, QPushButton, QButtonGroup
)
import torch
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
import numpy as np
from .src.encoder import encoder
from .src.classifier import classifier

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.encoder = encoder()
        self.classifier = classifier()
        self.scale = 0.5

        self.setWindowTitle("Text Classifier")

        self.layout = QVBoxLayout()
        
        desc_layout = QHBoxLayout()
        # Description Text
        desc_layout.addWidget(QLabel("Select device"))
        desc_layout.addWidget(QLabel("Select classification type"))
        self.layout.addLayout(desc_layout)
        
        # CUDA and CPU Radio Buttons
        self.device_group = QButtonGroup()
        cuda_radio = QRadioButton("CUDA")
        cpu_radio = QRadioButton("CPU")
        self.device_group.addButton(cuda_radio)
        self.device_group.addButton(cpu_radio)
        
        # Check CUDA availability, set default
        if torch.cuda.is_available():
            cuda_radio.setEnabled(True)
            cuda_radio.setChecked(True)
            cpu_radio.setEnabled(True)
        else:
            cuda_radio.setEnabled(False)
            cpu_radio.setEnabled(True)
            cpu_radio.setChecked(True)

        # Binary and Multiclass Radio Buttons
        self.classification_group = QButtonGroup()
        binary_radio = QRadioButton("Binary")
        multiclass_radio = QRadioButton("Multiclass")
        self.classification_group.addButton(binary_radio)
        self.classification_group.addButton(multiclass_radio)

        # Set default
        binary_radio.setChecked(True)
        
        # Set layout shape
        h_layout_radio = QHBoxLayout()
        device_layout = QVBoxLayout()
        class_layout = QVBoxLayout()

        device_layout.addWidget(cuda_radio)
        device_layout.addWidget(cpu_radio)
        h_layout_radio.addLayout(device_layout)

        class_layout.addWidget(binary_radio)
        class_layout.addWidget(multiclass_radio)
        h_layout_radio.addLayout(class_layout)

        self.layout.addLayout(h_layout_radio)

        # Input Text Box
        self.input_text_box = QTextEdit()
        self.input_text_box.setPlaceholderText("Please enter the input text")
        self.layout.addWidget(self.input_text_box)
        
        # Go Button
        go_button = QPushButton("Go")
        go_button.clicked.connect(self.execute_function)
        self.layout.addWidget(go_button)

        # Console Output Box
        self.console_output_box = QTextEdit()
        self.console_output_box.setReadOnly(True)
        
        self.layout.addWidget(self.console_output_box)
        
        self.canvas = None
    
        # Step 5: Set the size of the QWidget based on the Desktop Resolution
        self.resize(600, 750)
        # Set layout
        container = QWidget()
        container.setLayout(self.layout)
        self.setCentralWidget(container)
    
    def execute_function(self):
        input_text = self.input_text_box.toPlainText()
        device_selected = self.device_group.checkedButton().text().lower()
        classification_selected = self.classification_group.checkedButton().text()
        embeddings = self.encoder.encode(input_text, device_selected)
        res, percentages, labels = self.classifier.classify(classification_selected, embeddings, device_selected)
        
        output = (f"Device: {device_selected}\n"
                  f"Classification: {classification_selected}\n"
                  f"Result: {res}\n")
        
        del input_text, device_selected, classification_selected, embeddings, res
        self.console_output_box.append(output)
        self.display_donut_chart(percentages, labels)

    def display_donut_chart(self, percentages, class_names):
        # Remove any existing chart
        if self.canvas:
            self.layout.removeWidget(self.canvas)
            self.canvas.deleteLater()
            self.canvas = None
            

        sizes = percentages * 100
        explode = explode = [0.1] * len(class_names) # explode a slice if required

        # Create the pie chart with a hole in the center (donut chart)
        fig, ax = plt.subplots()
        wedges, _, autotexts = ax.pie(sizes, explode=explode,
                                    autopct='',
                                    startangle=140, wedgeprops=dict(width=0.3))

        # Calculate the positions for the labels and percentages
        kw_label = dict(arrowprops=dict(arrowstyle="-"), va="center")
        kw_pct = dict(arrowprops=dict(arrowstyle="-", linestyle='--'), va="center")

        # Limits for bounding box
        xlim, ylim = ax.get_xlim(), ax.get_ylim()

        for i, (wedge, size) in enumerate(zip(wedges, sizes)):
            angle = (wedge.theta2 - wedge.theta1) / 2.0 + wedge.theta1
            x = np.cos(np.radians(angle))
            y = np.sin(np.radians(angle))
            
            # Position for the distant labels
            x_label = 1.35 * np.sign(x)
            y_label = 1.4 * y
            
            # Ensure label stays within bounding limits
            if x_label > xlim[1]:
                x_label = xlim[1]
            if x_label < xlim[0]:
                x_label = xlim[0]
            if y_label > ylim[1]:
                y_label = ylim[1]
            if y_label < ylim[0]:
                y_label = ylim[0]
            
            alignment_label = {True: 'left', False: 'right'}[x > 0]
            
            # Position for the percentages near the center
            x_pct = 0.4 * x
            y_pct = 0.4 * y
            percentage = f'{size/sum(sizes)*100:.1f}%'
            
            # Annotate distant labels
            ax.annotate(class_names[i], xy=(x, y), xytext=(x_label, y_label),
                        horizontalalignment=alignment_label, **kw_label)
            
            # Annotate percentages closer to the center
            ax.annotate(percentage, xy=(x, y), xytext=(x_pct, y_pct),
                        horizontalalignment='center', **kw_pct)

        # Create a canvas to add the figure to QWidget
        self.canvas = FigureCanvas(fig)

        # Add the canvas to the existing layout
        self.layout.addWidget(self.canvas)

        # Redraw the QWidget to include the new chart
        self.update()

app = QApplication(sys.argv)
window = MainWindow()
window.show()
app.exec()