#Inizialization

from ultralytics import YOLO
import os

dataset_folder = r"D:\OneDrive - Università di Cagliari\3 ERASMUS\CORSI\CV\CV_PROJECT\CV-Project-GIT\Data\Datasets\first_dataset"

"""
    REFERENCE: 
    - https://docs.ultralytics.com/modes/train/#usage-examples
"""

# load the model
model = YOLO("yolov8n.yaml")
#train the model
result = model.train(
    data = os.path.join(dataset_folder, 'data.yaml'), # define the dataset
    epochs = 10,                                      # define the numer of epochs                          # set results path
    plots = True                                       # save plots and images during train/val
)
