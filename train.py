#Inizialization

from ultralytics import YOLO
import os

dataset_folder = ''

# Reference: https://docs.ultralytics.com/modes/train/#usage-examples
# load the model
model = YOLO("yolov8n.yaml")
#train the model
result = model.train(
    data = os.path.join(dataset_folder, 'data.yamal'), # define the dataset
    epochs = 100,                                      # define the numer of epochs                          # set results path
    plots = True                                       # save plots and images during train/val
)
