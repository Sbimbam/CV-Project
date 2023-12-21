from ultralytics import YOLO

# Reference: https://docs.ultralytics.com/modes/predict/

source_folder = r"H:\Il mio Drive\Data\Video test\demo.mp4"
model_path = r"H:\Il mio Drive\runs\detect\train3\weights\best.pt"
model = YOLO(model_path)
model.predict(
    source=source_folder, # data to be analyzed
    save=True,
    conf=0.25, #take only detections with confidence level higher that the set_value
    show=True # Display preds.
)
#it is possibile to save also the cropped license with save_crop=True

