from collections import defaultdict
import cv2
import numpy as np
from ultralytics.utils.plotting import colors
from ultralytics import YOLO

"""
    REFERENCE: 
    - https://docs.ultralytics.com/modes/track/#plotting-tracks-over-time
    - https://docs.ultralytics.com/modes/track/
"""

source_path = r"H:\Il mio Drive\Data\Video test\demo.mp4"
model_path = r"H:\Il mio Drive\runs\detect\train3\weights\best.pt"

# Load the YOLOv8 model
model = YOLO(model_path)

# Open the video file
cap = cv2.VideoCapture(source_path)

frameRate = int(cap.get(cv2.CAP_PROP_FPS))
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Store the track history
track_history = defaultdict(lambda: [])

# open video file for writing
videoWriter = cv2.VideoWriter(
        'videoOut.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'),
        frameRate, (frame_width, frame_height))

if not videoWriter.isOpened():
    print(f"Error: Unable to open video file for writing {source_path}.")
    exit(-1)

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run YOLOv8 tracking on the frame, persisting tracks between frames
        results = model.track(frame, persist=True)

        boxes = results[0].boxes.xywh.cpu()
        clss = results[0].boxes.cls.cpu().tolist()
        # Get the boxes and track IDs
        #print("Attributes: /n", results[0].boxes)
        if results[0].boxes.id is not None: track_ids = results[0].boxes.id.int().cpu().tolist()
        # I added this check to avoid errors if there is no detection in some frame
        # Visualize the results on the frame
        annotated_frame = results[0].plot()

        # Plot the tracks
        for box, track_id, cls in zip(boxes, track_ids, clss):
            x, y, w, h = box
            track = track_history[track_id]
            track.append((float(x), float(y)))  # x, y center point
            if len(track) > 30:  # retain 90 tracks for 90 frames
                track.pop(0)
            #print(dir(model.track))
            #color=frame[int(y),int(x)]
            # Draw the tracking lines
            points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(annotated_frame, [points], isClosed=False, color=colors(cls, True), thickness=3)

        # Display the annotated frame
        cv2.imshow("YOLOv8 Tracking", annotated_frame)
        videoWriter.write(annotated_frame)
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
videoWriter.release()
cv2.destroyAllWindows()
