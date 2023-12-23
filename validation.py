from ultralytics import YOLO

#   Reference:
#

"""
    REFERENCE: 
    - https://docs.ultralytics.com/guides/yolo-performance-metrics/
    - https://docs.ultralytics.com/modes/val/
    - https://docs.ultralytics.com/reference/engine/validator/#ultralytics.engine.validator.BaseValidator.print_results
"""


source_path = r"H:\Il mio Drive\Data\...\data.yaml"
model_path = r"H:\Il mio Drive\runs\detect\train3\weights\best.pt"

model = YOLO(model_path)  # load a custom model

# Validate the model
metrics = model.val(
    data= source_path, #data.yaml
    plots = True,
    rect = True)  # no arguments needed, dataset and settings remembered

##################################################
#########   Plot result from training   ##########
"""
    REFERENCE: 
    - https://docs.ultralytics.com/reference/utils/plotting/#ultralytics.utils.plotting.plt_color_scatter
"""

from ultralytics.utils.plotting import plot_results

plot_results('path..../results.csv') 