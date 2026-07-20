from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image

CLASS_NAMES = [
    "crazing", "inclusion", "patches",
    "pitted_surface", "rolled-in_scale", "scratches"
]

# Severity mapping for each defect type
SEVERITY_MAP = {
    "crazing"        : "High",
    "inclusion"      : "Medium",
    "patches"        : "Medium",
    "pitted_surface" : "High",
    "rolled-in_scale": "Critical",
    "scratches"      : "Low",
}

def load_model(model_path: str = "models/best.pt"):
    return YOLO(model_path)

def detect_defects(model, image: Image.Image, conf_threshold: float = 0.25):
    # Convert PIL to numpy (RGB) because YOLOv8 expects numpy arrays
    img_array = np.array(image)

    # Run inference (results is a list (one result per image))
    results = model.predict(source=img_array, conf=conf_threshold, verbose=False) 
    result  = results[0]

    # Draw image with bounding boxes
    annotated = result.plot() # BGR numpy array
    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB) # RGB since Streamlit and PIL expect RGB
    annotated_pil = Image.fromarray(annotated_rgb) # PIL image for display

    # Extract defect details for the report
    defects = []
    # Loop through every bounding box detected
    for box in result.boxes:
        cls_id     = int(box.cls[0])
        confidence = float(box.conf[0])
        class_name = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else f"class_{cls_id}"
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())  # bounding boxes coordinates from top left to bottom right

        defects.append({
            "class"     : class_name,
            "confidence": round(confidence * 100, 2),
            "severity"  : SEVERITY_MAP.get(class_name, "Unknown"),
            "bbox"      : {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
            "area_px"   : (x2 - x1) * (y2 - y1),
        })

    # Sort by confidence descending
    defects.sort(key=lambda x: x["confidence"], reverse=True) # useful for the table in Streamlit

    return annotated_pil, defects