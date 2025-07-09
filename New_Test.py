import cv2
import numpy as np
import os

# Configuration
weights_path = {"Path to the weight file"}
config_path = {"Path to the config file"}
labels_path = {"Path to the label file"}
img_path = {"Path to the image file"}
output_path = {"Path to the output file"}
confidence_threshold = 0.3
nms_threshold = 0.4

# Verify file existence
for path in [weights_path, config_path, labels_path, img_path]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

# Load the model
net = cv2.dnn.readNet(weights_path, config_path)
classes = []
with open(labels_path, "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Optional: Enable CUDA if available
# net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
# net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

# Load image
img = cv2.imread(img_path)
if img is None:
    raise FileNotFoundError(f"Failed to load image at {img_path}")
height, width, channels = img.shape

# Preprocess image
blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
net.setInput(blob)
outs = net.forward(output_layers)

# Process detections
class_ids = []
confidences = []
boxes = []
for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > confidence_threshold:
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)
            x = int(center_x - w / 2)
            y = int(center_y - h / 2)
            boxes.append([x, y, w, h])
            confidences.append(float(confidence))
            class_ids.append(class_id)

# Non-max suppression
indexes = cv2.dnn.NMSBoxes(boxes, confidences, confidence_threshold, nms_threshold)

# Draw bounding boxes
for i in range(len(boxes)):
    if i in indexes:
        x, y, w, h = boxes[i]
        label = str(classes[class_ids[i]])
        confidence = confidences[i]
        color = (0, 255, 0)  # Green
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        cv2.putText(img, f"{label} {confidence:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

# Save and display result
os.makedirs(os.path.dirname(output_path), exist_ok=True)
cv2.imwrite(output_path, img)
cv2.imshow("Output", img)
cv2.waitKey(0)
cv2.destroyAllWindows()