from ultralytics import YOLO
model = YOLO("yolov8n.pt")
results = model("https://ultralytics.com/images/bus.jpg") # Predict on an image from a URL
# Process results (results is a list of Results objects, one for each input source)
for result in results:
    boxes = result.boxes # Bounding boxes
    masks = result.masks # Segmentation masks (if using a segmentation model like yolov8n-seg.pt)
    keypoints = result.keypoints # Keypoints (if using a pose estimation model)
    probs = result.probs # Classification probabilities (if using a classification model)
    
    # Show results
    result.show() # Display the image with predictions
    # result.save(filename="result.jpg") # Save the image with predictions
