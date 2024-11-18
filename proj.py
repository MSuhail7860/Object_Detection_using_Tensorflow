import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import cv2
from pathlib import Path
import time
import tkinter as tk
from tkinter import filedialog

def select_image():
    """
    Open a file dialog to select an image file
    """
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    
    file_path = filedialog.askopenfilename(
        title="Select Image File",
        filetypes=[
            ("Image files", "*.jpg *.jpeg *.png *.bmp *.gif"),
            ("All files", ".")
        ]
    )
    
    return file_path if file_path else None

def load_model(model_url):
    """
    Load model from TensorFlow Hub with proper error handling
    """
    print(f"Loading model from: {model_url}")
    try:
        model = hub.load(model_url)
        print("Model loaded successfully!")
        return model
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        raise Exception(f"Failed to load model: {str(e)}")

def process_image(image_path):
    """
    Load and preprocess image for the model
    """
    # Check if image exists
    if not Path(image_path).exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    print(f"Processing image: {image_path}")
    
    # Read image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")
    
    # Convert BGR to RGB (TensorFlow models expect RGB)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Resize image
    image_resized = cv2.resize(image_rgb, (320, 320))
    
    # Convert to uint8 tensor and add batch dimension
    image_tensor = tf.convert_to_tensor(image_resized, dtype=tf.uint8)
    image_tensor = tf.expand_dims(image_tensor, axis=0)
    
    return image, image_tensor

def draw_detections(image, boxes, classes, scores, threshold=0.5):
    """
    Draw detection boxes and labels on the image
    """
    # COCO class labels
    COCO_CLASSES = {
        1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 5: 'airplane',
        6: 'bus', 7: 'train', 8: 'truck', 9: 'boat', 10: 'traffic light',
        11: 'fire hydrant', 13: 'stop sign', 14: 'parking meter', 15: 'bench',
        16: 'bird', 17: 'cat', 18: 'dog', 19: 'horse', 20: 'sheep',
        21: 'cow', 22: 'elephant', 23: 'bear', 24: 'zebra', 25: 'giraffe',
        27: 'backpack', 28: 'umbrella', 31: 'handbag', 32: 'tie', 33: 'suitcase',
        34: 'frisbee', 35: 'skis', 36: 'snowboard', 37: 'sports ball', 38: 'kite',
        39: 'baseball bat', 40: 'baseball glove', 41: 'skateboard', 42: 'surfboard',
        43: 'tennis racket', 44: 'bottle', 46: 'wine glass', 47: 'cup', 48: 'fork',
        49: 'knife', 50: 'spoon', 51: 'bowl', 52: 'banana', 53: 'apple',
        54: 'sandwich', 55: 'orange', 56: 'broccoli', 57: 'carrot', 58: 'hot dog',
        59: 'pizza', 60: 'donut', 61: 'cake', 62: 'chair', 63: 'couch',
        64: 'potted plant', 65: 'bed', 67: 'dining table', 70: 'toilet',
        72: 'tv', 73: 'laptop', 74: 'mouse', 75: 'remote', 76: 'keyboard',
        77: 'cell phone', 78: 'microwave', 79: 'oven', 80: 'toaster', 81: 'sink',
        82: 'refrigerator', 84: 'book', 85: 'clock', 86: 'vase', 87: 'scissors',
        88: 'teddy bear', 89: 'hair drier', 90: 'toothbrush'
    }
    
    h, w, _ = image.shape
    detections_to_show = 0
    
    for box, class_id, score in zip(boxes, classes, scores):
        if score >= threshold:
            detections_to_show += 1
            # Convert normalized coordinates to pixel values
            y_min, x_min, y_max, x_max = box
            x_min, x_max = int(x_min * w), int(x_max * w)
            y_min, y_max = int(y_min * h), int(y_max * h)
            
            # Get class name
            class_name = COCO_CLASSES.get(class_id, f'Class {class_id}')
            
            # Draw box
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            
            # Draw label with score
            label = f'{class_name}: {score:.2f}'
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(image, (x_min, y_min - label_size[1] - 10),
                         (x_min + label_size[0], y_min), (0, 255, 0), -1)
            cv2.putText(image, label, (x_min, y_min - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    
    print(f"Found {detections_to_show} objects with confidence >= {threshold}")
    return image

def main():
    # Model URL for EfficientDet-D0
    MODEL_URL = "https://tfhub.dev/tensorflow/efficientdet/d0/1"
    
    try:
        # Select image file
        print("Please select an image file...")
        image_path = select_image()
        
        if image_path is None:
            print("No image selected. Exiting...")
            return
        
        # Load model
        model = load_model(MODEL_URL)
        
        # Process image
        image, image_tensor = process_image(image_path)
        
        # Run inference
        print("Running inference...")
        start_time = time.time()
        
        # Get the detection function
        detect_fn = model.signatures['serving_default']
        result = detect_fn(input_tensor=image_tensor)
        
        inference_time = time.time() - start_time
        print(f"Inference completed in {inference_time:.2f} seconds")
        
        # Extract results
        detection_boxes = result['detection_boxes'][0].numpy()
        detection_classes = result['detection_classes'][0].numpy().astype(int)
        detection_scores = result['detection_scores'][0].numpy()
        
        # Draw detections
        output_image = draw_detections(image.copy(), detection_boxes, 
                                     detection_classes, detection_scores,
                                     threshold=0.3)  # Lowered threshold for more detections
        
        # Display results
        cv2.imshow('Object Detection Results', output_image)
        print("Press any key to close the image window...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        # Save output
        output_path = str(Path(image_path).parent / 'output_image.jpg')
        cv2.imwrite(output_path, output_image)
        print(f"Results saved to: {output_path}")
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")

if __name__ == "_main_":
    main()

