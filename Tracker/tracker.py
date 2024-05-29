import os
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
import torchvision.transforms as T
import cv2

# Load pre-trained Faster R-CNN model
model = fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()

# Define image transformation
transform = T.Compose([
    T.ToTensor()
])

# Function to preprocess thermal images
def preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img_tensor = transform(img)
    return img_tensor.unsqueeze(0)

# Function to detect vehicles
def detect_vehicles(model, img_tensor):
    with torch.no_grad():
        predictions = model(img_tensor)
    return predictions

# Function to draw boxes on the image
def draw_boxes(image, boxes):
    for box in boxes:
        x1, y1, x2, y2 = map(int, box)  # Convert coordinates to integers
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return image

# Directories
input_dir = '../data/training'
output_dir = '../data/results'
os.makedirs(output_dir, exist_ok=True)

# Process each image in the input directory
for filename in os.listdir(input_dir):
    if filename.endswith('.png'):
        image_path = os.path.join(input_dir, filename)
        
        # Preprocess image
        img_tensor = preprocess_image(image_path)
        
        # Detect vehicles
        predictions = detect_vehicles(model, img_tensor)
        
        # Extract boxes and scores
        boxes = predictions[0]['boxes'].cpu().numpy()
        scores = predictions[0]['scores'].cpu().numpy()
        
        # Filter boxes with a confidence threshold
        confidence_threshold = 0.5
        filtered_boxes = boxes[scores >= confidence_threshold]
        
        # Read the original image
        img = cv2.imread(image_path)
        
        # Draw boxes on the original image
        result_img = draw_boxes(img, filtered_boxes)
        
        # Save the result image
        output_path = os.path.join(output_dir, filename)
        cv2.imwrite(output_path, result_img)
        
        print(f"Processed and saved: {output_path}")

