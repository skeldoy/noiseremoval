import torch
from matplotlib import pyplot as plt
import cv2

# Load the pre-trained model from PyTorch Hub
model = torch.hub.load('ultralytics/yolov5', 'yolov5x')  # or yolov5n - yolov5x6, custom

# Set the image path
img_path = "crowd.jpg"

# Load the image with OpenCV
img = cv2.imread(img_path)

# Convert the image from BGR to RGB
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Perform inference on the image
results = model(img)

# Print detection results
print(results.pandas().xyxy[0])

# Draw bounding boxes on the image
for index, row in results.pandas().xyxy[0].iterrows():
    x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
    label = row['name']
    conf = round(row['confidence'], 2)
    text = f'{label} {conf}'
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(img, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

# Convert the image from RGB to BGR for displaying with Matplotlib
img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

# Display the image using Matplotlib
plt.imshow(img)
plt.show()
cv2.imwrite('output_image.jpg', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
