from ultralytics import YOLO
import cv2

# Load a model
model = YOLO('C:/Users/ekine/OneDrive/Desktop/CSCI/masters/ComputerVision/project/yolo_scripts/runs/detect/train3/weights/best.pt')
image_path = 'C:/Users/ekine/OneDrive/Desktop/CSCI/masters/ComputerVision/project/detection/test/images/dataset1_back_1.png'

# Load the original image
original_image = cv2.imread(image_path)
results = model(image_path, conf=0.4)

for r in results:
    for box_info in range(len(r.boxes)):
        a = r.boxes[box_info].xyxy[0].tolist()
        print("bounding box coordinates: ", a)
        top_left = round(int(a[0]),2), round(int(a[1]),2)
        bottom_right = round(int(a[2]),2), round(int(a[3]),2)
        color = (0, 255, 0)
        thickness = 2
        original_image = cv2.rectangle(original_image, top_left, bottom_right, color, thickness)

cv2.imshow("YOLOv8 Inference", original_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
