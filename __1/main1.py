import cv2
from yolo_segmentation1 import YOLOSEG
import cvzone
from tracker1 import*
import numpy as np

import os    
import logging
from datetime import datetime

Fc = 0 # declared to check the logs for only one iteration

# Create logs directory if it doesn't exist
if not os.path.exists('logs'):
    os.makedirs('logs')
# Configure logging
log_filename = f'logs/egg_counter_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler()  # This will also print logs to console
    ]
)
# Create a logger instance
logger = logging.getLogger(__name__)
# Add logging messages at key points in your code
logger.info("Starting egg counting application")

ys = YOLOSEG("best.pt")

my_file = open("C:\\Users\\kavis\\Desktop\\Egg_Project\\egg_Counter_Y'\\coco1.txt", "r")
data = my_file.read()
class_list = data.split("\n")

cap = cv2.VideoCapture('C:\\Users\\kavis\\Desktop\\Egg_Project\\egg.mp4')
# cap = cv2.VideoCapture('C:\\Users\\kavis\\Desktop\\Egg_Project\\egg.mp4')
if(cap.isOpened()):
    logger.info("the video file is captured successfully")
else:
    logger.error("ERROR IN CAPTURING THE VIDEO FILE")

# Get video properties for the output video
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Create VideoWriter object
# You can change the codec ('mp4v' or 'XVID') if needed
output_path = 'output_video_3_area_1.mp4'
out = cv2.VideoWriter(output_path, 
                     cv2.VideoWriter_fourcc(*'mp4v'),
                     fps, 
                     (1020, 500))  # Match this with your resize dimensions

count = 0

def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:  
        point = [x, y]
        logger.info("Mouse event captured")
        logger.info("The Point is as follows = %s", point)

cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)
tracker = Tracker()

# area = [(50,475),(875,475),(875,490),(50,490)]  # Bottom horizontal zone
area_center = [(40,240),(865,240),(865,255),(40,255)]  # Center horizontal zone
# area_left = [(30,10),(870,10),(870,25),(30,25)]  # Top horizontal zone

# counter1 = []
# counter2 = []  
counter3 = []  # New counter for center zone

while True:
    ret, frame = cap.read()
    if not ret:
        break
    if ret == 1:
        Fc = Fc + 1 
    
    frame = cv2.resize(frame, (1020, 500))
    overlay = frame.copy()
    alpha = 0.5

    if(Fc == 1):
        logger.info("Processing frame...")
        logger.info("Frame shape: %s", frame.shape)
        
    ys.set_logging(False)
    bboxes, classes, segmentations, scores = ys.detect(frame,Fc)
    if(Fc == 1):
        logger.info("Detection results - BBoxes: %s, Classes: %s, Scores: %s", bboxes, classes, scores)

    tracker.set_logging(True) # if you want to log the errors from this file then log the errors to the true
    bbox_idx = tracker.update(bboxes,Fc)
    for bbox, seg in zip(bbox_idx, segmentations):
        x3, y3, x4, y4, id = bbox
        cx = int(x3 + x4) // 2
        cy = int(y3 + y4) // 2
        
        # result1 = cv2.pointPolygonTest(np.array(area, np.int32), (cx, cy), False)
        # result2 = cv2.pointPolygonTest(np.array(area_left, np.int32), (cx, cy), False)
        result3 = cv2.pointPolygonTest(np.array(area_center, np.int32), (cx, cy), False)  # Add center check
        
        # keft_area
        # if result1 >= 0:
        #     logger.info("Object %s entered Bottom area", id)
        #     cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1)
        #     cv2.fillPoly(overlay, [seg], (0, 0, 255))
        #     cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 2, frame)
        #     cvzone.putTextRect(frame, f'{id}', (x3, y3), 1, 1)
        #     if counter1.count(id) == 0:
        #         counter1.append(id) 
              
        # right_area  
        # if result2 >= 0:
        #     logger.info("Object %s entered Left area", id)
        #     cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1)
        #     cv2.fillPoly(overlay, [seg], (0, 0, 255))
        #     cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 2, frame)
        #     cvzone.putTextRect(frame, f'{id}', (x3, y3), 1, 1)
        #     if counter2.count(id) == 0:
        #         counter2.append(id)
              
        if result3 >= 0:
            if(Fc==1):
                logger.info("Object %s entered Center area", id)
            cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1)
            cv2.fillPoly(overlay, [seg], (0, 0, 255))
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 2, frame)
            cvzone.putTextRect(frame, f'{id}', (x3, y3), 1, 1)
            if counter3.count(id) == 0:
                counter3.append(id)
              
    # Draw all three areas
    # cv2.polylines(frame, [np.array(area, np.int32)], True, (255, 0, 0), 2)
    # cv2.polylines(frame, [np.array(area_left, np.int32)], True, (255, 0, 0), 2)
    cv2.polylines(frame, [np.array(area_center, np.int32)], True, (255, 0, 0), 2)  # Draw center area

    # Update counts display
    # ca1 = len(counter1)
    # ca2 = len(counter2)
    ca3 = len(counter3)
    
    # logger.info("Counts - Top: %s, Center: %s, Bottom: %s", ca1, ca3, ca2)

    # Display all counts with black text
    # cvzone.putTextRect(frame, f'Top Count: {ca1}', (20, 30), 1, 1, colorR=(0, 0, 0), colorB=(255, 255, 255))
    cvzone.putTextRect(frame, f'Center Count: {ca3}', (20, 55), 1, 1, colorR=(0, 0, 0), colorB=(255, 255, 255))
    # cvzone.putTextRect(frame, f'Bottom Count: {ca2}', (20, 80), 1, 1, colorR=(0, 0, 0), colorB=(255, 255, 255))

    # Write the frame to output video
    out.write(frame)

    cv2.imshow("RGB", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Release everything
logger.info("THE FINAL VALUE OF FC IS ",Fc)
cap.release()
out.release()
cv2.destroyAllWindows()

