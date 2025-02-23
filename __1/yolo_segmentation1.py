# https://pysource.com/2023/02/21/yolo-v8-segmentation
from ultralytics import YOLO
import numpy as np
import logging

class YOLOSEG:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        # Create a logger instance
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.CRITICAL)  # Set default logging level to CRITICAL to suppress logs

    def set_logging(self, enable):
        if enable:
            self.logger.setLevel(logging.INFO)  # Enable logging
        else:
            self.logger.setLevel(logging.CRITICAL)  # Disable logging

    def detect(self, img ,Fc):
        # Log the shape of the input image
        if(Fc == 1):
            self.logger.info("Input image shape: %s", img.shape)

        height, width, channels = img.shape
        results = self.model.predict(source=img.copy(), save=False, save_txt=False)
        result = results[0]

        # Log the number of results detected
        if(Fc == 1):
            self.logger.info("Number of detection results: %s", len(result))

        segmentation_contours_idx = []
        if len(result) > 0 and hasattr(result, 'masks') and result.masks is not None:
            for seg in result.masks.data:  # Remove the [0] indexing here
                # Convert mask array to contour points
                seg_points = np.array(np.where(seg.cpu().numpy())).T.astype(np.float32)
                
                # Swap x and y coordinates if needed
                if seg_points.shape[1] == 2:
                    # Scale points to image dimensions
                    seg_points[:, 0] = seg_points[:, 0] * width / seg.shape[1]
                    seg_points[:, 1] = seg_points[:, 1] * height / seg.shape[0]
                else:
                    # Handle the case where we only have one dimension
                    seg_points = np.column_stack((seg_points[:, 0] * width / seg.shape[1],
                                                np.zeros_like(seg_points[:, 0])))

                segment = seg_points.astype(np.int32)
                segmentation_contours_idx.append(segment)

        bboxes = np.array(result.boxes.xyxy.cpu(), dtype="int")
        # Get class ids
        class_ids = np.array(result.boxes.cls.cpu(), dtype="int")
        # Get scores
        scores = np.array(result.boxes.conf.cpu(), dtype="float").round(2)

        # Log the detection results
        # self.logger.info("Detection results - BBoxes: %s, Class IDs: %s, Scores: %s", bboxes, class_ids, scores)
        if(Fc == 1):
            self.logger.info("Number of segmentation contours detected: %s", len(segmentation_contours_idx))

        return bboxes, class_ids, segmentation_contours_idx, scores
