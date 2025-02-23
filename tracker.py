import math
import logging

class Tracker:
    def __init__(self):
        # Store the center positions of the objects
        self.center_points = {}
        # Keep the count of the IDs
        self.id_count = 0
        # Create a logger instance
        self.logger = logging.getLogger(__name__)

    def update(self, objects_rect):
        # Objects boxes and ids
        objects_bbs_ids = []

        # Get center point of new object
        for rect in objects_rect:
            x, y, w, h = rect
            cx = (x + x + w) // 2
            cy = (y + y + h) // 2

            # Log the calculated center point
            # self.logger.info("Calculated center point: (%s, %s)", cx, cy)

            # Find out if that object was detected already
            same_object_detected = False
            for id, pt in self.center_points.items():
                dist = math.hypot(cx - pt[0], cy - pt[1])

                if dist < 35:
                    self.center_points[id] = (cx, cy)
                    objects_bbs_ids.append([x, y, w, h, id])
                    same_object_detected = True
                    # Log existing object detection
                    # self.logger.info("Existing object detected: ID %s, Updated center: (%s, %s)", id, cx, cy)
                    break

            # New object is detected we assign the ID to that object
            if not same_object_detected:
                self.center_points[self.id_count] = (cx, cy)
                objects_bbs_ids.append([x, y, w, h, self.id_count])
                # self.logger.info("New object detected: Assigned ID %s, Center: (%s, %s)", self.id_count, cx, cy)
                self.id_count += 1

        # Log the state of center points before cleanup
        # self.logger.info("Center points before cleanup: %s", self.center_points)

        # Clean the dictionary by center points to remove IDs not used anymore
        new_center_points = {}
        for obj_bb_id in objects_bbs_ids:
            _, _, _, _, object_id = obj_bb_id
            center = self.center_points[object_id]
            new_center_points[object_id] = center

        # Update dictionary with IDs not used removed
        self.center_points = new_center_points.copy()

        # Log the state of center points after cleanup
        # self.logger.info("Center points after cleanup: %s", self.center_points)

        return objects_bbs_ids
