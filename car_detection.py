import cv2
import numpy as np
from ultralytics import YOLO
from itertools import combinations

model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture(0)
#cap = cv2.VideoCapture(0) 


DISTANCE_THRESHOLD = 100  



while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = cv2.resize(frame, (640, 360))
    
    # tracking
    results = model.track(
        frame,
        stream=True,
        persist=True,
        tracker="bytetrack.yaml"
    )

    # store car centers for distance calculation
    car_centers = []
    car_ids = []

    for r in results:
        for box in r.boxes:
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            label = model.names[cls_id]

            # Only cars
            if label != "car":
                continue

            if conf < 0.20:
                continue
            
            # box and center coordinates here
            x1, y1, x2, y2 = box.xyxy[0].tolist()

            w = x2 - x1
            h = y2 - y1

            frame_h, frame_w = frame.shape[:2]

            # ignore bounding boxes too large (likely your interior)
            if w > frame_w * 0.80 or h > frame_h * 0.80:
                continue

            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            # tracking ID (works across frames)
            track_id = int(box.id[0]) if box.id is not None else -1

            # save center for distance calculation
            car_centers.append((cx, cy))
            car_ids.append(track_id)

            # draw bounding box around targets
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)


            overlay = frame.copy()
            # Rectangle coords
            p1 = (int(x1), int(y1))
            p2 = (int(x2), int(y2))


            #cv2.rectangle(overlay, p1, p2, (0, 255, 0), -1)  # -1 = filled rectangle
            cv2.rectangle(overlay, p1, p2, (0, 255, 0), 2)


            alpha = 0.5  # <-- opacity
            frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)


            
            complete_label = f"{label} {track_id} {conf:.2f}"
            cv2.putText(frame, complete_label, (int(x1), int(y1)-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    # distance calculation & nearest-car
    # set i = index of first car, j = index of second car
    # use 'combinations' to find all the unique pairs of cars without dupes 

    min_dist = float('inf')
    closest_pair = None

    for (i, (cx1, cy1)), (j, (cx2, cy2)) in combinations(enumerate(car_centers), 2):

        # euclidean distance between car centers 
        dist = np.sqrt((cx1 - cx2)**2 + (cy1 - cy2)**2)

        # check if didtance less than current least distance
        if dist < min_dist:
            min_dist = dist
            closest_pair = (i, j, cx1, cy1, cx2, cy2)


    # draw one line between the closest two cars
    if closest_pair is not None:
        i, j, cx1, cy1, cx2, cy2 = closest_pair
        cv2.line(frame, (cx1, cy1), (cx2, cy2), (0,0,255), 2)

        # draw distance label in pixels
        mid_x = int((cx1 + cx2)/2)
        mid_y = int((cy1 + cy2)/2)
        cv2.putText(frame, f"{int(min_dist)}px", (mid_x, mid_y-5),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)


        # if distance too close, draw warning text
        if dist < DISTANCE_THRESHOLD:
            # draw warning text near line
            cv2.putText(frame, "Too Close!", (mid_x, mid_y+15),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)

    
    cv2.imshow("YOLOv8 Car Tracking + Distances", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
