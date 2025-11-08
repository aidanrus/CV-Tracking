import mediapipe as mp
import cv2

# init drawing tools and model tools to draw and stuff
mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

# this is the video capture object
cap = cv2.VideoCapture(0)

# init holistic model
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic: 
    
    while cap.isOpened():
        ret, frame = cap.read()

        # recolor feed from bgr to rgb cuz opencv uses bgr by default
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # make detections on viceo object
        results = holistic.process(image)
        #print(results.pose_landmarks)

        # we got these to use: face_landmarks, pose_landmarks, left_hand_landmarks, right_hand_landmarks

        # recolor image back to BGR for rendering
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        
        # mp_holistic.FACEMESH_TESSELATION draws the mesh that connects the landmarks
        #face rendering

        mp_drawing.draw_landmarks(image, results.face_landmarks)

        #pose rendering

        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)

        # left hand rendering
        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

        # right hand rendering
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        
        
        cv2.imshow('Holistic Model Detections', image)
    
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
cap.release()
cv2.destroyAllWindows()