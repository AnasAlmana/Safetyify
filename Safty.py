#video_path = 'vid/one.mp4'

def model_infer(video_path,filename):
    import cv2
    import numpy as np
    from ultralytics import YOLO
    


    # Initialize both YOLO models
    human_model = YOLO("yolov8l.pt")
    object_model = YOLO("best.pt")
    cap = cv2.VideoCapture(video_path)

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    out = cv2.VideoWriter(f'static/output/{filename}', fourcc, fps, frame_size)

    # Define colors for each class for visual distinction
    colors = {
        'Hardhat': (50, 50, 255),
        'Mask': (0, 255, 255),
        'NO-Hardhat': (0, 0, 255),
        'NO-Mask': (255, 0, 0),
        'NO-Safety Vest': (255, 255, 0),
        'Person': (0, 255, 0),
        'Safety Cone': (0, 128, 255),
        'Safety Vest': (255, 0, 128),
        'machinery': (128, 0, 255),
        'vehicle': (255, 128, 0)
    }

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Detect humans first
        human_detections = human_model(frame, device="cpu")[0]
        # Filter for class ID 0 (human/person) and then transfer to CPU
        human_boxes = np.array(human_detections.boxes.xyxy[human_detections.boxes.cls == 0].cpu(), dtype="int")
        
        # If humans detected, proceed with object detection
        if len(human_boxes) > 0:
            object_detections = object_model(frame, device="cpu")[0]
            bboxes = np.array(object_detections.boxes.xyxy.cpu(), dtype="int")
            classes = np.array(object_detections.boxes.cls.cpu(), dtype="int")

            for index, (cls, bbox) in enumerate(zip(classes, bboxes)):
                class_name = list(colors.keys())[cls]  # Get the name of the class
                (x, y, x2, y2) = bbox
                cv2.rectangle(frame, (x,y), (x2,y2), colors[class_name], 2)
                confidence_score = object_detections.boxes.conf[index].item()  # Retrieve the confidence score using the detection index
                cv2.putText(frame, f"{class_name} {confidence_score:.2f}", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors[class_name], 2)

        # Save the processed frame
        out.write(frame)
        

        # Optional: Display the frame
        # cv2.imshow("Processed Frame", frame)
        key = cv2.waitKey(1)
        if key == 27:
            break
    
    cap.release()
    out.release()
    cv2.destroyAllWindows()


def convert(video_path):
    import moviepy.editor as moviepy
    clip = moviepy.VideoFileClip(video_path)
    clip.write_videofile("static/uploads/out.mp4")
