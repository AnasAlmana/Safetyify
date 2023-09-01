# from ultralyticsplus import YOLO, render_result

# # load model
# model = YOLO('keremberke/yolov8m-protective-equipment-detection')

# # set model parameters
# model.overrides['conf'] = 0.25  # NMS confidence threshold
# model.overrides['iou'] = 0.45  # NMS IoU threshold
# model.overrides['agnostic_nms'] = False  # NMS class-agnostic
# model.overrides['max_det'] = 1000  # maximum number of detections per image

# # set image
# image = 'https://github.com/ultralytics/yolov5/raw/master/data/images/zidane.jpg'

# # perform inference
# results = model.predict(image)

# # observe results
# print(results[0].boxes)
# render = render_result(model=model, image=image, result=results[0])
# render.show()




# from ultralytics import YOLO
# import cv2
# import numpy as np

# model = YOLO("best.pt")

# cap = cv2.VideoCapture("SafetyCulture/vid/test.mp4")

# # Define the codec and create VideoWriter object
# fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4 format
# fps = int(cap.get(cv2.CAP_PROP_FPS))
# frame_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
# out = cv2.VideoWriter('output_detect.mp4', fourcc, fps, frame_size)

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     results = model(frame, device="mps")
#     result = results[0]
#     bboxes = np.array(result.boxes.xyxy.cpu(), dtype="int")
#     classes = np.array(result.boxes.cls.cpu(), dtype="int")

#     for cls, bbox, in zip(classes, bboxes):
#         (x, y, x2, y2) = bbox
#         cv2.rectangle(frame, (x,y), (x2,y2), (0,0,225), 2)
#         cv2.putText(frame, str(cls), (x, y - 5), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 225), 2)

#     # Write the frame to the output video
#     out.write(frame)

#     # cv2.imshow("Img", frame)
#     key = cv2.waitKey(1)
#     if key == 27:
#         break

# cap.release()
# out.release()  # Finalize the video file.
# cv2.destroyAllWindows()



# from ultralytics import YOLO
# import cv2 as cv 
# import numpy as np
# def main():
#     # Load YOLO model
#     yolo_model = YOLO("best.pt")

#     cap = cv.VideoCapture("vid/test.mp4")
    
#     # Define the codec and create VideoWriter object
#     fourcc = cv.VideoWriter_fourcc(*'mp4v') 
#     fps = int(cap.get(cv.CAP_PROP_FPS))
#     frame_size = (int(cap.get(cv.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv.CAP_PROP_FRAME_HEIGHT)))
#     out = cv.VideoWriter('output_detect.mp4', fourcc, fps, frame_size)
    
#     # Class names as per your dataset
#     classes = ['Hardhat', 'Mask', 'NO-Hardhat', 'NO-Mask', 'NO-Safety Vest', 'Person', 'Safety Cone', 'Safety Vest', 'machinery', 'vehicle']

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break

#         results = yolo_model(frame, device="mps")
#         print(type(results))  # Check the type of results
#         print(results)  # Print results to see the format
        
#         for det in results:
#             if len(det) < 6:  # 4 for xyxy, 1 for conf, and 1 for cls
#                 continue
#             xyxy, conf, cls = det[:4], det[4], det[5]
#             label = classes[int(cls)]
#             frame = cv.putText(frame, label, (int(xyxy[0]), int(xyxy[1]) - 5), cv.FONT_HERSHEY_PLAIN, 2, (0, 0, 225), 2)
#             frame = cv.rectangle(frame, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (0, 0, 225), 2)


#         # Write the frame with detections to the output video
#         out.write(frame)

#         # Uncomment to view the video while processing
#         # cv.imshow("Output", frame)
#         key = cv.waitKey(1)
#         if key == 27:
#             break

#     cap.release()
#     out.release()
#     cv.destroyAllWindows()

# if __name__ == "__main__":
#     main()




# from ultralytics import YOLO
# import cv2
# import numpy as np

# model = YOLO("best.pt")

# cap = cv2.VideoCapture("vid/test.mp4")

# # Define the codec and create VideoWriter object
# fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4 format
# fps = int(cap.get(cv2.CAP_PROP_FPS))
# frame_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
# out = cv2.VideoWriter('output_detect.mp4', fourcc, fps, frame_size)

# # Class names as per your dataset
# class_names = ['Hardhat', 'Mask', 'NO-Hardhat', 'NO-Mask', 'NO-Safety Vest', 'Person', 'Safety Cone', 'Safety Vest', 'machinery', 'vehicle']

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     results = model(frame, device="mps")
#     result = results[0]
#     bboxes = np.array(result.boxes.xyxy.cpu(), dtype="int")
#     classes_indices = np.array(result.boxes.cls.cpu(), dtype="int")

#     for cls_idx, bbox in zip(classes_indices, bboxes):
#         (x, y, x2, y2) = bbox
#         class_name = class_names[cls_idx]  # Fetch class name using index
#         cv2.rectangle(frame, (x, y), (x2, y2), (0, 0, 225), 2)
#         cv2.putText(frame, class_name, (x, y - 5), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 225), 2)

#     # Write the frame to the output video
#     out.write(frame)

#     # cv2.imshow("Img", frame)
#     key = cv2.waitKey(1)
#     if key == 27:
#         break

# cap.release()
# out.release()  # Finalize the video file.
# cv2.destroyAllWindows()







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