# import cv2
# from ultralytics import YOLO

# model = YOLO("yolov8x.pt")

# results = model.predict(source="0", show=True, device="mps")

# print(results)


import cv2
from ultralytics import YOLO

# Load the model
model = YOLO("yolov8m.pt")

# Start video capture
cap = cv2.VideoCapture(0)  # Use 0 for the first connected camera

while True:
    # Read a frame from the camera
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Make predictions
    results = model.predict(source=frame, show=False, device="mps")

    # Render results on the frame
    annotated_frame = results[0].plot()  # Annotate frame with predictions

    # Display the frame
    cv2.imshow("YOLO Predictions", annotated_frame)

    # Break the loop if 'ESC' is pressed
    key = cv2.waitKey(1)
    if key == 27:  # ASCII code for ESC
        print("ESC pressed, exiting...")
        break

# Release the camera and close OpenCV windows
cap.release()
cv2.destroyAllWindows()


