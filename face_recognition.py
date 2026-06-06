import cv2

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

count = 1

print("Press 'c' to capture photo")
print("Press 'q' to exit")

while True:

    ret, frame = cap.read()

    if not ret:
        break

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5
    )

    for (x,y,w,h) in faces:

        cv2.rectangle(
            frame,
            (x,y),
            (x+w,y+h),
            (0,255,0),
            2
        )

    cv2.imshow(
        "Face Detection",
        frame
    )

    key = cv2.waitKey(1)

    # Capture image

    if key == ord('c'):

        filename = f"captured_face_{count}.jpg"

        cv2.imwrite(
            filename,
            frame
        )

        print("Saved:", filename)

        count += 1

    # Exit

    elif key == ord('q'):

        break

cap.release()

cv2.destroyAllWindows()
