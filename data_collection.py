import cv2 
import numpy as np 
import os

# Init Camera
cap = cv2.VideoCapture(0)

# Load Haar Cascade (make sure the file exists in the same folder)
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")
if face_cascade.empty():
    print("⚠️ Could not load Haar Cascade. Check the path!")
    exit()

skip = 0
face_data = []

# Create "data" folder if it doesn't exist
dataset_path = os.path.join(os.getcwd(), "data")
os.makedirs(dataset_path, exist_ok=True)

file_name = input("Enter the name of the person : ").strip()

# Number of face samples you want
max_samples = 50  

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # detect faces
    faces = face_cascade.detectMultiScale(gray_frame, 1.3, 5)
    if len(faces) == 0:
        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue

    # sort by area -> largest face
    faces = sorted(faces, key=lambda f: f[2] * f[3])

    for face in faces[-1:]:  # take the largest face
        x, y, w, h = face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)

        offset = 10
        face_section = frame[y - offset:y + h + offset, x - offset:x + w + offset]

        # Avoid invalid crop
        if face_section.size == 0:
            continue

        face_section = cv2.resize(face_section, (100, 100))

        skip += 1
        if skip % 10 == 0:   # save every 10th frame
            face_data.append(face_section)
            print(f"✅ Collected samples: {len(face_data)} / {max_samples}")

    cv2.imshow("Frame", frame)
    cv2.imshow("Face Section", face_section)

    # Stop if required samples collected
    if len(face_data) >= max_samples:
        break

    key = cv2.waitKey(1) & 0xFF
    if key == 27:   # ESC key
        break


# Only save if data is collected
if len(face_data) > 0:
    face_data = np.asarray(face_data)
    face_data = face_data.reshape((face_data.shape[0], -1))
    print("Final shape of data:", face_data.shape)

    file = os.path.join(dataset_path, file_name + ".npy")
    np.save(file, face_data)
    print(f"🎉 Data successfully saved at: {file}")
else:
    print("⚠️ No face data collected. File not saved.")

cap.release()
cv2.destroyAllWindows()
