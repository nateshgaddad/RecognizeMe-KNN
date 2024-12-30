import cv2

cam = cv2.VideoCapture(0)

while True:
    sucess, img = cam.read()
    if not sucess:
        print("reading camera failed!")
    cv2.imshow("Image Window", img)
    cv2.waitKey(1)


