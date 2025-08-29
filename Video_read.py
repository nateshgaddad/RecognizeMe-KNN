import cv2

cam = cv2.VideoCapture(0)

while True:
    success, img = cam.read()
    if not success:
        print("reading camera failed!")
        break
    
    cv2.imshow("Image Window", img)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):  # press q to quit
        break

cam.release()
cv2.destroyAllWindows()
