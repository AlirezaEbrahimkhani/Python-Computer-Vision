import numpy as np
import cv2 as cv

backSub = cv.createBackgroundSubtractorMOG2()
system_camera = cv.VideoCapture(0)

if not system_camera.isOpened():
    print('Unable to open camera')
    exit(0)

while True:
    ret, frame = system_camera.read()
    if frame is None:
        break

    fgMask = backSub.apply(frame)

    cv.rectangle(frame, (10, 2), (100, 20), (255, 255, 255), -1)
    cv.putText(frame, str(system_camera.get(cv.CAP_PROP_POS_FRAMES)), (15, 15),
               cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

    # Apply median filter
    median = cv.medianBlur(fgMask, 5)
    gauss = cv.GaussianBlur(fgMask, (5, 5), 0)

    cv.imshow('Frame', frame)
    cv.imshow('FG Mask', fgMask)
    cv.imshow('Gaussian Filter applied', gauss)
    cv.imshow('Median Filter applied', median)

    # Waits for a user input to quit the application
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
system_camera.release()
system_camera.destroyAllWindows()
