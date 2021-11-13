from __future__ import print_function
import cv2 as cv
import numpy as np

system_camera = cv.VideoCapture(0)

if not (system_camera.isOpened()):
    print("Could not open video device")

system_camera.set(cv.CAP_PROP_FRAME_WIDTH, 640)
system_camera.set(cv.CAP_PROP_FRAME_HEIGHT, 480)


while(True):
    # Create Background Subtractor objects
    backSub = cv.createBackgroundSubtractorMOG2()
    # Capture frame-by-frame
    ret, frame = system_camera.read()
    
    # Update the background model
    fgMask = backSub.apply(frame)

    # Get the frame number and write it on the current frame
    cv.rectangle(frame, (10, 2), (100, 20), (255, 255, 255), -1)
    cv.putText(frame, str(system_camera.get(cv.CAP_PROP_POS_FRAMES)), (15, 15),
               cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
    
    # Apply median filter
    median = cv.medianBlur(fgMask, 5)
    gauss = cv.GaussianBlur(fgMask, (5, 5), 0)
    median_filter_frame = np.concatenate((median, gauss), axis=1)

    # Display the resulting frame
    cv.imshow('Frame', frame)
    cv.imshow('FG Mask', fgMask)
    cv.imshow('Median Filter Apply', median_filter_frame)

    # Waits for a user input to quit the application
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
system_camera.release()
system_camera.destroyAllWindows()
