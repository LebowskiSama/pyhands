import cv2

video = cv2.VideoCapture('http://192.168.1.3:4747/mjpegfeed')
ret = video.set(3,800)
ret = video.set(4,600)

while True:

    # Expand frame to [1, None, None, 3], to make it model friendly
    ret, frame = video.read()
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    cv2.imshow('Object detector', frame)

    if cv2.waitKey(1) == ord('q'):
        break

video.release()
cv2.destroyAllWindows()