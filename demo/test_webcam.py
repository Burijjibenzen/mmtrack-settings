import cv2
import time

db_click = False

def mouse_callback(event, x, y, flags, param):
    global db_click
    if event == cv2.EVENT_LBUTTONDBLCLK:
        db_click = True

cv2.namedWindow("Output")
cv2.setMouseCallback("Output", mouse_callback)

# 调用usb摄像头
cap = cv2.VideoCapture('/home/szy/res.mp4')
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc("M", "J", "P", "G")) 

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

print(cap.get(cv2.CAP_PROP_FPS))

prev_time = time.time()

# 显示
while True:
    ret, frame = cap.read()

    # 计算帧率
    current_time = time.time()
    fps = 1 / (current_time - prev_time)
    prev_time = current_time

    # cv2.putText(frame, f'FPS: {fps:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Output", frame)

    if db_click:
        print("detected")
        break

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# 关闭
cap.release()
cv2.destroyAllWindows()