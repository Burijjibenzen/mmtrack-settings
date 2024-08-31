import os
import os.path as osp
import cv2
import mmcv
import time
import tempfile

# 设置环境变量
import sys
sys.path.append('../mmtracking')

from mmtrack.apis import inference_sot, init_model
from argparse import ArgumentParser

############## Settings ##############
# 路径
path = '/home/szy/test.mp4'
# path = 0
# 按原始帧率显示已有视频
slow = True 

# 摄像头分辨率
rsl_w = 1920 
rsl_h = 1080

# 选择框的大小
half_len = 20 

# 设置输出图像框大小
screen_width = 1920 / 2
screen_height = 1080 / 2
######################################

db_click = False
posX = 0
posY = 0

def mouse_callback(event, x, y, flags, param):
    global db_click, posX, posY
    if event == cv2.EVENT_LBUTTONDBLCLK:
        db_click = True
        print(f'({x}, {y})')
        posX = x
        posY = y
    # return

def main():
    parser = ArgumentParser()
    parser.add_argument('config', help='Config file')
    parser.add_argument('--output', help='output video file (mp4 format)')
    parser.add_argument('--checkpoint', help='Checkpoint file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--show',
        action='store_true',
        default=True,
        help='whether to show visualizations.')
    parser.add_argument(
        '--color', default=(0, 255, 0), help='Color of tracked bbox lines.')
    parser.add_argument(
        '--thickness', default=3, type=int, help='Thickness of bbox lines.')
    parser.add_argument('--gt_bbox_file', help='The path of gt_bbox file')
    args = parser.parse_args()

    global chosen, db_click

    # opencv 默认分辨率是 640 * 480 分辨率设置高后 会导致帧率降低
    # 或者直接拉伸图片用较低的分辨率 从而更改窗口大小

    cap = cv2.VideoCapture(path)

    # 查看原始的视频格式 YUYV的数据量较大，影响了摄像头的读取，占用的带宽很大
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    codec = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])
    print(f"Original video format: {codec}")

    # 转换视频格式 MJPG
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('M', 'J', 'P', 'G'))
    
    # 查看转换好的视频格式 MJPG 占用小
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    codec = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])
    print(f"Converted video format: {codec}")

    # 设置视频分辨率
    cap.set(3, rsl_w)
    cap.set(4, rsl_h)

    prev_time = time.time() # 计算 fps 用

    if not cap.isOpened():
        print("Error: Cannot open camera")
        return

    # 输出设置
    OUT_VIDEO = False
    if args.output is not None:
        if args.output.endswith('.mp4'):
            OUT_VIDEO = True
            out_dir = tempfile.TemporaryDirectory()
            out_path = out_dir.name
            _out = args.output.rsplit(os.sep, 1)
            if len(_out) > 1:
                os.makedirs(_out[0], exist_ok=True)
        else:
            out_path = args.output
            os.makedirs(out_path, exist_ok=True)

    # 从 config 和 checkpoint 构建模型
    # print(args.config)
    model = init_model(args.config, args.checkpoint, device=args.device)

    cv2.namedWindow("Select")
    cv2.setMouseCallback("Select", mouse_callback)

    frame_id = 0
    chosen = False
    while True:
        ret, img = cap.read()
        if not ret:
            break

        # 计算并显示帧率
        current_time = time.time()
        fps = 1 / (current_time - prev_time)
        prev_time = current_time

        resized_frame = cv2.resize(img, (int(screen_width), int(screen_height)))
        cv2.putText(resized_frame, f'FPS: {fps:.2f}', (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        if db_click and not chosen:
            if args.gt_bbox_file is not None:
                bboxes = mmcv.list_from_file(args.gt_bbox_file)
                init_bbox = list(map(float, bboxes[0].split(',')))
            else:
                init_bbox = list((posX - half_len, posY - half_len, half_len * 2, half_len * 2))
                cv2.rectangle(resized_frame, (posX - half_len, posY - half_len), (posX + half_len, posY + half_len), (0, 0, 255))
                cv2.putText(resized_frame, "press c to continue", (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                cv2.imshow("Select", resized_frame)
                if cv2.waitKey(1000) & 0xFF == ord('c'):
                    # 将 (x1, y1, w, h) 转换为 (x1, y1, x2, y2)
                    init_bbox[2] += init_bbox[0]
                    init_bbox[3] += init_bbox[1]
                    db_click = False
                    chosen = True

            print(init_bbox)

        if chosen:
            result = inference_sot(model, resized_frame, init_bbox, frame_id=frame_id)
            # result = inference_sot(model, img, init_bbox, frame_id=frame_id)

            # 输出跟踪信息
            print("TopLeftX:", result['track_bboxes'][0], "TopLeftY:", result['track_bboxes'][1], "RightBottomX:", result['track_bboxes'][2], "RightBottomY:", result['track_bboxes'][3])
            confidence = result['track_bboxes'][-1]
            cv2.putText(resized_frame, f'CONFIDENCE: {confidence * 100:.3f}%', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            model.show_result(
                resized_frame,
                # img,
                result,
                show=args.show,
                wait_time=1,
                thickness=args.thickness)

            if args.output is not None:
                if OUT_VIDEO:
                    out_file = osp.join(out_path, f'{frame_id:06d}.jpg')
                else:
                    out_file = None  # 不保存每帧的图片
            frame_id += 1
        else:
            cv2.imshow("Select", resized_frame)
            if slow == True:
                time.sleep(1 / cap.get(cv2.CAP_PROP_FPS))

        # 按 'q' 键退出循环
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    if args.output and OUT_VIDEO:
        print(
            f'\nmaking the output video at {args.output} with the captured frames')
        mmcv.frames2video(out_path, args.output, fps=30, fourcc='mp4v')
        out_dir.cleanup()

if __name__ == '__main__':
    main()
