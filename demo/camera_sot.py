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

    # opencv 默认分辨率是 640 * 480 分辨率设置高后 会导致帧率降低
    # 或者直接拉伸图片用较低的分辨率 从而更改窗口大小

    cap = cv2.VideoCapture('/home/szy/test.mp4')
    # cap = cv2.VideoCapture(0)     # 打开摄像头

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
    cap.set(3, 1920)
    cap.set(4, 1080)

    prev_time = time.time() # 计算 fps 用

    if not cap.isOpened():
        print("Error: Cannot open camera")
        return
    
    # 设置输出图像框大小
    screen_width = 1920 / 2
    screen_height = 1080 / 2

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
    model = init_model(args.config, args.checkpoint, device=args.device)

    frame_id = 0
    while True:
        ret, img = cap.read()
        if not ret:
            break

        # 计算并显示帧率
        current_time = time.time()
        fps = 1 / (current_time - prev_time)
        prev_time = current_time
        cv2.putText(img, f'FPS: {fps:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        resized_frame = cv2.resize(img, (int(screen_width), int(screen_height)))

        if frame_id == 0:
            if args.gt_bbox_file is not None:
                bboxes = mmcv.list_from_file(args.gt_bbox_file)
                init_bbox = list(map(float, bboxes[0].split(',')))
            else:
                init_bbox = list(cv2.selectROI("Camera", resized_frame, False, False))
                # init_bbox = list(cv2.selectROI("Camera", img, False, False))

            # 将 (x1, y1, w, h) 转换为 (x1, y1, x2, y2)
            init_bbox[2] += init_bbox[0]
            init_bbox[3] += init_bbox[1]

        result = inference_sot(model, resized_frame, init_bbox, frame_id=frame_id)
        # result = inference_sot(model, img, init_bbox, frame_id=frame_id)

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
