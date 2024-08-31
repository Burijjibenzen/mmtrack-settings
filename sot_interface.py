import os
import os.path as osp
import cv2
import mmcv
import time
import tempfile

from mmtrack.apis import inference_sot, init_model

class SingleObjectTracker:
    def __init__(self, model="mixformer", device="cuda:0"):
        '''
        ### 指定模型和设备 以及对应的 checkpoint
        
        model : 单目标跟踪模型名称 e.g. 'mixformer' / 'siamese_rpn' / 'stark'
        
        device: 设备名称         e.g. 'cuda:0' / 'cpu'
        '''
        if model == "mixformer":
            self.model_name = model
            self.config = 'configs/sot/mixformer/mixformer_cvt_500e_trackingnet.py'
            self.checkpoint = 'checkpoint/mixformer_cvt_500e_lasot.pth'
        elif model == "siamese_rpn":
            self.model_name = model
            self.config = 'configs/sot/siamese_rpn/siamese_rpn_r50_20e_lasot.py'
            self.checkpoint = 'checkpoint/siamese_rpn_r50_20e_lasot_20220420_181845-dd0f151e.pth'
        elif model == "stark":
            self.model_name = model
            self.config = 'configs/sot/stark/stark_st1_r50_500e_got10k.py'
            self.checkpoint = 'checkpoint/stark_st1_r50_500e_got10k_20220223_125400-40ead158.pth'
        else:
            raise Exception("Error: Incorrect Model Name \n Please Choose Model From [mixformer] [siamese_rpn] [stark]")
        
        if device == "cuda:0":
            self.device = device      # 指定设备
        elif device == "cpu":
            self.device = device
            if model == "mixformer":
                raise Exception("Error: MixFormer has no CPU version \n Support [cuda:0] Only")
        else:
            raise Exception("Error: Incorrect Device Name \n Please Device Between [cuda:0] [cpu]")
        
        self.db_click = False  # 是否双击
        self.chosen = False    # 是否选中
        self.posX = 0          # 鼠标位置 X
        self.posY = 0          # 鼠标位置 Y
        self.half_len = 20     # 选择框的大小
        self.res = []          # 结果
    
    def set_param(self, input_path, screen_width=1920/2, screen_height=1080/2, output_path=None, rsl_w=640, rsl_h=480, slow=True):
        '''
        ### 设置视频参数与显示

        input_path   : 输入视频路径/摄像头                                      e.g. 'demo.mp4' / 0

        screen_width : 若显示跟踪结果，显示窗口的宽度                             e.g. 1920
        
        screen_height: 若显示跟踪结果，显示窗口的高度                             e.g. 1080
        
        output_path  : 若将跟踪结果保存为视频, 则写路径, 否则为 None               e.g. 'result.mp4' / None
        
        rsl_w        : 若输入为摄像头, 设置的分辨率                              e.g. 1920
        
        rsl_h        : 若输入为摄像头, 设置的分辨率                              e.g. 1080
        
        slow         : 若用双击选择 bounding box, 且输入为视频, 选择时原速/加速播放 e.g. True / False
        '''
        self.cap = cv2.VideoCapture(input_path)

        self.screen_width = screen_width
        self.screen_height = screen_height
        self.slow = slow # 是否按原始帧率播放视频以供选择
        self.output_path = output_path

        # 查看原始的视频格式 YUYV的数据量较大，影响了摄像头的读取，占用的带宽很大
        fourcc = int(self.cap.get(cv2.CAP_PROP_FOURCC))
        codec = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])
        print(f"Original video format: {codec}")
        # 转换视频格式 MJPG
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('M', 'J', 'P', 'G'))
        # 查看转换好的视频格式 MJPG 占用小
        fourcc = int(self.cap.get(cv2.CAP_PROP_FOURCC))
        codec = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])
        print(f"Converted video format: {codec}")

        # 设置视频分辨率
        self.cap.set(3, rsl_w)
        self.cap.set(4, rsl_h)

        if not self.cap.isOpened():
            if input_path == 0:
                raise Exception("Error: Cannot Open Camera")
            else:
                raise Exception("Error: Cannot Open Video")

        # 是否要输出视频
        if self.output_path is not None:
            if self.output_path.endswith('.mp4'):
                self.out_dir = tempfile.TemporaryDirectory()
                self.out_path = self.out_dir.name
                _out = self.output_path.rsplit(os.sep, 1)
                if len(_out) > 1:
                    os.makedirs(_out[0], exist_ok=True)
            else:
                self.out_path = self.output_path
                os.makedirs(self.out_path, exist_ok=True)

        return
    
    def fit(self):
        '''
        ### 推理
        '''
        self.model = init_model(self.config, self.checkpoint, device=self.device)
        return

    def mouse_callback(self, event, x, y, flags, param):
        '''
        ### 鼠标回调函数
        '''
        if event == cv2.EVENT_LBUTTONDBLCLK:
            self.db_click = True
            # print(f'({x}, {y})')
            self.posX = x
            self.posY = y
    
    def track(self, show=True, thickness=3, bbox_output_path=None, bbox_file=None):
        '''
        ### 实现跟踪结果的展示与输出

        show            : 是否显示跟踪结果                 e.g. True / False
        
        thickness       : 跟踪框厚度                      e.g. 3
        
        bbox_output_path: 是否以文件形式输出跟踪框坐标       e.g. 'bbox_output.txt' / None
        
        bbox_file       : 以文件形式/鼠标框选给出初始跟踪目标 e.g. 'bbox_input.txt' / None
        
        返回一个 list 内容为跟踪框坐标 和 置信度
        '''
        prev_time = time.time() # 计算 fps 用
        frame_id = 0
        self.chosen = False
        cv2.namedWindow("Select")
        cv2.setMouseCallback("Select", self.mouse_callback)

        while True:
            ret, img = self.cap.read()
            if not ret:
                break

            # 计算并显示帧率
            current_time = time.time()
            fps = 1 / (current_time - prev_time)
            prev_time = current_time

            resized_frame = cv2.resize(img, (int(self.screen_width), int(self.screen_height)))
            cv2.putText(resized_frame, f'FPS: {fps:.2f}', (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

            if self.db_click and not self.chosen or frame_id == 0:
                if bbox_file is not None and frame_id == 0:
                    bboxes = mmcv.list_from_file(bbox_file)
                    init_bbox = list(map(float, bboxes[0].split(',')))
                    self.chosen = True
                    print('Initial Bounding Box: ', init_bbox)

                elif self.db_click and not self.chosen:
                    init_bbox = list((self.posX - self.half_len, self.posY - self.half_len, self.half_len * 2, self.half_len * 2))
                    cv2.rectangle(resized_frame, (self.posX - self.half_len, self.posY - self.half_len), (self.posX + self.half_len, self.posY + self.half_len), (0, 0, 255))
                    cv2.putText(resized_frame, "press c to continue", (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                    cv2.imshow("Select", resized_frame)
                    if cv2.waitKey(1000) & 0xFF == ord('c'):
                        # 将 (x1, y1, w, h) 转换为 (x1, y1, x2, y2)
                        init_bbox[2] += init_bbox[0]
                        init_bbox[3] += init_bbox[1]
                        self.db_click = False
                        self.chosen = True
                    # print(init_bbox)

            if self.chosen:
                result = inference_sot(self.model, resized_frame, init_bbox, frame_id=frame_id)

                self.res.append(result['track_bboxes'].tolist())

                if bbox_output_path is not None:
                    with open(bbox_output_path, 'a+') as f:
                        for i in result['track_bboxes']:
                            f.write(str(i))
                            f.write(' ')
                        f.write('\n')

                # 输出跟踪信息
                # print("TopLeftX:", result['track_bboxes'][0], "TopLeftY:", result['track_bboxes'][1], "RightBottomX:", result['track_bboxes'][2], "RightBottomY:", result['track_bboxes'][3])
                
                confidence = result['track_bboxes'][-1]
                if self.model_name == "mixformer":
                    cv2.putText(resized_frame, f'CONFIDENCE: {confidence * 100:.3f}%', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

                if self.output_path is not None:
                    out_file = osp.join(self.out_path, f'{frame_id:06d}.jpg')
                else:
                    out_file = None

                self.model.show_result(
                    resized_frame,
                    result,
                    show=show,
                    wait_time=1,
                    out_file=out_file,
                    thickness=thickness)

                frame_id += 1
            else:
                cv2.imshow("Select", resized_frame)
                if self.slow == True:
                    time.sleep(1 / self.cap.get(cv2.CAP_PROP_FPS))

            # 按 'q' 键退出循环
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()

        if self.output_path:
            print(
                f'\nmaking the output video at {self.output_path} with the captured frames')
            mmcv.frames2video(self.out_path, self.output_path, fps=30, fourcc='mp4v')
            self.out_dir.cleanup()

        if bbox_output_path:
            print('Bounding Box Results Has Been Written to ', bbox_output_path)

        return self.res