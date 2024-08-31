# 单目标跟踪环境配置及实现 2024/8/23

## <1> MMTracking 工具包
内置单目标跟踪算法： 

- [SiamRPN++](https://arxiv.org/pdf/1812.11703) (CVPR 2019)
- [STARK](https://github.com/researchmm/Stark) (ICCV 2021)
- [MixFormer](https://github.com/MCG-NJU/MixFormer) (CVPR 2020)

### 【环境配置】-GPU 版本
#### 1、虚拟环境创建
```
conda create --name open-mmlab python=3.8 -y
conda activate open-mmlab
```
- Linux Ubuntu 20.04
- Python 3.8.19

#### 2、安装 Pytorch
- Pytorch 1.13.0 (torchaudio 0.13.0 | torchvision 0.14.0)

*查看 CUDA 版本，这里是 CUDA12.2，但是如果装对应版本的Pytorch，后续安装 mmcv 会出现版本不匹配问题，较为麻烦。**此处选择 CUDA11.7 版本对应的 Pytorch **。为与 mmcv 版本匹配，安装较低版本的 Pytorch 1.x*

```
conda install pytorch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 pytorch-cuda=11.7 -c pytorch -c nvidia
```

#### 3、安装 MMCV
兼容的 MMTracking，MMCV 和 MMDetection 版本如下，请安装正确的版本以避免安装问题。

| MMTracking version |        MMCV version        |     MMDetection version      |
| :----------------: | :------------------------: | :--------------------------: |
|       master       | mmcv-full>=1.3.17, \<2.0.0 | MMDetection>=2.19.1, \<3.0.0 |
|       0.14.0       | mmcv-full>=1.3.17, \<2.0.0 | MMDetection>=2.19.1, \<3.0.0 |
|       0.13.0       | mmcv-full>=1.3.17, \<1.6.0 | MMDetection>=2.19.1, \<3.0.0 |

GPU 版本 MMtracking，安装如下：
- mmcv-full 1.7.2
- mmdet 2.28.0

*注意不要安装成 mmcv 或者 mmcv-lite（会因为缺少函数而运行失败），如果 pip install一直卡在 Build Wheel，请到[官网](https://download.openmmlab.com/mmcv/dist/cu117/torch1.13.0/index.html)下载对应版本，以下网址中的 X 改成对应的 CUDA 版本以及 torch 版本(e.g. cu117, torch1.13.0)*
`https://download.openmmlab.com/mmcv/dist/cuXXX/torchX.XX.X/index.html`

安装：
```
pip install mmcv-full==1.7.2 -f https://download.openmmlab.com/mmcv/dist/cu117/torch1.13.0/index.html
pip install mmdet==2.28.0
```

#### 4、安装依赖
将 MMTracking 仓库克隆到本地：
```
git clone https://github.com/open-mmlab/mmtracking.git
cd mmtracking
```
安装依赖：*应该**安装 requirements.txt 中所有依赖**，这里只想推理不进行模型训练和评估，所以：*

在 demo_sot.py 起始位置设置环境变量，否则找不到文件，添加以下语句：

```python
# 设置环境变量
import sys
sys.path.append("../mmtracking")
```

直接无参数执行 demo 文件夹下的 demo_sot.py，遇到没有的第三方库，pip install 即可

直至命令行输出：
```
usage: demo_sot.py [-h] [--input INPUT] [--output OUTPUT] [--checkpoint CHECKPOINT] [--device DEVICE]
                   [--show] [--color COLOR] [--thickness THICKNESS] [--fps FPS]
                   [--gt_bbox_file GT_BBOX_FILE]
                   config
demo_sot.py: error: the following arguments are required: config
```
所有第三方库安装完毕！

### 【环境配置】-CPU版本
#### 1、虚拟环境创建
```
conda create --name open-mmlab python=3.8 -y
conda activate open-mmlab
```
- Windows 11
- Python 3.8.19

#### 2、安装 Pytorch-cpu
- Pytorch 1.9.1 (torchaudio 0.9.1 | torchvision 0.10.1)

试过高版本的 Pytorch 会出现版本不匹配问题，较为麻烦，保险起见，降低版本。

```
pip install torch==1.9.1 torchvision==0.10.1 torchaudio==0.9.1
```

#### 3、安装 MMCV
兼容的 MMTracking，MMCV 和 MMDetection 版本如下，请安装正确的版本以避免安装问题。

| MMTracking version |        MMCV version        |     MMDetection version      |
| :----------------: | :------------------------: | :--------------------------: |
|       master       | mmcv-full>=1.3.17, \<2.0.0 | MMDetection>=2.19.1, \<3.0.0 |
|       0.14.0       | mmcv-full>=1.3.17, \<2.0.0 | MMDetection>=2.19.1, \<3.0.0 |
|       0.13.0       | mmcv-full>=1.3.17, \<1.6.0 | MMDetection>=2.19.1, \<3.0.0 |

CPU 版本 MMtracking，安装如下：
- mmcv-full 1.7.2
- mmdet 2.28.0

*注意不要安装成 mmcv 或者 mmcv-lite（会因为缺少函数而运行失败），如果 pip install一直卡在 Build Wheel，请到[官网](https://download.openmmlab.com/mmcv/dist/cpu/torch1.9.0/index.html)下载对应版本，以下网址中的 X 改成对应的 torch 版本(e.g. torch1.9.0)*
`https://download.openmmlab.com/mmcv/dist/cpu/torchX.XX.X/index.html`

安装：
```
pip install mmcv-full==1.7.2 -f https://download.openmmlab.com/mmcv/dist/cpu/torch1.9.0/index.html
pip install mmdet==2.28.0
```

#### 4、安装依赖
同 GPU 版本

### 【调试以及实现】
#### 1、下载模型权重
在项目文件夹中的 `configs/sot` 文件夹中，有打包好的三种单目标跟踪模型，每个模型文件夹中有对应的 README.md 文件，里面有模型权重下载链接。为便于查找，现整理如下：

- MixFormer 
    - LaSOT [config](./mixformer_cvt_500e_lasot.py) | [model](https://download.openmmlab.com/mmtracking/sot/mixformer/mixformer_cvt_500e_lasot/mixformer_cvt_500e_lasot.pth)

    - TrackingNet [config](./mixformer_cvt_500e_trackingnet.py) | [model](https://download.openmmlab.com/mmtracking/sot/mixformer/mixformer_cvt_500e_lasot/mixformer_cvt_500e_lasot.pth)

    - GOT10k [config](./mixformer_cvt_500e_got10k.py) | [model](https://download.openmmlab.com/mmtracking/sot/mixformer/mixformer_cvt_500e_got10k/mixformer_cvt_500e_got10k.pth)

- SiamRPN++
    - LaSOT (SiamRPN++) [config](siamese_rpn_r50_20e_lasot.py) | [model](https://download.openmmlab.com/mmtracking/sot/siamese_rpn/siamese_rpn_r50_1x_lasot/siamese_rpn_r50_20e_lasot_20220420_181845-dd0f151e.pth)

    - LaSOT (SiamRPN++ FP16) [config](siamese_rpn_r50_fp16_20e_lasot.py) | [model](https://download.openmmlab.com/mmtracking/fp16/siamese_rpn_r50_fp16_20e_lasot_20220422_181501-ce30fdfd.pth)

    - UAV123 (SiamRPN++) [config](siamese_rpn_r50_20e_uav123.py) | [model](https://download.openmmlab.com/mmtracking/sot/siamese_rpn/siamese_rpn_r50_1x_uav123/siamese_rpn_r50_20e_uav123_20220420_181845-dc2d4831.pth)

    - TrackingNet (SiamRPN++) [config](siamese_rpn_r50_20e_trackingnet.py) | [model](https://download.openmmlab.com/mmtracking/sot/siamese_rpn/siamese_rpn_r50_1x_lasot/siamese_rpn_r50_20e_lasot_20220420_181845-dd0f151e.pth)

    - OTB100 (SiamRPN++) [config](siamese_rpn_r50_20e_otb100.py) | [model](https://download.openmmlab.com/mmtracking/sot/siamese_rpn/siamese_rpn_r50_1x_otb100/siamese_rpn_r50_20e_otb100_20220421_144232-6b8f1730.pth)

    - VOT2018 (SiamRPN++) [config](siamese_rpn_r50_20e_vot2018.py) | [model](https://download.openmmlab.com/mmtracking/sot/siamese_rpn/siamese_rpn_r50_1x_vot2018/siamese_rpn_r50_20e_vot2018_20220420_181845-1111f25e.pth)

- STARK
    - LaSOT (STARK-ST1) [config](stark_st1_r50_500e_lasot.py) | [model](https://download.openmmlab.com/mmtracking/sot/stark/stark_st1_r50_500e_lasot/stark_st1_r50_500e_lasot_20220414_185654-9c19e39e.pth)

    - LaSOT (STARK-ST2) [config](stark_st2_r50_50e_lasot.py) |   [model](https://download.openmmlab.com/mmtracking/sot/stark/stark_st2_r50_50e_lasot/stark_st2_r50_50e_lasot_20220416_170201-b1484149.pth)

    - TrackingNet (STARK-ST1) [config](stark_st1_r50_500e_lasot.py) | [model](https://download.openmmlab.com/mmtracking/sot/stark/stark_st1_r50_500e_lasot/stark_st1_r50_500e_lasot_20220414_185654-9c19e39e.pth)

    - TrackingNet (STARK-ST2) [config](stark_st2_r50_50e_lasot.py) | [model](https://download.openmmlab.com/mmtracking/sot/stark/stark_st2_r50_50e_lasot/stark_st2_r50_50e_lasot_20220416_170201-b1484149.pth)

    - GOT10k (STARK-ST1) [config](stark_st1_r50_500e_lasot.py) | [model](https://download.openmmlab.com/mmtracking/sot/stark/stark_st1_r50_500e_got10k/stark_st1_r50_500e_got10k_20220223_125400-40ead158.pth)

    - GOT10k (STARK-ST2) [config](stark_st2_r50_50e_lasot.py) | [model](https://download.openmmlab.com/mmtracking/sot/stark/stark_st2_r50_50e_got10k/stark_st2_r50_50e_got10k_20220226_124213-ee39bbff.pth)

在项目文件夹中新建文件夹 `checkpoints`，将下载好的模型权重放置到该文件夹中。

#### 2、demo 运行测试
```shell
python demo/demo_sot.py ${CONFIG_FILE} --input ${INPUT} --checkpoint ${CHECKPOINT_FILE} [--output ${OUTPUT}] [--device ${DEVICE}] [--show] [--gt_bbox_file ${GT_BBOX_FILE}]
```

`INPUT` 和 `OUTPUT` 支持 `mp4`视频格式和文件夹格式。

可选项:

- `OUTPUT`: 输出视频路径
- `DEVICE`: 可以是 `cpu` 或者 `cuda:0` 等等
- `--show`: 是否显示跟踪结果视频
- `--gt_bbox_file`: 初始跟踪框文件路径

例如:

假设已下载对应权重到 `checkpoints/` 路径中

```shell
python demo/demo_sot.py configs/sot/stark/stark_st1_r50_500e_got10k.py --device cpu --show --checkpoint checkpoint/stark_st1_r50_500e_got10k_20220223_125400-40ead158.pth --input demo/demo.mp4
```

#### 3、摄像头测试 camera_sot.py
使用摄像头进行跟踪，命令行参数举例，可以更换为其他模型比较效果：
```shell
python demo/camera_sot.py configs/sot/siamese_rpn/siamese_rpn_r50_20e_lasot.py --device cpu --show --checkpoint checkpoint/siamese_rpn_r50_20e_lasot_20220420_181845-dd0f151e.pth
```

`camera_sot.py` 中对视频格式进行转换，解决运行效果帧率较低的问题：
```python
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
```

这里可供调整读取摄像头分辨率，OpenCV 默认读取的摄像头分辨率为 640x480：
```python
# 设置分辨率
cap.set(3, 1920)
cap.set(4, 1080)
```

这里可供调整输出框的大小，有可能太小或者超出屏幕，按需调整：
```python
# 设置输出图像框大小
screen_width = 1920 / 2
screen_height = 1080 / 2
```

#### 4、SingleObjectTracker 类的封装
为更方便获取跟踪结果，以及调试代码，不再采用命令行参数输入形式。将跟踪算法封装为 `SingleObjectTracker` 类，方便调用。

- 构造函数：
```python
# 指定模型和设备 以及对应的 checkpoint
model : 单目标跟踪模型名称 e.g. 'mixformer' / 'siamese_rpn' / 'stark'
device: 设备名称          e.g. 'cuda:0' / 'cpu'
```

- `set_param` 方法：
```python
# 设置视频参数与显示
input_path   : 输入视频路径/摄像头                                        e.g. 'demo.mp4' / 0
screen_width : 若显示跟踪结果，显示窗口的宽度                              e.g. 1920
screen_height: 若显示跟踪结果，显示窗口的高度                              e.g. 1080
output_path  : 若将跟踪结果保存为视频, 则写路径, 否则为 None                e.g. 'result.mp4' / None
rsl_w        : 若输入为摄像头, 设置的分辨率                                e.g. 1920
rsl_h        : 若输入为摄像头, 设置的分辨率                                e.g. 1080
slow         : 若用双击选择 bounding box, 且输入为视频, 选择时原速/加速播放 e.g. True / False
```

- `fit` 方法：用于推理
- `mouse_callback` 方法：鼠标回调函数
- `track` 方法：
```python
# 实现跟踪结果的展示与输出
show            : 是否显示跟踪结果                  e.g. True / False
thickness       : 跟踪框厚度                       e.g. 3
bbox_output_path: 是否以文件形式输出跟踪框坐标       e.g. 'bbox_output.txt' / None
bbox_file       : 以文件形式/鼠标框选给出初始跟踪目标 e.g. 'bbox_input.txt' / None
# 返回一个 list 内容为跟踪框坐标 和 置信度
```

【P.S.】
- MixFormer 没有 CPU 推理版本，其余两种都有
- MixFormer 可以输出置信度，其余两种不能

【BUG】STARK 算法用 CPU 推理可以画出跟踪框，但是用 GPU 就不画不出来