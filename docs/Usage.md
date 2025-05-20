# Usage

###### 环境配置

torch2.6.0+cu124

主要内容：在 requirements.txt 里

```
!pip install -r requirements.txt
```

```
numpy==1.24.4
opencv-python
loguru
scikit-image
scikit-learn
tqdm
torchvision>=0.10.0
Pillow
thop
ninja
tabulate
lap
motmetrics
filterpy
h5py
matplotlib
scipy
prettytable
easydict
pyyaml
yacs
termcolor
gdown
cython
cython_bbox
faiss-cpu
ultralytics
onnx==1.15.0
onnx-simplifier
onnxruntime==1.15.1
insightface==0.7.3
tensorboard
```

为了能让版本适配，能够正常运行，需要再运行下面内容

```python
!pip uninstall tensorflow tensorboard jax jaxlib -y
!pip install tensorboard==2.12.0
```



###### 源代码更改

如果是你要从github上下载源代码的话，需要进行下面操作

```python
!grep -rl 'from collections import Mapping' /content/BoT-SORT/ | xargs sed -i 's/from collections import Mapping/from collections.abc import Mapping/g'
!sed -i 's/torch.load(w, map_location=map_location)/torch.load(w, map_location=map_location, weights_only=False)/g' /content/BoT-SORT/yolov7/models/experimental.py
!grep -rl 'dtype=np.float' /content/BoT-SORT/ | xargs sed -i 's/dtype=np.float/dtype=float/g'
```



###### 命令行运行

```python
# 训练 few-shot 部分
!python few_shot/proto.py
```

```python
# 图片测试部分
!python tools/yolov8_seg.py --weights yolov8s-seg.pt --source TEST/ooooutput/
```

```python
# 运行对视频的全部处理
!python tools/mc_demo_yolov8.py --weights yolov8s-seg.pt --source shoopingmall.mp4 --device cuda --project runs/detect --name exp1 --with-reid
```

