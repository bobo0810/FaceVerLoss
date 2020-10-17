# FaceVerLoss
#### 收录到[PytorchNetHub](https://github.com/bobo0810/PytorchNetHub)

# 说明
- 基于标签分类的分类器
- 源码解读,便于理解
- 感谢各位大佬

### 汇总
|模型|原仓库|备注|来源|更新|
|:---:|:----:|:---:|:------:|:------:|
|[CosFace/AMSoftmax](https://arxiv.org/pdf/1801.05599.pdf)|[原地址](https://github.com/cavalleria/cavaface.pytorch)|乘法角间隔||2020.9|
|[ArcFace](https://arxiv.org/abs/1801.07698)|[原地址](https://github.com/cavalleria/cavaface.pytorch)|加法角间隔|CVPR2019|2020.9|
|[CircleLoss](https://arxiv.org/abs/2002.10857)|[原地址](https://github.com/xialuxi/CircleLoss_Face)|加权角间隔|CVPR2020 Oral|2020.9|

### 决策边界
|Loss|Decision Boundary|
|:---:|:----:|
|Softmax|![](https://github.com/bobo0810/FaceVerLoss/blob/master/imgs/Softmax.gif)|
|SphereFace|![](https://github.com/bobo0810/FaceVerLoss/blob/master/imgs/SphereFace.gif)|
|CosFace/AMSoftmax|![](https://github.com/bobo0810/FaceVerLoss/blob/master/imgs/CosFace.gif)|
|ArcFace|![](https://github.com/bobo0810/FaceVerLoss/blob/master/imgs/ArcFace.gif)|
|MV-Arc-Softmax|![](https://github.com/bobo0810/FaceVerLoss/blob/master/imgs/MV-Arc-Softmax.gif)|
|CurricularFace|![](https://github.com/bobo0810/FaceVerLoss/blob/master/imgs/CurricularFace.gif)|
___

# ArcFace

Paper伪代码
![](https://github.com/bobo0810/FaceVerLoss/blob/master/imgs/ArcFace.png)

# CircleLoss

待做：
- [ ] 公式推导
