# Research

本仓库整理了视频情感识别方向的研究代码、数据集入口与论文材料，主要围绕用户生成视频中的多模态情感识别任务展开。项目包含两个模型实验目录：`DPFNet` 与 `PE-CLIP`，并配套提供 VideoEmotion-8 等数据集链接、数据预处理脚本、训练/测试入口和相关论文 PDF。

## 主要内容

- `DPFNet/`：双通路融合网络相关实验代码，包含模型定义、数据加载、损失函数、训练、测试和视频/音频预处理工具。
- `PE-CLIP/`：PE-CLIP 相关实验代码，包含原型演化/文本向量辅助的视频情感识别实现，以及训练、验证和测试流程。
- `Dataset/`：常用视频情感识别数据集入口，包括 MusicVideo-6、VideoEmotion-8 和 Ekman-6。
- `Paper/`：本项目相关论文与研究材料。

## 目录结构

```text
Research/
├── DPFNet/        # DPFNet 模型代码与实验脚本
├── PE-CLIP/       # PE-CLIP 模型代码与实验脚本
├── Dataset/       # 数据集下载链接与说明
├── Paper/         # 论文 PDF
├── README.md      # 项目总览
└── .gitignore
```

## 环境依赖

两个子项目都提供了独立的 `requirements.txt`。建议为每个实验创建独立 Python 环境后安装依赖：

```bash
cd DPFNet
pip install -r requirements.txt
```

或：

```bash
cd PE-CLIP
pip install -r requirements.txt
```

项目预处理还依赖 FFmpeg，用于视频抽帧和音频提取。

## 数据准备

抽帧处理过的图片数据集链接见 `Dataset/README.md`。
音频模态需要下载原始视频进行提取。
以 VideoEmotion-8 为例，下载原始视频后可使用各子项目 `tools/` 目录下的脚本完成预处理：

```bash
python tools/video2jpg.py
python tools/n_frames.py
python tools/ve8_json.py
python tools/video2mp3.py
```

默认代码假设数据目录大致如下：

```text
VideoEmotion8/
├── VideoEmotion8--imgs/
├── VideoEmotion8--mp3/
└── ve8_01.json
```

运行前请根据本机路径修改对应子项目中的 `opts.py`，尤其是 `root_path`、`video_path`、`audio_path`、`annotation_path`、`result_path` 和预训练权重路径。

## 运行方式

进入对应模型目录后运行：

```bash
python main.py
```

如需单独训练、预训练、验证或测试，可查看对应目录中的：

- `train.py`
- `pretrain.py`
- `pretrain1.py`
- `validation.py`
- `test.py`

## 说明
大体积数据、抽帧结果、音频文件和模型权重未纳入 Git 管理。若需要共享权重文件，邮件联系2920217477@qq.com。
