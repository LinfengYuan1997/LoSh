This repo contains the official PyTorch implementation of the **CVPR 2024** [paper](https://arxiv.org/abs/2306.08736): 

<div align="center">
<h1>
<b>
LoSh: Long-Short Text Joint Prediction Network for Referring Video Object Segmentation
</b>
</h1>
<h4>
<b>
Linfeng Yuan, Miaojing Shi, Zijie Yue, Qijun Chen
    
College of Electronic and Information Engineering, Tongji University
</b>
</h4>
</div>

<p align="center"><img src="docs/overview.png" width="800"/></p>

## Abstract
Referring video object segmentation (RVOS) aims to segment the target instance referred by a given text expression in a video clip. The text expression normally contains sophisticated description of the instance's appearance, action, and relation with others. It is therefore rather difficult for a RVOS model to capture all these attributes correspondingly in the video; in fact, the model often favours more on the action- and relation-related visual attributes of the instance. This can end up with partial or even incorrect mask prediction of the target instance. We tackle this problem by taking a subject-centric short text expression from the original long text expression. The short one retains only the appearance-related information of the target instance so that we can use it to focus the model's attention on the instance's appearance. We let the model make joint predictions using both long and short text expressions; and insert a long-short cross-attention module to interact the joint features and a long-short predictions intersection loss to regulate the joint predictions. Besides the improvement on the linguistic part, we also introduce a forward-backward visual consistency loss, which utilizes optical flows to warp visual features between the annotated frames and their temporal neighbors for consistency. We build our method on top of two state of the art pipelines. Extensive experiments on A2D-Sentences, Refer-YouTube-VOS, JHMDB-Sentences and Refer-DAVIS17 show impressive improvements of our method.

## Environment Installation

First, clone this repo to your PC or server: 

`git clone https://github.com/LinfengYuan1997/LoSh.git`

Then, create the virtual environment in Anaconda3:

`conda create -n losh python=3.9 pip -y`

`conda activate losh`

- Pytorch 1.10:

`conda install pytorch==1.10.0 torchvision==0.11.1 -c pytorch -c conda-forge`

- COCO API:

`pip install -U 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'`

- Additional required packages:

`pip install -r requirements.txt`

## Data Preparation

The setup of this repo follows [MTTR](https://github.com/mttr2021/MTTR), [Referformer](https://github.com/wjn922/ReferFormer), and [SgMg](https://github.com/bo-miao/SgMg).

Please refer to [data.md](docs/data.md) for data preparation.

## Training and Evaluation

### Train
All the models are trained using 4 Tesla A40 GPU with 48G GRAM. You can adjust the batch size or window size to adpat to your devices.

```
python main.py -rm train -c ${config_path} -ws 10 -bs 3 -ng 4
```

For example,
```
python main.py -rm train -c configs/a2d_sentences.yaml -ws 10 -bs 3 -ng 4
```

### Evaluate
```
python main.py -rm eval -c ${config_path} -ckpt ${ckpt_path} -ws 10 -bs 3 -ng 1
```

For example,
```
python main.py -rm eval -c configs/a2d_sentences.yaml -ckpt ./a2d_sentences.pth -ws 10 -bs 3 -ng 1
```



## Acknowledgements
This repo is based on the following repos, thanks for their fantastic work!
- [MTTR](https://github.com/mttr2021/MTTR)
- [Referformer](https://github.com/wjn922/ReferFormer)
- [SgMg](https://github.com/bo-miao/SgMg)

## Citation

```
@inproceedings{yuan2024losh,
  title={Losh: Long-short text joint prediction network for referring video object segmentation},
  author={Yuan, Linfeng and Shi, Miaojing and Yue, Zijie and Chen, Qijun},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={14001--14010},
  year={2024}
}
```
