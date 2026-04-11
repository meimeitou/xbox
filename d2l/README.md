# D2L (pytorch)

## 资源

- github: <https://github.com/d2l-ai>
- 课程书籍: <https://d2l.ai/>
- 课程视频: <https://www.bilibili.com/video/BV1gyrPBPEzr>
- 课程： https://courses.d2l.ai/zh-v2/

## 环境准备

```bash
conda create -n d2l python=3.9 -y
conda activate d2l
pip install jupyter d2l torch torchvision

# 幻灯片插件
pip install rise
```

打开 jupter terminal 安装额外依赖：

```shell
# 下载Jupyter notebook
mkdir d2l-zh && cd d2l-zh
# ppt演示文档
git clone https://github.com/d2l-ai/d2l-pytorch-slides
# 课程代码
curl https://zh-v2.d2l.ai/d2l-zh-2.0.0.zip -o d2l-zh.zip
unzip d2l-zh.zip && rm d2l-zh.zip

jupyter lab --ip=0.0.0.0 --ServerApp.token='' --ServerApp.allow_origin='*'
```
