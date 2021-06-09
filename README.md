A simple experimental search engine for [Open Image Dataset](https://storage.googleapis.com/openimages/web/index.html), featuring:
- user input prompt
- single / multiple class search
- search by image

Powered by: Semantic UI + Flask

The dataset required can be downloaded separately [here](https://storage.googleapis.com/openimages/web/download.html).

## 运行说明

首先下载此项目：

```bash
git clone https://github.com/111116/open-images-search
cd open-images-search
```

需在 [此处](https://storage.googleapis.com/openimages/web/download.html) 下载运行所需的数据集标注与索引，共约7G，并放在对应目录下：

```plain
./backend/oidv6-train-images-with-labels-with-rotation.csv
./backend/boxed/annotations/oidv6-train-annotations-bbox.csv
./backend/boxed/annotations/train-annotations-human-imagelabels-boxable.csv
./backend/boxed/annotations/class-descriptions-boxable.csv
./backend/oidv6-train-annotations-human-imagelabels.csv
./backend/oidv6-class-descriptions.csv
```

此外，为了使用图搜图功能，首先需要准备好图片，将图片放入 `./backend/data/medium/` 文件夹中，图片命名为 `图片ID.jpg`（如果使用 open images dataset 默认就是此格式），接着使用 `resnet.py` 进行预处理，预处理好的图片表示会存储在 `data/embedding_dict_new.pkl` 中。

安装依赖项：

```
pip install -r requirement.txt
```

使用 Caddy 启动前端

```
sudo caddy start
```

使用 Flask 启动后端

```
cd backend
FLASK_APP=query flask run --port 3002 --host 0.0.0.0 --no-reload
```

还需要修改前端代码中对应 API 的路径，对应服务器的 IP