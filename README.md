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

安装依赖项：

运行前端与后端：

