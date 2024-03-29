from flask import request, Flask
from flask import redirect
from flask_cors import CORS, cross_origin
from os.path import exists
from collections import namedtuple
from PIL import Image
from resnet import extract_feature, prepare_image

import numpy as np
from numpy import dot
from numpy.linalg import norm
import pickle, csv, random, string
import hashlib
import time
import cv2, imutils, imageio
import re

app = Flask(__name__)
cors = CORS(app)
app.config["CORS_HEADERS"] = "Content-Type"

if exists(".cache/classes.pkl"):
    classes = pickle.load(open(".cache/classes.pkl", "rb"))
else:
    # class label -> imageID
    classes = {}  # mapping from label to list of imageID
    with open("oidv6-train-annotations-human-imagelabels.csv") as f:
        next(f)  # ignore title line
        csvreader = csv.reader(f)
        for imageid, src, label, confidence in csvreader:
            if confidence == "1":
                if not label in classes:
                    classes[label] = []
                classes[label].append(imageid)
    pickle.dump(classes, open(".cache/classes.pkl", "wb"))

if exists(".cache/class_dict.pkl"):
    class_dict = pickle.load(open(".cache/class_dict.pkl", "rb"))
else:
    # class name -> class label
    class_dict = {}
    with open("oidv6-class-descriptions.csv", "r") as f:
        next(f)  # ignore title line
        csvreader = csv.reader(f)
        for label, name in csvreader:
            class_dict[name.lower()] = label
    pickle.dump(class_dict, open(".cache/class_dict.pkl", "wb"))


if exists(".cache/fetched_set.pkl"):
    fetched_set = pickle.load(open(".cache/fetched_set.pkl", "rb"))
else:
    fetched_list = []
    with open("boxed/annotations/oidv6-train-annotations-bbox.csv") as f:
        next(f)
        for imageId, *misc in csv.reader(f):
            fetched_list.append(imageId)
    fetched_set = set(fetched_list)
    pickle.dump(fetched_set, open(".cache/fetched_set.pkl", "wb"))


ImageInfo = namedtuple(
    "ImageInfo",
    "ImageID Subset OriginalURL OriginalLandingURL License AuthorProfileURL Author Title OriginalSize OriginalMD5 Thumbnail300KURL Rotation Fetched",
)
if exists(".cache/imageinfo.pkl"):
    imageinfo = pickle.load(open(".cache/imageinfo.pkl", "rb"))
else:
    # imageID -> image info
    imageinfo = {}
    with open("oidv6-train-images-with-labels-with-rotation.csv", "r") as f:
        next(f)  # ignore title line
        for line in csv.reader(f):
            info = ImageInfo(*line, line[0] in fetched_set)
            if info.Subset == "train":
                imageinfo[info.ImageID] = info
    pickle.dump(imageinfo, open(".cache/imageinfo.pkl", "wb"))

cache_pool = {}
if exists(".cache/cache_pool.pkl"):
    imageinfo = pickle.load(open(".cache/cache_pool.pkl", "rb"))

embedding_dict = pickle.load(open("data/embedding_dict_new.pkl", "rb"))


def to_response(imgs):
    return {"urllist": list(map(lambda x: dict(imageinfo[x]._asdict()), imgs))}


@app.route("/query", methods=["GET"])
@cross_origin()
def search():
    query = request.args.get("q", "")
    if not query:
        return {"urllist": []}
    elif query in cache_pool:
        return cache_pool[query]
    elif not query.lower() in class_dict:
        queries = list(
            map(
                lambda s: s.strip(),
                re.split(
                    " in a | on a | with a | under a | and | with | on | under | in | a | an | the ",
                    " " + query.lower(),
                ),
            )
        )
        queries = list(filter(None, queries))
        for q in queries:
            if not q in class_dict:
                return {"urllist": []}
        results = list(
            set.intersection(*map(lambda x: set(classes[class_dict[x]]), queries))
        )
        return to_response(results[0:100])
    return to_response(classes[class_dict[query.lower()]][0:100])


@app.route("/prompt", methods=["GET"])
@cross_origin()
def related_tags():
    query = request.args.get("q", "").lower()
    tags = []
    for label in class_dict:
        if label.startswith(query):
            tags.append(label)
    if query in tags:
        tags.insert(0, tags.pop(tags.index(query)))
    return {"results": list(map(lambda x: {"title": x}, tags[0:10]))}


def distance(a, b):
    # (np.sum(np.absolute(embedding - embedding_dict[imgID])), imgID)
    return -dot(a, b) / (norm(a) * norm(b))


def retrive(embedding):
    return to_response(
        map(
            lambda x: x[1],
            sorted(
                [
                    (distance(embedding, embedding_dict[imgID]), imgID)
                    for imgID in embedding_dict
                ],
                key=lambda x: x[0],
            )[:50],
        )
    )


def process(file):
    hash = hashlib.sha224(file).hexdigest()
    start = time.time()
    embedding = extract_feature(prepare_image(file))
    print(f"feature_extract time: ", time.time() - start)
    start = time.time()
    result = retrive(embedding)
    print(f"retrive time: ", time.time() - start)
    return hash, result


@app.route("/upload", methods=["POST"])
@cross_origin()
def handleupload():
    token = "".join(
        random.choice(string.ascii_uppercase + string.digits) for _ in range(16)
    )
    f = request.files["file"]
    hash, result = process(f.read())
    cache_pool[hash] = result
    return redirect(f"/result.html?q={hash}")
