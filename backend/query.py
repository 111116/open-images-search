from flask import request, Flask
from flask import redirect
from flask_cors import CORS, cross_origin
from os.path import exists
from collections import namedtuple
import pickle, csv, random, string

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


cachePool = {}


@app.route("/query", methods=["GET"])
@cross_origin()
def search():
    query = request.args.get("q", "").lower()
    if not query:
        return {"urllist": []}
    elif query in cachePool:
        return cachePool[query]
    elif not query in class_dict:
        return {"urllist": []}
    return {
        "urllist": list(
            map(
                lambda x: dict(imageinfo[x]._asdict()),
                classes[class_dict[query]][0:100],
            )
        )
    }


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


def process(filename):
    print(filename)
    pass


@app.route("/upload", methods=["POST"])
@cross_origin()
def search():
    token = "".join(
        random.choice(string.ascii_uppercase + string.digits) for _ in range(16)
    )
    f = request.files['file']
    sfname = 'images/'+token
    f.save(sfname)
    cachePool[token] = process(sfname)
    redirect(f"/result.html?q={token}")
