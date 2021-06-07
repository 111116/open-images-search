from flask import request, Flask
from os.path import exists
import pickle, csv

app = Flask(__name__)

if exists("classes.pkl"):
    classes = pickle.load(open("classes.pkl", "rb"))
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
    pickle.dump(classes, open("classes.pkl", "wb"))

if exists("class_dict.pkl"):
    class_dict = pickle.load(open("class_dict.pkl", "rb"))
else:
    # class name -> class label
    class_dict = {}
    with open("oidv6-class-descriptions.csv", "r") as f:
        next(f)  # ignore title line
        csvreader = csv.reader(f)
        for label, name in csvreader:
            class_dict[name.lower()] = label
    pickle.dump(class_dict, open("class_dict.pkl", "wb"))

if exists("imageurl.pkl"):
    imageurl = pickle.load(open("imageurl.pkl", "rb"))
else:
    # imageID -> image URL
    imageurl = {}
    with open("oidv6-train-images-with-labels-with-rotation.csv", "r") as f:
        next(f)  # ignore title line
        for (
            ImageID,
            Subset,
            OriginalURL,
            OriginalLandingURL,
            License,
            AuthorProfileURL,
            Author,
            Title,
            OriginalSize,
            OriginalMD5,
            Thumbnail300KURL,
            Rotation,
        ) in csv.reader(f):
            if Subset == "train":
                imageurl[ImageID] = [OriginalURL, OriginalLandingURL, Thumbnail300KURL]
    pickle.dump(imageurl, open("imageurl.pkl", "wb"))


# result formating
def image_lookup(imageid):
    urls = imageurl[imageid]
    return {
        "ImageID": imageid,
        "OriginalURL": urls[0],
        "OriginalLandingURL": urls[1],
        "Thumbnail300KURL": urls[2],
    }


@app.route("/query", methods=["GET"])
def search():
    query = request.args.get("q", "").lower()
    if not query:
        return {"urllist": []}
    return {"urllist": list(map(image_lookup, classes[class_dict[query]][0:100]))}

@app.route("/prompt", methods=["GET"])
def related_tags():
    query = request.args.get("q", "").lower()
    tags = []
    for label in class_dict:
        if label.startswith(query):
            tags.append(label)
    return {"results": list(map(lambda x:{'title':x}, tags[0:10]))}
