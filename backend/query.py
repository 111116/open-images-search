from flask import request, Flask

app = Flask(__name__)


import csv

# class label -> imageID
classes = {} # mapping from label to list of imageID
with open('oidv6-train-annotations-human-imagelabels.csv') as f:
    next(f) # ignore title line
    csvreader = csv.reader(f)
    for imageid, src, label, confidence in csvreader:
        if confidence == '1':
            if not label in classes:
                classes[label] = []
            classes[label].append(imageid)

# class name -> class label
class_dict = {}
with open("oidv6-class-descriptions.csv","r") as f:
    next(f) # ignore title line
    csvreader = csv.reader(f)
    for label,name in csvreader:
        class_dict[name.lower()] = label

# imageID -> image URL
imageurl = {}
with open('oidv6-train-images-with-labels-with-rotation.csv','r') as f:
    next(f) # ignore title line
    for ImageID,Subset,OriginalURL,OriginalLandingURL,License,AuthorProfileURL,Author,Title,OriginalSize,OriginalMD5,Thumbnail300KURL,Rotation in csv.reader(f):
        if Subset == 'train':
            imageurl[ImageID] = [OriginalURL, OriginalLandingURL, Thumbnail300KURL]

# result formating
def image_lookup(imageid):
    urls = imageurl[imageid]
    return {
        'ImageID': imageid,
        'OriginalURL': urls[0],
        'OriginalLandingURL': urls[1],
        'Thumbnail300KURL': urls[2]
    }


@app.route("/query", methods=["GET"])
def search():
    query = request.args.get("q", "")
    if not query:
        return {"urllist": []}
    return {
        "urllist": list(map(image_lookup, classes[class_dict[query][0:100]]))
    }
