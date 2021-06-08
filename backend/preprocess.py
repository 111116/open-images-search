from flask import request, Flask
from flask_cors import CORS, cross_origin
from os.path import exists, join
from shutil import copyfile
import pickle, csv, os

DATA_PATH = "/data/se/train/"

class_dict = {}
label2name = {}
with open("boxed/annotations/class-descriptions-boxable.csv", "r") as f:
    csvreader = csv.reader(f)
    for label, name in csvreader:
        class_dict[name.lower()] = label
        label2name[label] = name.lower()

fetched_set = pickle.load(open(".cache/fetched_set.pkl", "rb"))

classes = {}
with open("boxed/annotations/train-annotations-human-imagelabels-boxable.csv") as f:
    next(f)
    csvreader = csv.reader(f)
    for imageid, src, label, confidence in csvreader:
        if confidence == "1":
            if not label in classes:
                classes[label] = []
            classes[label].append(imageid)

count = [(len(classes[key]), label2name[key]) for key in classes]

count.sort(key=lambda x: x[0], reverse=True)

# os.rmdir("data/small")
# os.mkdir("data/small")
# for _, label in count[:50]:
#     os.mkdir(f"data/small/{label}")
#     cnt = 0
#     for imgID in classes[class_dict[label]]:
#         imgPath = join(DATA_PATH, f"{imgID}.jpg")
#         if exists(imgPath):
#             copyfile(imgPath, f"data/small/{label}/{imgID}.jpg")
#             cnt += 1
#             if cnt >= 200:
#                 break

os.rmdir("data/big")
os.mkdir("data/big")
for _, label in count[:50]:
    os.mkdir(f"data/small/{label}")
    cnt = 0
    for imgID in classes[class_dict[label]]:
        imgPath = join(DATA_PATH, f"{imgID}.jpg")
        if exists(imgPath):
            copyfile(imgPath, f"data/small/{label}/{imgID}.jpg")
            cnt += 1
            if cnt >= 200:
                break
