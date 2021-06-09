import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import models
from torchvision.models.resnet import Bottleneck, BasicBlock, ResNet
from tqdm import tqdm
import torch.utils.model_zoo as model_zoo

import numpy as np
import imageio
import os
import pickle, imutils, cv2


# configs for histogram
RES_model = "resnet152"  # model type
pick_layer = "avg"  # extract feature of this layer
d_type = "d1"  # distance type

depth = 3  # retrieved depth, set to None will count the ap for whole database

""" MMAP
     depth
      depthNone, resnet152,avg,d1, MMAP 0.78474710149
      depth100,  resnet152,avg,d1, MMAP 0.819713653589
      depth30,   resnet152,avg,d1, MMAP 0.884925001919
      depth10,   resnet152,avg,d1, MMAP 0.944355078125
      depth5,    resnet152,avg,d1, MMAP 0.961788675194
      depth3,    resnet152,avg,d1, MMAP 0.965623938039
      depth1,    resnet152,avg,d1, MMAP 0.958696281702
      (exps below use depth=None)
      resnet34,avg,cosine, MMAP 0.755842698037
      resnet101,avg,cosine, MMAP 0.757435452078
      resnet101,avg,d1, MMAP 0.764556148137
      resnet152,avg,cosine, MMAP 0.776918319273
      resnet152,avg,d1, MMAP 0.78474710149
      resnet152,max,d1, MMAP 0.748099342614
      resnet152,fc,cosine, MMAP 0.776918319273
      resnet152,fc,d1, MMAP 0.70010267663
"""

means = (
    np.array([103.939, 116.779, 123.68]) / 255.0
)  # mean of three channels in the order of BGR


# from https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
model_urls = {
    "resnet18": "https://download.pytorch.org/models/resnet18-5c106cde.pth",
    "resnet34": "https://download.pytorch.org/models/resnet34-333f7ec4.pth",
    "resnet50": "https://download.pytorch.org/models/resnet50-19c8e357.pth",
    "resnet101": "https://download.pytorch.org/models/resnet101-5d3b4d8f.pth",
    "resnet152": "https://download.pytorch.org/models/resnet152-b121ed2d.pth",
}


class ResidualNet(ResNet):
    def __init__(self, model=RES_model, pretrained=True):
        if model == "resnet18":
            super().__init__(BasicBlock, [2, 2, 2, 2], 1000)
            if pretrained:
                self.load_state_dict(model_zoo.load_url(model_urls["resnet18"]))
        elif model == "resnet34":
            super().__init__(BasicBlock, [3, 4, 6, 3], 1000)
            if pretrained:
                self.load_state_dict(model_zoo.load_url(model_urls["resnet34"]))
        elif model == "resnet50":
            super().__init__(Bottleneck, [3, 4, 6, 3], 1000)
            if pretrained:
                self.load_state_dict(model_zoo.load_url(model_urls["resnet50"]))
        elif model == "resnet101":
            super().__init__(Bottleneck, [3, 4, 23, 3], 1000)
            if pretrained:
                self.load_state_dict(model_zoo.load_url(model_urls["resnet101"]))
        elif model == "resnet152":
            super().__init__(Bottleneck, [3, 8, 36, 3], 1000)
            if pretrained:
                self.load_state_dict(model_zoo.load_url(model_urls["resnet152"]))

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)  # x after layer4, shape = N * 512 * H/32 * W/32
        max_pool = torch.nn.MaxPool2d(
            (x.size(-2), x.size(-1)),
            stride=(x.size(-2), x.size(-1)),
            padding=0,
            ceil_mode=False,
        )
        Max = max_pool(x)  # avg.size = N * 512 * 1 * 1
        Max = Max.view(Max.size(0), -1)  # avg.size = N * 512
        avg_pool = torch.nn.AvgPool2d(
            (x.size(-2), x.size(-1)),
            stride=(x.size(-2), x.size(-1)),
            padding=0,
            ceil_mode=False,
            count_include_pad=True,
        )
        avg = avg_pool(x)  # avg.size = N * 512 * 1 * 1
        avg = avg.view(avg.size(0), -1)  # avg.size = N * 512
        fc = self.fc(avg)  # fc.size = N * 1000
        output = {"max": Max, "avg": avg, "fc": fc}
        return output


res_model = ResidualNet(model=RES_model)
if torch.cuda.is_available():
    res_model = res_model.cuda()


def prepare_image(file):
    img = imageio.imread(file, pilmode="RGB")
    # print(f"original_shape: {img.shape}")
    img = cv2.resize(img, (224, 224))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # print(f"resize_shape: {img.shape}")
    return img


def extract_feature(img):
    res_model.eval()
    img = img[:, :, ::-1]  # switch to BGR
    img = np.transpose(img, (2, 0, 1)) / 255.0
    img[0] -= means[0]  # reduce B's mean
    img[1] -= means[1]  # reduce G's mean
    img[2] -= means[2]  # reduce R's mean
    img = np.expand_dims(img, axis=0)
    if torch.cuda.is_available():
        inputs = torch.autograd.Variable(torch.from_numpy(img).cuda().float())
    else:
        inputs = torch.autograd.Variable(torch.from_numpy(img).float())
    d_hist = res_model(inputs)[pick_layer]
    d_hist = d_hist.data.cpu().numpy().flatten()
    d_hist /= np.sum(d_hist)  # normalize
    return d_hist


def extract_feature_batched(imgs):
    res_model.eval()
    inputs = []
    batch_size = len(imgs)
    for img in imgs:
        img = img[:, :, ::-1]  # switch to BGR
        img = np.transpose(img, (2, 0, 1)) / 255.0
        img[0] -= means[0]  # reduce B's mean
        img[1] -= means[1]  # reduce G's mean
        img[2] -= means[2]  # reduce R's mean
        img = np.expand_dims(img, axis=0)
        inputs.append(torch.autograd.Variable(torch.from_numpy(img).cuda().float()))
    inputs = torch.cat(inputs)
    d_hist = res_model(inputs)[pick_layer]
    d_hist = d_hist.data.view(batch_size, -1).cpu().numpy()
    d_hist /= np.sum(d_hist, axis=1)[:, None]  # normalize
    return d_hist


def single():
    embedding_dict = {}
    for label in os.listdir("../data/small/"):
        cnt, all = 0, 0
        for filename in tqdm(os.listdir(f"../data/small/{label}")):
            try:
                img_path = f"../data/small/{label}/{filename}"
                embedding = extract_feature(img_path)
                embedding_dict[filename.split(".")[0]] = embedding
                cnt += 1
            except:
                pass
            all += 1
        print(f"{cnt}/{all}")
        pickle.dump(embedding_dict, open("embedding_dict.pkl", "wb"))


def divide_chunks(l, n):
    for i in range(0, len(l), n):
        yield l[i : i + n]


def batch():
    embedding_dict = {}
    cnt = 0
    file_list = list(os.listdir("./data/medium/"))
    for chunk in tqdm(list(divide_chunks(file_list, 64))):
        img_chunks = [prepare_image(f"./data/medium/{item}") for item in chunk]
        hists = extract_feature_batched(img_chunks)
        for idx in range(len(hists)):
            embedding = hists[idx]
            embedding_dict[chunk[idx].split(".")[0]] = embedding
        cnt += 1
        if cnt % 10 == 0:
            pickle.dump(embedding_dict, open("embedding_dict_new.pkl", "wb"))
    pickle.dump(embedding_dict, open("embedding_dict_new.pkl", "wb"))


if __name__ == "__main__":
    batch()
