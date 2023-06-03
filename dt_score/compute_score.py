import sys
sys.path.append(".")
sys.path.append("..")

import argparse
import os 
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt

import torch 
from torchvision import transforms

from model.classifier import get_classifier


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--save_root', type=str, default="workspace/gender_exp", help='path to save the images')
parser.add_argument('--dt_save_path', type=str, default="dt_curve.png", help='path to save the DT curve images')
parser.add_argument('--step', type=float, default=0.1, help='step size for attribute variation')
parser.add_argument('--attr1', type=str, default='male', help='[smiling | male | young | pose | eyeglasses]')
parser.add_argument('--attr2', type=str, default='smiling', help='[smiling | male | young | pose | eyeglasses]')
parser.add_argument('--pretrain_root', type=str, default=r'./pretrain', help='path to the pretrain dir')

opt, _ = parser.parse_known_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

classifier_pri = get_classifier(os.path.join(opt.pretrain_root, "classifier", opt.attr1, "weight.pkl"), device)
classifier_pri.eval()
classifier_cond = get_classifier(os.path.join(opt.pretrain_root, "classifier", opt.attr2, "weight.pkl"), device)
classifier_cond.eval()

src_label = 1 - (opt.step > 0)

# compute classification score
attr1_label_list = []
attr2_label_list = []
for idx in tqdm(os.listdir(opt.save_root)):
    cur_root = os.path.join(opt.save_root, idx)
    img_list = []
    img_path_list = sorted(os.listdir(cur_root))
    for img_pth in img_path_list:
        cur_img_pth = os.path.join(cur_root, img_pth)
        img_list.append(transforms.ToTensor()(Image.open(cur_img_pth)))
    img_list = torch.stack(img_list, dim=0).to(device)
    
    attr1 = classifier_pri(img_list)  # [n,2]
    attr2 = classifier_cond(img_list)
    
    attr1_label = torch.argmax(attr1, dim=-1)  # [n]
    attr2_label = torch.argmax(attr2, dim=-1)

    if attr1_label[0].item() != src_label:
        continue

    attr1_label_list.append(attr1_label)
    attr2_label_list.append(attr2_label)

attr1_label_list = torch.stack(attr1_label_list, dim=0).float()  # [nimg,nstep]
attr2_label_list = torch.stack(attr2_label_list, dim=0).float()

transform_score = (attr1_label_list - attr1_label_list[..., :1]).abs().mean(dim=0)
disentangle_score = 1 - (attr2_label_list - attr2_label_list[..., :1]).abs().mean(dim=0)

# plot DT curve
plt.plot(transform_score.cpu().numpy(), disentangle_score.cpu().numpy())
plt.savefig(opt.dt_save_path)

# compute DT score (AUC)
area = 0
for i in range(len(transform_score) - 1):
    length = transform_score[i + 1] - transform_score[i]
    cur_area = (disentangle_score[i] + disentangle_score[i + 1]) * length / 2
    area += cur_area
print("DT score: %.4f" % area)
