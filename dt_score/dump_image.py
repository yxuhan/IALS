import sys
sys.path.append(".")
sys.path.append("..")

import argparse
import os 
from tqdm import tqdm
import numpy as np 

import torch 
import torch.nn.functional as F
from torchvision.utils import save_image

from model.classifier import get_classifier
from model.stylegan import get_stylegan
from utils.utils import get_instance_specific_direction, orthogonalization, orthogonalization_all


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--n_images', type=int, default=100, help='num of images to compute DT metric')
parser.add_argument('--save_root', type=str, default="workspace", help='path to save the images')
parser.add_argument('--pretrain_root', type=str, default=r'./pretrain', help='path to the pretrain dir')
parser.add_argument('--truncation', type=float, default=0.5, help='truncation trick in stylegan')
parser.add_argument('--lambda1', type=float, default=0, help='weight for instance-specific information in primal attribute')
parser.add_argument('--lambda2', type=float, default=0.75, help='weight for instance-specific information in condition attribute')
parser.add_argument('--step', type=float, default=-0.15, help='step size for attribute variation')
parser.add_argument('--n_steps', type=int, default=15, help='# of steps')
parser.add_argument('--dataset', type=str, default='ffhq', help='name of the face dataset [ffhq | celebahq]')
parser.add_argument('--base', type=str, default='interfacegan', help='use the attribute level direction solved by [interfacegan | ganspace]')
parser.add_argument('--attr1', type=str, default='young', help='[smiling | male | young | pose | eyeglasses]')
parser.add_argument('--attr2', type=str, default='eyeglasses', help='[smiling | male | young | pose | eyeglasses]')

opt, _ = parser.parse_known_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

g_mapping, g_style = get_stylegan(os.path.join(opt.pretrain_root, "stylegan", opt.dataset, "weight.pkl"), device)
g_mapping.eval()
g_style.eval()

avg_code = np.load(os.path.join(opt.pretrain_root, "stylegan", opt.dataset, "avg_code.npy"))  # (512)
avg_code = torch.from_numpy(avg_code).unsqueeze(0).to(device)  # (1,512)

classifier_pri = get_classifier(os.path.join(opt.pretrain_root, "classifier", opt.attr1, "weight.pkl"), device)
classifier_pri.eval()
classifier_cond = get_classifier(os.path.join(opt.pretrain_root, "classifier", opt.attr2, "weight.pkl"), device)
classifier_cond.eval()

attr_level_dir_pri = torch.tensor(np.load(os.path.join(opt.pretrain_root, "attr_level_direction", opt.base, opt.dataset, "%s.npy" % opt.attr1)), dtype=torch.float).to(device)
attr_level_dir_cond = torch.tensor(np.load(os.path.join(opt.pretrain_root, "attr_level_direction", opt.base, opt.dataset, "%s.npy" % opt.attr2)), dtype=torch.float).to(device)

attr_level_dir_pri = attr_level_dir_pri / torch.norm(attr_level_dir_pri)
attr_level_dir_cond = attr_level_dir_cond / torch.norm(attr_level_dir_cond)

target = torch.tensor([1]).to(device)
for idx in tqdm(range(opt.n_images)):
    cur_save_root = os.path.join(opt.save_root, "%05d" % idx)
    os.makedirs(cur_save_root, exist_ok=True)

    with torch.no_grad():
        torch.manual_seed(idx * 1000)
        z = torch.randn(1, 512).to(device)
        w = g_mapping(z)
        w = opt.truncation * w + (1 - opt.truncation) * avg_code

    x = [w.clone()]
    for i in range(opt.n_steps):
        # render an instance-aware direction for primal attribute
        if opt.lambda1 == 1:
            ins_aware_dir_pri = attr_level_dir_pri
        else:
            ins_specific_dir_pri = get_instance_specific_direction(g_style, classifier_pri, w, target, device)
            ins_aware_dir_pri = opt.lambda1 * attr_level_dir_pri + (1-opt.lambda1) * ins_specific_dir_pri
        
        # render an instance-aware direction for condition attribute
        if opt.lambda2 == 1:
            ins_aware_dir_cond = attr_level_dir_cond
        else:
            ins_specific_dir_cond = get_instance_specific_direction(g_style, classifier_cond, w, target, device)
            ins_aware_dir_cond = opt.lambda2 * attr_level_dir_cond + (1-opt.lambda2) * ins_specific_dir_cond 

        # vector orthogonalization
        if opt.lambda2 == 1:
            ortho = orthogonalization(ins_aware_dir_pri, ins_aware_dir_cond)
        else:
            ortho = orthogonalization_all(ins_aware_dir_pri, ins_aware_dir_cond, attr_level_dir_cond)

        w += ortho * opt.step
        x.append(w.clone())

    with torch.no_grad():
        x = torch.cat(x)
        res = g_style(x)  # [n_step+1,3,1024,1024]
        res = F.avg_pool2d(res, 4, 4)  # [n_step+1,3,256,256]
        res = (res + 1) / 2
        for t in range(len(res)):
            save_image(res[t], os.path.join(cur_save_root, "%05d.png" % t))
