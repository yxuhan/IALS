import argparse
import os 
import numpy as np 

import torch 
import torch.nn.functional as F
from torchvision.utils import save_image

from model.stylegan import get_stylegan


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--pretrain_root', type=str, default=r'./pretrain', help='path to the pretrain dir')
parser.add_argument('--truncation', type=float, default=0.5, help='truncation trick in stylegan')
parser.add_argument('--step', type=float, default=0.4, help='step size for attribute variation')
parser.add_argument('--n_steps', type=int, default=5, help='# of steps')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--save_path', type=str, default='test.jpg', help='path to save editing result')
parser.add_argument('--dataset', type=str, default='ffhq', help='name of the face dataset [ffhq | celebahq]')
parser.add_argument('--attr', type=str, default='pose', help='[smiling | male | young | pose | eyeglasses]')
parser.add_argument('--base', type=str, default='interfacegan', help='use the attribute level direction solved by [interfacegan | ganspace]')
parser.add_argument('--real_image', type=int, choices=[0,1], default=0, help='edit real image or not, if true, you should specify the ;atent_code_path')
parser.add_argument('--latent_code_path', type=str, help='latent code path for real image')

opt, _ = parser.parse_known_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

g_mapping, g_style = get_stylegan(os.path.join(opt.pretrain_root, "stylegan", opt.dataset, "weight.pkl"), device)
g_mapping.eval()
g_style.eval()

avg_code = np.load(os.path.join(opt.pretrain_root, "stylegan", opt.dataset, "avg_code.npy"))  # (512)
avg_code = torch.from_numpy(avg_code).unsqueeze(0).to(device)  # (1,512)

attr_level_direction = torch.tensor(np.load(os.path.join(opt.pretrain_root, "attr_level_direction", opt.base, opt.dataset, "%s.npy" % opt.attr)), dtype=torch.float).to(device)
with torch.no_grad():    
    if opt.real_image == 1:
        w = torch.tensor(np.load(opt.latent_code_path), dtype=torch.float).to(device)  # (18,512)
        w.unsqueeze_(0)  # (1,18,512)
    else:
        torch.manual_seed(opt.seed)
        z = torch.randn(1, 512).to(device)
        w = g_mapping(z)
        w = opt.truncation * w + (1 - opt.truncation) * avg_code

    x = []

    # moving the latent code towards the semantic direction
    for j in range(opt.n_steps):
        x.append(w + j * opt.step * attr_level_direction)
    
    # generate and save the facial editing result
    x = torch.cat(x)
    origin_img = g_style(x)
    origin_img = (origin_img + 1) / 2
    origin_img = F.avg_pool2d(origin_img, 4, 4)
    save_image(origin_img, opt.save_path)
