import argparse
import os 
from tqdm import tqdm
import numpy as np 

import torch 
import torch.nn.functional as F
from torchvision.utils import save_image

from model.classifier import get_classifier
from model.stylegan import get_stylegan
from utils.utils import get_instance_specific_direction


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--pretrain_root', type=str, default=r'./pretrain', help='path to the pretrain dir')
parser.add_argument('--dataset', type=str, default='ffhq', help='name of the face dataset [ffhq | celebahq]')
parser.add_argument('--n_images', type=int, default=500, help='average instance-specific information on n_images')
parser.add_argument('--attr', type=str, default='male', help='[smiling | male | young | pose | eyeglasses]')
parser.add_argument('--truncation', type=float, default=0.5, help='truncation trick in stylegan')

opt, _ = parser.parse_known_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

classifier = get_classifier(os.path.join(opt.pretrain_root, "classifier", opt.attr, "weight.pkl"), device)
classifier.eval()

g_mapping, g_style = get_stylegan(os.path.join(opt.pretrain_root, "stylegan", opt.dataset, "weight.pkl"), device)
g_mapping.eval()
g_style.eval()

avg_code = np.load(os.path.join(opt.pretrain_root, "stylegan", opt.dataset, "avg_code.npy"))  # (512)
avg_code = torch.from_numpy(avg_code).unsqueeze(0).to(device)  # (1,512)

attr_level_direction = torch.zeros(1, 512).to(device)
target = torch.tensor([0]).to(device)

for i in tqdm(range(opt.n_images)):
    with torch.no_grad():
        z = torch.randn(1, 512).to(device)
        w = g_mapping(z)
        w = opt.truncation * w + (1 - opt.truncation) * avg_code
        origin_img = g_style(w)
        origin_img = (origin_img + 1) / 2
        origin_img = F.avg_pool2d(origin_img, 4, 4)
        pred_origin_img = classifier(origin_img)
        if torch.argmax(pred_origin_img, dim=1).item() == target.item():
            continue
    ins_specific_direction = get_instance_specific_direction(g_style, classifier, w, target, device)
    attr_level_direction += ins_specific_direction

attr_level_direction = - attr_level_direction / torch.norm(attr_level_direction)

##### TEST
with torch.no_grad():
    z = torch.randn(1, 512).to(device)
    w = g_mapping(z)
    w = opt.truncation * w + (1 - opt.truncation) * avg_code
    x = []
    for j in range(3):
        x.insert(0, w - j * 0.7 * attr_level_direction)
        x.append(w + j * 0.7 * attr_level_direction)
    x = torch.cat(x)
    origin_img = g_style(x)
    origin_img = (origin_img + 1) / 2
    origin_img = F.avg_pool2d(origin_img, 4, 4)
    save_image(origin_img, 'test.jpg')

attr_level_direction = attr_level_direction.to('cpu').numpy()
np.save(os.path.join(opt.pretrain_root, "attr_level_direction", "ours", opt.dataset, "%s.npy" % opt.attr), attr_level_direction)
