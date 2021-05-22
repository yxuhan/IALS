import torch
import torch.nn as nn
from torchvision import transforms
import torch.optim as optim
import torch.nn.functional as F
from torchvision.utils import save_image

import argparse
import numpy as np
from PIL import Image
import os

from model.stylegan import get_stylegan
from model.vgg import vgg16


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--pretrain_root', type=str, default=r'./pretrain', help='path to the pretrain dir')
parser.add_argument('--truncation', type=float, default=0.5, help='truncation trick in stylegan')
parser.add_argument('--n_iters', type=int, default=1000, help='# of steps')
parser.add_argument('--dataset', type=str, default='ffhq', help='name of the face dataset [ffhq | celebahq]')
parser.add_argument('--img_path', default=r'image\real_face_sample.jpg', type=str, help='path for the real img')
parser.add_argument('--code_save_path', type=str, default='rec.npy', help='path for saving the reconstructed latent code')
parser.add_argument('--img_save_path', type=str, default='rec.jpg', help='path for saving the reconstructed img')


opt, _ = parser.parse_known_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

_, g_s = get_stylegan(os.path.join(opt.pretrain_root, "stylegan", opt.dataset, "weight.pkl"), device)
g_s.eval()

vgg = vgg16(os.path.join(opt.pretrain_root, "vgg", "imagenet_vgg16.pth"))
vgg.to(device)
vgg.eval()  # import vgg to compute perceptual loss

avg_code = np.load(os.path.join(opt.pretrain_root, "stylegan", opt.dataset, "avg_code.npy"))  # (512)
avg_code = torch.from_numpy(avg_code).unsqueeze(0).to(device)  # (1,512)

img = Image.open(opt.img_path)
resized_img = transforms.Resize((256, 256))(img)
img = transforms.Resize((1024, 1024))(img)

img = transforms.ToTensor()(img).unsqueeze(0).to(device)
img = 2 * img - 1

resized_img = transforms.ToTensor()(resized_img).unsqueeze(0).to(device)
resized_img = 2 * resized_img - 1
with torch.no_grad():
    conv1_1, conv1_2, conv3_2, conv4_2 = vgg(resized_img)

latent = avg_code.expand_as(torch.ones(1, 18, 512)).clone()
latent.requires_grad = True

criterion = nn.MSELoss(reduction='sum').to(device)
optimizer = optim.Adam([{'params': latent}], lr=0.01)

for iters in range(opt.n_iters):
    optimizer.zero_grad()
    pred = g_s(latent)
    resized_pred = F.avg_pool2d(pred, 4, 4)
    c11, c12, c32, c42 = vgg(resized_pred)
    l_img = criterion(img, pred)
    l_c11 = criterion(conv1_1, c11)
    l_c12 = criterion(conv1_2, c12)
    l_c32 = criterion(conv3_2, c32)
    l_c42 = criterion(conv4_2, c42)
    loss = l_img + l_c11 + l_c12 + l_c32 + l_c42
    # loss = criterion(img, pred)
    loss.backward()
    optimizer.step()
    print("\r[iter: %d/%d] [pixel-wise loss: %.2f]" % (iters, opt.n_iters, loss.item()), end='')

np.save(opt.code_save_path, latent[0].cpu().detach().numpy())
rec = g_s(latent).to('cpu')
save_image((rec+1)/2, opt.img_save_path)
