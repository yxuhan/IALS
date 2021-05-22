import torch 
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


def get_instance_specific_direction(g_style, classifier, w, target, device):
    '''
    compute the instance-specific semantic direction via backpropgating the classifier's gradient

    Step1: generate the image from the style code w
    Step2: downsample the StyleGAN generated image's resolution from 1024x1024 to 256x256
    Step3: compute the classification loss
    Step4: output the gradient of the style code as the instance-specific direction
    '''
    ins_specific_direction = torch.zeros(1, 512).to(device)
    ins_specific_direction.requires_grad = True

    optimizer = optim.SGD([{'params': ins_specific_direction}], lr=0.1)
    criterion_classify = nn.CrossEntropyLoss().to(device)
    
    # generate the image from the style code w
    img = g_style(w + ins_specific_direction.expand_as(w))
    img = (img + 1) / 2
    
    # downsample the resolution from 1024x1024 to 256x256
    img = F.avg_pool2d(img, 4, 4)
    
    # compute the classification loss
    pred = classifier(img)
    optimizer.zero_grad()
    loss = criterion_classify(pred, target)
    loss.backward()
    optimizer.step()

    ret = ins_specific_direction.detach()
    return ret / torch.norm(ret)


def orthogonalization(pri, cond):
    '''
    two vectors orthogonalization operation

    Step1: normalize pri
    Step2: normalize cond
    Step3: compute ret = pri- (pri, cond) pri where (,) is the inner product 
    '''
    pri = pri / torch.norm(pri)
    cond = cond / torch.norm(cond)
    ret = pri - torch.dot(pri[0], cond[0]) * cond
    return ret / torch.norm(ret)


def orthogonalization_all(pri, cond1, cond2):
    '''
    three vectors orthogonalization operation 
    please refer to https://github.com/genforce/interfacegan/blob/master/utils/manipulator.py#L181

    Step1: normalize pri, cond1 and cond2
    Step2: compute the closed-form solution of alpha and beta s.t. (pri, pri-alpha*cond1-beta*cond2)=0
    '''
    pri = pri / torch.norm(pri)
    cond1 = cond1 / torch.norm(cond1)
    cond2 = cond2 / torch.norm(cond2)

    pri_cond1 = torch.dot(pri[0], cond1[0])
    pri_cond2 = torch.dot(pri[0], cond2[0])

    cond1_cond2 = torch.dot(cond1[0], cond2[0])
    alpha = (pri_cond1 - pri_cond2 * cond1_cond2) / (1 - cond1_cond2 ** 2)
    beta = (pri_cond2 - pri_cond1 * cond1_cond2) / (1 - cond1_cond2 ** 2)
    ret = pri - alpha * cond1 - beta * cond2

    return ret / torch.norm(ret)
