import torch


def IOU(input, target, device):
    if input.is_cuda:
        s = torch.FloatTensor(1).to(device).zero_()
    else:
        s = torch.FloatTensor(1).zero_()

    eps = 0.0001
    for i, c in enumerate(zip(input, target)):
        inter = torch.dot(input.view(-1), target.view(-1))  # 求交集，数据全部拉成一维 然后就点积
        union = torch.sum(input) + torch.sum(target) + eps  # 就并集，数据全部求和后相加，eps防止分母为0

        iou = (inter.float() + eps) / (union.float() - inter.float())  # iou计算公式，交集比上并集

        s += iou

    return s / (i + 1)