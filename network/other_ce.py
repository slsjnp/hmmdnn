import torch
import torch.nn as nn
from torchvision import models
import torch.distributions as dt
import torch.nn.functional as F

from functools import partial

nonlinearity = partial(F.relu, inplace=True)


# random.seed(seed)

# small image size 64x64

# additive fixation noise

# sample the initial ROI mean locations

# FOV mask - spatial size

# sample the initial scale locations

# FOV mask - scale size

# init input

# Gaussian noise

# sample fixations

# dense layer

# add fixation noise (saccade noise)

# sample fixation scale

### apply multi-scale masking for each fixation #########################

### feature extractors (shared weights) for each scale (2 layers) ###########

# for each scale, create the convolution layers
# use L2 regularization

# aggregate features over fixations

# multiscale flatten and concatenate into a giant feature vector

# face classifier (shared layers)
# use dense layers for the classifier
# apply the classifier to each aggregated feature map (for each fixation)
def gaussian_noise(DATA_augnoise, mean=0, std=0.1):
    sample = torch.empty(DATA_augnoise.shape, device=DATA_augnoise.device, dtype=DATA_augnoise.dtype).normal_(mean, std)
    return sample + DATA_augnoise


def _normal_sample(shape, device, dtype, loc, stddev):
    # tmp = dt.normal.Normal(loc=loc, scale=stddev)
    tmp = torch.empty(shape, device=device, dtype=dtype).normal_(loc, stddev)
    return tmp


def normal_samples1(r, loc, std):
    return _normal_sample((r.shape[0], 1), r.device, r.dtype, loc, std)


def normal_samplesx(r, loc, std):
    return _normal_sample((r.shape[0], 2), r.device, r.dtype, loc, std)


def normal_samples1_1(r, loc, std):
    return _normal_sample((r.shape[0], 1), r.device, r.dtype, loc, std)


def gauss_img_xys(x_std, s_std, myfactor, myscale):
    # Masking function; the input is the image and xys location of the Gaussian
    def mask_func(tensors):
        img, xy, s = tensors
        img_shape = img.shape
        # Make indexing arrays
        xx = torch.arange(img_shape[3], dtype=img.dtype).cuda()
        yy = torch.arange(img_shape[2], dtype=img.dtype).cuda()
        # Get coordinates
        x = xy[:, 0:1] * myfactor  # change to downscaled coordinates
        y = xy[:, 1:2] * myfactor
        # make scale multiplier
        dist_s = (myscale - s) / s_std
        # Make X and Y distances
        dist_x = (xx - x) / x_std
        dist_y = (yy - y) / x_std
        # Make full mask
        mask = torch.unsqueeze(dist_x * dist_x, 1) + torch.unsqueeze(dist_y * dist_y, 2) + torch.unsqueeze(
            dist_s * dist_s, 2)
        mask = torch.exp(-0.5 * mask)
        # Add channels dimension
        mask = torch.unsqueeze(mask, 1)
        # Multiply image and mask
        # mask = K.cast(mask, img.dtype)
        return img * mask

    return mask_func


class Conv2d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, strides=1, activation='relu', padding=0):
        super(Conv2d, self).__init__()
        if activation == 'relu':
            self.conv = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size, stride=strides, padding=padding),
                nn.ReLU(inplace=True),
            )

    def forward(self, x):
        return self.conv(x)


class Hmmcnn(nn.Module):
    def __init__(self,
                 in_ch,
                 out_ch,
                 std,
                 scale_factors,
                 numclasses,
                 numfix=3,
                 fixnoise_std=3.,  # additive noise to fixation locations
                 fix_mask_std=4.,  # fixation FOV (mask) size in standard deviationss
                 fix_scale_std=0.25,  # scale mask size in standard deviations

                 init_roistd=10.6,  # initial standard deviation of ROI: 20/120 = ?/64
                 init_scalestd=1.,  # initial standard deviation of scale of ROI

                 reg_centerbias=None,  # regularization weight for center bias
                 reg_scalebias=None,  # regularization weight for scale bias
                 reg_scalebias_center=None,  # the "center" for scale bias

                 dataaug_imgnoise=0.04,  # data augmentation: image noise std

                 ):
        super(Hmmcnn, self).__init__()

        # init param
        self.param = {
            "numclasses": numclasses,
            "arch_fc": (40,),
            "scale_factors": scale_factors,
            "fix_mask_std": fix_mask_std,
            "fixnoise_std": fixnoise_std,
            "fix_scale_std": fix_scale_std,
            "init_roistd": init_roistd, "init_scalestd": init_scalestd, "reg_centerbias": reg_centerbias,
            "reg_scalebias": reg_scalebias,
            "reg_scalebias_center": reg_scalebias_center,
            "dataaug_imgnoise": dataaug_imgnoise, "numfix": numfix,

        }
        self.MODEL_ms = len(scale_factors)

        # first fixation scale
        self.fix1_dense1 = nn.Linear(in_ch, 2)
        self.fix1_dense1_relu = nn.ReLU(inplace=True)
        self.fix1_dense2 = nn.Linear(in_ch, 1)
        self.fix1_dense2_relu = nn.ReLU(inplace=True)

        # second fixation scale
        self.fix2_dense1 = nn.Linear(in_ch, 2)
        self.fix2_dense1_relu = nn.ReLU(inplace=True)
        self.fix2_dense2 = nn.Linear(in_ch, 1)
        self.fix2_dense2_relu = nn.ReLU(inplace=True)

        # third fixation scale
        self.fix3_dense1 = nn.Linear(in_ch, 2)
        self.fix3_dense1_relu = nn.ReLU(inplace=True)
        self.fix3_dense2 = nn.Linear(in_ch, 1)
        self.fix3_dense2_relu = nn.ReLU(inplace=True)

        # img mask
        self.gauss_img_xys = []
        for scale in range(self.MODEL_ms):
            tmp = gauss_img_xys(self.param['fix_mask_std'], self.param['fix_scale_std'],
                                self.param['scale_factors'][scale], scale)
            self.gauss_img_xys.append(tmp)

        # conv 
        self.conv1_1 = Conv2d(in_ch=1, out_ch=8, kernel_size=3, activation='relu')
        self.conv1_2 = Conv2d(in_ch=1, out_ch=8, kernel_size=3, activation='relu')
        self.conv1_3 = Conv2d(in_ch=1, out_ch=8, kernel_size=3, activation='relu')

        # conv 
        self.conv2_1 = Conv2d(in_ch=8, out_ch=16, kernel_size=3, activation='relu')
        self.conv2_2 = Conv2d(in_ch=8, out_ch=16, kernel_size=3, activation='relu')
        self.conv2_3 = Conv2d(in_ch=8, out_ch=16, kernel_size=3, activation='relu')

        self.class_dense1 = nn.Linear(72448, self.param['arch_fc'][0])
        self.class_dense1_relu = nn.ReLU(inplace=True)

        self.class_dense2 = nn.Linear(72448, self.param['arch_fc'][0])
        self.class_dense2_relu = nn.ReLU(inplace=True)

        self.class_dense3 = nn.Linear(72448, self.param['arch_fc'][0])
        self.class_dense3_relu = nn.ReLU(inplace=True)

        self.final_dense1 = nn.Linear(self.param['arch_fc'][0], self.param['numclasses'])
        self.final_dense1_relu = nn.ReLU(inplace=True)

        self.final_dense2 = nn.Linear(self.param['arch_fc'][0], self.param['numclasses'])
        self.final_dense2_relu = nn.ReLU(inplace=True)

        self.final_dense3 = nn.Linear(self.param['arch_fc'][0], self.param['numclasses'])
        self.final_dense3_relu = nn.ReLU(inplace=True)

    def forward(self, input):

        input1, input2, input3 = input
        input = input1
        normal_s1 = normal_samples1(input, 0, 3.)
        fix1 = self.fix1_dense1(normal_s1)
        fix1 = self.fix1_dense1_relu(fix1)
        noise1 = normal_samplesx(input, 0, 3.)
        fix1n = fix1 + noise1
        normal_s1_1 = normal_samples1_1(input, 0, 3.)
        fix1s = self.fix1_dense2(normal_s1_1)
        fix1s = self.fix1_dense2_relu(fix1s)

        normal_s2 = normal_samples1(input, 0, 3.)
        fix2 = self.fix2_dense1(normal_s2)
        fix2 = self.fix2_dense1_relu(fix2)
        noise2 = normal_samplesx(input, 0, 3.)
        fix2n = fix2 + noise2
        normal_s2_1 = normal_samples1_1(input, 0, 3.)
        fix2s = self.fix2_dense2(normal_s2_1)

        normal_s3 = normal_samples1(input, 0, 3.)
        fix3 = self.fix3_dense1(normal_s3)
        fix3 = self.fix3_dense1_relu(fix3)
        noise3 = normal_samplesx(input, 0, 3.)
        fix3n = fix3 + noise3
        normal_s3_1 = normal_samples1_1(input, 0, 3.)
        fix3s = self.fix3_dense2(normal_s3_1)

        # apply multi-scale masking for each fixation
        # ?????????????????????
        input_img = [input1, input2, input3]
        # for scale in range(self.MODEL_ms):
        #     tmp = Input(shape=(1, self.param['bbox_ms'][scale][0], self.param['bbox_ms'][scale][1]))
        #     input_img.append(tmp)

        # gaussianNoise ??? dataloader?????????
        input_mask = []
        for scale in range(self.MODEL_ms):
            tmp = gaussian_noise(input_img[scale])
            input_mask.append(tmp)

        # gaussian mask
        # ??????????????? ???????????? ???9???
        mask1_img_ms = []
        mask2_img_ms = []
        mask3_img_ms = []
        for scale in range(self.MODEL_ms):
            tmp1 = self.gauss_img_xys[scale]((input_mask[scale], fix1n, fix1s))
            mask1_img_ms.append(tmp1)

            tmp2 = self.gauss_img_xys[scale]((input_mask[scale], fix2n, fix2s))
            mask2_img_ms.append(tmp2)

            tmp3 = self.gauss_img_xys[scale]((input_mask[scale], fix3n, fix3s))
            mask3_img_ms.append(tmp3)

        # fix0 1 2
        conv1_1_ms = []
        conv1_2_ms = []
        conv1_3_ms = []

        # ?????????????????????
        for scale in range(self.MODEL_ms):
            tmp1 = self.conv1_1(mask1_img_ms[scale])
            conv1_1_ms.append(tmp1)

            tmp2 = self.conv1_2(mask2_img_ms[scale])
            conv1_2_ms.append(tmp2)

            tmp3 = self.conv1_3(mask3_img_ms[scale])
            conv1_3_ms.append(tmp3)

        conv2_1_ms = []
        conv2_2_ms = []
        conv2_3_ms = []
        for scale in range(self.MODEL_ms):
            tmp1 = self.conv2_1(conv1_1_ms[scale])
            conv2_1_ms.append(tmp1)

            tmp2 = self.conv2_2(conv1_1_ms[scale])
            conv2_2_ms.append(tmp2)

            tmp3 = self.conv2_3(conv1_1_ms[scale])
            conv2_3_ms.append(tmp3)

        # aggregate remaining using max
        L_aggi_ms = [[None for i in range(len(self.param['scale_factors']))] for j in range(self.param['numfix'])]
        L_aggi_ms[0] = [conv2_1_ms[0], conv2_2_ms[1], conv2_3_ms[2]]
        for index in range(1, self.param['numfix']):
            for scale in range(len(self.param['scale_factors'])):
                L_aggi_ms[index][scale] = torch.max(eval('conv2_' + str(index) + '_ms')[scale],
                                                        eval('conv2_' + str(index + 1) + '_ms')[scale])

        # multiscale fatten and concatenate into a giant feature vector
        L_maski_F_ms = [[None for i in range(len(self.param['scale_factors']))] for j in range(self.param['numfix'])]
        L_maski_F = [None for i in range(self.param['numfix'])]
        for i in range(self.param['numfix']):
            for scale in range(len(self.param['scale_factors'])):
                L_maski_F_ms[i][scale] = torch.flatten(L_aggi_ms[i][scale], start_dim=1)
        for i in range(self.param['numfix']):
            if len(self.param['scale_factors']) == 1:
                L_maski_F[i] = L_maski_F_ms[i][0]
            else:
                L_maski_F[i] = torch.cat([L_maski_F_ms[i][j] for j in range(len(self.param['scale_factors']))], 1)

        # face classifier(share layers)
        fci = []
        for scale in range(len(self.param['scale_factors'])):
            d1 = eval('self.class_dense' + str(scale + 1))(L_maski_F[scale])
            d1_relu = eval('self.class_dense' + str(scale + 1) + '_relu')(d1)
            d2 = eval('self.final_dense' + str(scale + 1))(d1_relu)
            tmp = eval('self.final_dense' + str(scale + 1) + '_relu')(d2)
            tmp = torch.softmax(tmp, dim=1)
            fci.append(tmp)
        return fci