import cv2
from PIL import Image
import numpy as np
from skimage.io import imread

filepath = './liver.jpg'

cv2_im = cv2.imread(filepath)
print('cv2_im shape ', cv2_im.shape)  # (height, width, ch)

im = Image.open(filepath)
print('PIL image size', im.size)  # (width, height, ch)

pil_im = np.asarray(im)
print('pil_im shape ', pil_im.shape)  # (height, width, ch)

sk_im = imread(filepath)
print('sk_im shape', sk_im.shape)  # (height, width, ch)

# pydicom: dcm_slice = pydicom.dcmread(filename)
#          slice_data = dcm_slice.pixel_array   shape:(height, width)
#          若想更改原始像素size或其他操作，必须写回至原始二进制属性PixelData 中
#          dcm_slice.PixelData = dcm_slice.pixel_array.tobytes()
# SimpleITK: itk_im = sitk.ReadImage()  size:(width, height, depth)
#            np_im = sitk.GetArrayFromImage(itk_im) shape:(depth, height, width)

"""
还需要注意，对于opencv来说，无论读取灰度图还是彩图都是(H, W, 3)
的shape，灰度图的读取会把单通道复制三遍。因此，读取灰度图时得显示声明img = cv2.imread('gray.jpg', cv2.CV_LOAD_IMAGE_GRAYSCALE)
或者img = cv2.imread('gray.jpg', 0)
其中0代表灰度图，1
代表彩图，读出来的shape为（H, W)



1.当PixelSpacing存在且PixelSpacingCalibrationType存在且ImagerPixelSpacing存在的情况下，我们应该使用PixelSpacing作为图像渲染过程中以及后面的长度测量，面积测量和图像标尺的像素间距。此时应该注明CALIBRATED

2.当PixelSpacing不存在且ImagerPixelSpacing存在且EstimatedRadiographicMagnificationFactor存在的情况下，我们应该使用(ImagerPixelSpacing/EstimatedRadiographicMagnificationFactor)的结果做为像素间距。此时应该注明MAGNIFIED

3.当ImagerPixelSpacing存在且PixelSpacing不存在且EstimatedRadiographicMagnificationFactor不存在的情况下，我们应该使用ImagerPixelSpacing作为像素间距，此时应该注明DETECTOR


"""
