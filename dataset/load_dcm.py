import glob
import os
import SimpleITK as sitk
import cv2
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O(e.g. pd.read_csv)
import pydicom
import pydicom.uid
# tkinter is needed to before PIL.Image
from tkinter.filedialog import *
import PIL.Image as Image
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
# from pydicom.pixel_data_handlers.gdcm_handler import get_pixeldata
from scipy import ndimage as ndi
from matplotlib import pyplot as plt
from skimage import measure

# from pylab import *


"""Hounsfield Unit = pixel_value * rescale_slope + rescale_intercept"""
join = os.path.join
path = "/home/sj/workspace/data/chao_data/Train_Sets/CT/1/DICOM_anon/"

have_numpy = True

try:
    import numpy
except ImportError:
    have_numpy = False
    raise

sys_is_little_endian = (sys.byteorder == 'little')

NumpySupportedTransferSyntaxes = [
    pydicom.uid.ExplicitVRLittleEndian,
    pydicom.uid.ImplicitVRLittleEndian,
    pydicom.uid.DeflatedExplicitVRLittleEndian,
    pydicom.uid.ExplicitVRBigEndian,
]


# 支持"传输"语法
def supports_transfer_syntax(dicom_dataset):
    return (dicom_dataset.file_meta.TransferSyntaxUID in
            NumpySupportedTransferSyntaxes)


def needs_to_convert_to_RGB(dicom_dataset):
    return False


def should_change_PhotometricInterpretation_to_RGB(dicom_dataset):
    return False


def get_pixeldata(dicom_dataset):
    """If NumPy is available, return an ndarray of the Pixel Data.
    Raises
    ------
    TypeError
        If there is no Pixel Data or not a supported data type.
    ImportError
        If NumPy isn't found
    NotImplementedError
        if the transfer syntax is not supported
    AttributeError
        if the decoded amount of data does not match the expected amount
    Returns
    -------
    numpy.ndarray
       The contents of the Pixel Data element (7FE0,0010) as an ndarray.
    """
    if (dicom_dataset.file_meta.TransferSyntaxUID not in
            NumpySupportedTransferSyntaxes):
        raise NotImplementedError("Pixel Data is compressed in a "
                                  "format pydicom does not yet handle. "
                                  "Cannot return array. Pydicom might "
                                  "be able to convert the pixel data "
                                  "using GDCM if it is installed.")

    if not have_numpy:
        msg = ("The Numpy package is required to use pixel_array, and "
               "numpy could not be imported.")
        raise ImportError(msg)
    if 'PixelData' not in dicom_dataset:
        raise TypeError("No pixel data found in this dataset.")

    # Make NumPy format code, e.g. "uint16", "int32" etc
    # from two pieces of info:
    # dicom_dataset.PixelRepresentation -- 0 for unsigned, 1 for signed;
    # dicom_dataset.BitsAllocated -- 8, 16, or 32
    if dicom_dataset.BitsAllocated == 1:
        # single bits are used for representation of binary data
        format_str = 'uint8'
    elif dicom_dataset.PixelRepresentation == 0:
        format_str = 'uint{}'.format(dicom_dataset.BitsAllocated)
    elif dicom_dataset.PixelRepresentation == 1:
        format_str = 'int{}'.format(dicom_dataset.BitsAllocated)
    else:
        format_str = 'bad_pixel_representation'
    try:
        numpy_dtype = numpy.dtype(format_str)
    except TypeError:
        msg = ("Data type not understood by NumPy: "
               "format='{}', PixelRepresentation={}, "
               "BitsAllocated={}".format(
            format_str,
            dicom_dataset.PixelRepresentation,
            dicom_dataset.BitsAllocated))
        raise TypeError(msg)

    if dicom_dataset.is_little_endian != sys_is_little_endian:
        numpy_dtype = numpy_dtype.newbyteorder('S')

    pixel_bytearray = dicom_dataset.PixelData

    if dicom_dataset.BitsAllocated == 1:
        # if single bits are used for binary representation, a uint8 array
        # has to be converted to a binary-valued array (that is 8 times bigger)
        try:
            pixel_array = numpy.unpackbits(
                numpy.frombuffer(pixel_bytearray, dtype='uint8'))
        except NotImplementedError:
            # PyPy2 does not implement numpy.unpackbits
            raise NotImplementedError(
                'Cannot handle BitsAllocated == 1 on this platform')
    else:
        pixel_array = numpy.frombuffer(pixel_bytearray, dtype=numpy_dtype)
    length_of_pixel_array = pixel_array.nbytes
    expected_length = dicom_dataset.Rows * dicom_dataset.Columns
    if ('NumberOfFrames' in dicom_dataset and
            dicom_dataset.NumberOfFrames > 1):
        expected_length *= dicom_dataset.NumberOfFrames
    if ('SamplesPerPixel' in dicom_dataset and
            dicom_dataset.SamplesPerPixel > 1):
        expected_length *= dicom_dataset.SamplesPerPixel
    if dicom_dataset.BitsAllocated > 8:
        expected_length *= (dicom_dataset.BitsAllocated // 8)
    padded_length = expected_length
    if expected_length & 1:
        padded_length += 1
    if length_of_pixel_array != padded_length:
        raise AttributeError(
            "Amount of pixel data %d does not "
            "match the expected data %d" %
            (length_of_pixel_array, padded_length))
    if expected_length != padded_length:
        pixel_array = pixel_array[:expected_length]
    if should_change_PhotometricInterpretation_to_RGB(dicom_dataset):
        dicom_dataset.PhotometricInterpretation = "RGB"
    if dicom_dataset.Modality.lower().find('ct') >= 0:  # CT图像需要得到其CT值图像
        pixel_array = pixel_array * dicom_dataset.RescaleSlope + dicom_dataset.RescaleIntercept  # 获得图像的CT值
    pixel_array = pixel_array.reshape(dicom_dataset.Rows, dicom_dataset.Columns * dicom_dataset.SamplesPerPixel)
    return pixel_array, dicom_dataset.Rows, dicom_dataset.Columns


def setDicomWinWidthWinCenter(img_data, winwidth, wincenter, rows, cols):
    # if can not set 'writeable = True', make a new np.array()
    img_temp = np.array(img_data)
    img_temp.flags.writeable = True
    min = (2 * wincenter - winwidth) / 2.0 + 0.5
    max = (2 * wincenter + winwidth) / 2.0 + 0.5
    dFactor = 255.0 / (max - min)

    for i in numpy.arange(rows):
        for j in numpy.arange(cols):
            img_temp[i, j] = int((img_temp[i, j] - min) * dFactor)

    min_index = img_temp < min
    img_temp[min_index] = 0
    max_index = img_temp > max
    img_temp[max_index] = 255

    return img_temp


def load_scan(path):
    # path_list = os.listdir(path)
    slices = [pydicom.read_file(join(path, s)) for s in os.listdir(path)]
    # slices is sorted by ImagePositionPatient
    slices.sort(key=lambda x: int(x.ImagePositionPatient[2]))
    try:
        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
    except:
        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)
    for s in slices:
        s.SliceThickness = slice_thickness
    return slices


def multi_dcm(path, save_path):
    path_list = os.listdir(path)
    re_pattern = re.compile(r'(\d+)')
    path_list.sort(key=lambda x: int(re_pattern.findall(x)[0]))
    a = os.path.exists(save_path)
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    for s in path_list:
        dicom = pydicom.read_file(join(path, s))
        pixel_array, dicom.Rows, dicom.Columns = get_pixeldata(dicom)
        img_data = pixel_array
        winwidth = 500
        wincenter = 50
        rows = dicom.Rows
        cols = dicom.Columns
        dcm_temp = setDicomWinWidthWinCenter(img_data, winwidth, wincenter, rows, cols)
        # dcm_img = cv2.imshow('imgshow', dcm_temp)
        dcm_img = Image.fromarray(dcm_temp)
        dcm_img = dcm_img.convert('L')
        # dcm_img.show()
        save_file_path = save_path + '/{}.png'.format(s[:-4])
        dcm_img.save(save_file_path)


def get_pixels_hu(slices):
    image = np.stack([s.pixel_array for s in slices])
    # Convert to int16 (from sometimes int16)
    # Should be possible as values should always be low enough (<32k)
    image = image.astype(np.int16)
    # Set outside-of-scan pixels to 0
    # The intercept is usually -1024, so air is approximately 0
    image[image == -2000] = 0
    for slice_number in range(len(slices)):
        intercept = slices[slice_number].RescaleIntercept  # Intercept
        slope = slices[slice_number].RescaleIntercept
        if slope != 1:
            image[slice_number] = slope * image[slice_number].astype(np.float64)
            image[slice_number] = image[slice_number].astype(np.int16)
        image[slice_number] += np.int16(intercept)
    return np.array(image, dtype=np.int16)


def resample(image, scan, new_spaceing=[1, 1, 1]):
    # Determine current pixel spacing
    a = scan[0].PixelSpacing
    spacing = map(float, [scan[0].SliceThickness + scan[0].PixelSpacing[0]])
    spacing = np.array(list(spacing))
    resize_factor = spacing / new_spaceing
    new_real_shape = image.shape * resize_factor
    new_shape = np.round(new_real_shape)  # 返回浮点数x的四舍五入值。
    real_resize_factor = new_shape / image.shape
    new_spaceing = spacing / real_resize_factor
    image = ndi.interpolation.zoom(image, real_resize_factor, mode='nearest')
    return image, new_spaceing


def normalize(image):
    image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
    image[image > 1] = 1.
    image[image < 0] = 0.
    return image


MIN_BOUND = -1000.0
MAX_BOUND = 400.0
# save_path = "/home/sj/workspace/data/chao_data/Train_Sets/CT/2/dcm2png/"
# multi_dcm(path, save_path)

train_path = '/home/sj/workspace/data/chao_data/Train_Sets/'
test_path = '/home/sj/workspace/data/chao_data/Test_Sets/'


def deal_chao(path, type):
    if type == 'CT':
        path = path + 'CT/'
        for case in os.listdir(path):
            dcm_data = path + case + '/DICOM_anon/'
            label_data = path + case + '/Ground/'
            save_path = path + case + '/CTdcm2png/'
            # for img in os.listdir(dcm_data):
            multi_dcm(dcm_data, save_path)
    elif type == 'MR':
        path = path + 'MR/'
        for case in os.listdir(path):
            dcm_data = path + case + '/T2SPIR/DICOM_anon/'
            label_data = path + case + '/T2SPIR/Ground/'
            save_path = path + case + '/T2SPIR/MRdcm2png/'
            # for img in os.listdir(dcm_data):
            multi_dcm(dcm_data, save_path)


deal_chao(train_path, 'MR')
deal_chao(test_path, 'MR')
# first_patient = load_scan(path)
# first_patient_pixels = get_pixels_hu(first_patient)
# plt.hist(first_patient_pixels.flatten(), bins=80, color='c')
# plt.xlabel("Housfield Units (HU)")
# plt.ylabel("Frequency")
# plt.show()

# pix_resampled, spacing = resample(first_patient_pixels, first_patient, [1, 1, 1])


def plot_3d(image, threshold=-300):
    # Position the scan upright,
    # so the head of the patient would be at the top facing the camera
    p = image.transpose(2, 1, 0)  # 将扫描件竖直放置
    verts, faces = measure.marching_cubes(p, threshold)  # Liner推进立方体算法来查找3D体积数据中的曲面。
    # faces = measure.marching_cubes(p, threshold)
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    # Fancy indexing: `verts[faces]` to generate a collection of triangles
    mesh = Poly3DCollection(verts[faces], alpha=0.1)  # 创建3Dpoly
    face_color = [0.5, 0.5, 1]
    mesh.set_facecolor(face_color)  # 设置颜色
    ax.add_collection3d(mesh)
    ax.set_xlim(0, p.shape[0])
    ax.set_ylim(0, p.shape[1])
    ax.set_zlim(0, p.shape[2])
    plt.show()

# 调用函数
# plot_3d(pix_resampled, 400)
