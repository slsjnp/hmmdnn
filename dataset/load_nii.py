import matplotlib
from matplotlib import pylab as plt
import nibabel as nib
from nibabel.viewers import OrthoSlicer3D

file = '/home/sj/workspace/data/LITS-Challenge-Train-Data/data/volume-0.nii.gz'
img = nib.load(file)
a = img.get_fdata()
print(img)
print(img.header['db_name'])

width, height, queue = img.dataobj.shape

OrthoSlicer3D(img.dataobj).show()

num = 1
for i in range(0, queue, 10):
    img_arr = img.dataobj[:, :, i]
    plt.subplot(5, 4, num)
    plt.imshow(img_arr, cmap='gray')
    num += 1

plt.show()
