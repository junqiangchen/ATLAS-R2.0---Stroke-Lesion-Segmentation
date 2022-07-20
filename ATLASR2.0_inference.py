import torch
import os
from model import *
from dataprocess.utils import file_name_path
import SimpleITK as sitk

# Use CUDA
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
use_cuda = torch.cuda.is_available()


def inferencebinaryunet3dtest():
    newSize = (144, 176, 144)
    Unet3d = BinaryUNet3dModel(image_depth=144, image_height=176, image_width=144, image_channel=1, numclass=1,
                               batch_size=1, loss_name='BinaryDiceLoss', inference=True,
                               model_path=r'log\ATLASR2.0\dice\BinaryUNet3d.pth')
    datapath = r"D:\challenge\data\ATLASR2.0\validation\Image"
    makspath = r"D:\challenge\data\ATLASR2.0\validation\Maskpytorch"
    image_path_list = file_name_path(datapath, False, True)
    for i in range(len(image_path_list)):
        imagepathname = datapath + "/" + image_path_list[i]
        sitk_image = sitk.ReadImage(imagepathname)
        sitk_mask = Unet3d.inference(sitk_image, newSize)
        maskpathname = makspath + "/" + image_path_list[i]
        sitk.WriteImage(sitk_mask, maskpathname)


if __name__ == '__main__':
    inferencebinaryunet3dtest()
