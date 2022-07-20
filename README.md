# ATLAS-R2.0---Stroke-Lesion-Segmentation
> This is an example of the MR imaging is used to Segment Stroke-Lesion-Segmentation.
![](ATLASR2.0.png)

## Prerequisities
The following dependencies are needed:
- numpy >= 1.11.1
- SimpleITK >=1.0.1
- pytorch-gpu ==1.10.0
- pandas >=0.20.1
- scikit-learn >= 0.17.1

## How to Use
* 1、when download the all project,check out the data folder all csv,put your train data into same folder.or you can run ATLASR2.0data3dpreparewithSize.py to generate train data and validation data.
* 2、run ATLASR2.0_train.py for Unet3d segmeatation training:make sure train data have effective path
* 3、run ATLASR2.0_inference.py for Unet3d segmeatation inference:make sure test data have effective path

## Result

* dice:train loss,train accuracy,validation loss,validation accuracy
![](dice_loss.png)
![](dice_accu.png)

* validation dataset segmentation result,left result is GT,right is predict
![](1.png)
![](2.png)
![](3.png)
![](4.png)
![](5.png)

* more detail and trained model can follow my WeChat Public article.

## Contact
* https://github.com/junqiangchen
* email: 1207173174@qq.com
* Contact: junqiangChen
* WeChat Number: 1207173174
* WeChat Public number: 最新医学影像技术
