DeepLabv3FineTuning

This repository contains code for Fine Tuning [DeepLabV3 ResNet101](https://arxiv.org/abs/1706.05587) in PyTorch. The model is from the [torchvision module](https://pytorch.org/docs/stable/torchvision/models.html#semantic-segmentation). The tutorial can be found here: [https://expoundai.wordpress.com/2019/08/30/transfer-learning-for-segmentation-using-deeplabv3-in-pytorch/](https://expoundai.wordpress.com/2019/08/30/transfer-learning-for-segmentation-using-deeplabv3-in-pytorch/)

I've fine tuned the model for the [CrackForest](https://github.com/cuilimeng/CrackForest-dataset) data-set. 

The model was fine tuned for 25 epochs and achieves an testing AUROC value of 0.837214.

The segmentation output of the model on a sample image are shown below.

![Sample segmentation output](./CFExp/SegmentationOutput.png)

To run the code on your dataset use the following command.

```
python main.py "data_directory_path" "experiment_folder_where weights and log file need to be saved"
```
It has following two optional arguments:
```
--epochs : Specify the number of epochs. Default is 25.
--batchsize: Specify the batch size. Default is 4.
```
The datahandler module has two functions for creating datasets fron single and different folders.

1. ```get_dataloader_sep_folder(data_dir, imageFolder='Images', maskFolder='Masks', batch_size=4)```

Create Train and Test dataloaders from two separate Train and Test folders. The directory structure should be as follows.
```
data_dir
--Train
------Image
---------Image1
---------ImageN
------Mask
---------Mask1
---------MaskN
--Train
------Image
---------Image1
---------ImageN
------Mask
---------Mask1
---------MaskN
```
2. ```get_dataloader_single_folder(data_dir, imageFolder='Images', maskFolder='Masks', fraction=0.2, batch_size=4)```

Create from a single folder. The structure should be as follows.
```
--data_dir
------Image
---------Image1
---------ImageN
------Mask
---------Mask1
---------MaskN
```

The repository also contains a JupyterLab file with the loss and metric plots as well as the sample prediction code.
