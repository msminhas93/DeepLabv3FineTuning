import unittest
from pathlib import Path

import torch
from PIL import Image
from segdataset import SegmentationDataset
from torch.utils.data import DataLoader
from torchvision import transforms


class Test_TestSegmentationDataset(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        seg_dataset = SegmentationDataset("CrackForest",
                                          "Images",
                                          "Masks",
                                          transforms=transforms.Compose(
                                              [transforms.ToTensor()]))
        seg_dataloader = DataLoader(seg_dataset,
                                    batch_size=4,
                                    shuffle=False,
                                    num_workers=8)
        cls.samples = next(iter(seg_dataloader))

    def test_image_tensor_dimensions(self):
        image_tensor_shape = Test_TestSegmentationDataset.samples[
            'image'].shape
        self.assertEqual(image_tensor_shape[0], 4)
        self.assertEqual(image_tensor_shape[1], 3)
        self.assertEqual(image_tensor_shape[2], 320)
        self.assertEqual(image_tensor_shape[3], 480)

    def test_mask_tensor_dimensions(self):
        mask_tensor_shape = Test_TestSegmentationDataset.samples['mask'].shape
        self.assertEqual(mask_tensor_shape[0], 4)
        self.assertEqual(mask_tensor_shape[1], 1)
        self.assertEqual(mask_tensor_shape[2], 320)
        self.assertEqual(mask_tensor_shape[3], 480)

    def test_mask_img_pair(self):
        ref_image_tensor = transforms.ToTensor()(Image.open(
            Path("CrackForest/Images/001.jpg")))
        ref_mask_tensor = transforms.ToTensor()(Image.open(
            Path("CrackForest/Masks/001_label.PNG")))
        datagen_image_tensor = Test_TestSegmentationDataset.samples['image'][0]
        datagen_mask_tensor = Test_TestSegmentationDataset.samples['mask'][0]
        self.assertTrue(torch.equal(ref_image_tensor, datagen_image_tensor))
        self.assertTrue(torch.equal(ref_mask_tensor, datagen_mask_tensor))
