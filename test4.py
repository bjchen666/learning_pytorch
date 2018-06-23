from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F

class MyRandomCrop(transforms.RandomCrop):
    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be cropped.

        Returns:
            PIL Image: Cropped image.
        """
        if self.padding > 0:
            img = F.pad(img, self.padding)

        # pad the width if needed
        if self.pad_if_needed and img.size[0] < self.size[1]:
            img = F.pad(img, (int((1 + self.size[1] - img.size[0]) / 2), 0))
        # pad the height if needed
        if self.pad_if_needed and img.size[1] < self.size[0]:
            img = F.pad(img, (0, int((1 + self.size[0] - img.size[1]) / 2)))

        i, j, h, w = self.get_params(img, self.size)

        return F.crop(img, i, j, h, w),(i,j)

img=Image.open('/home/gshdong/deeplab_demo/VOCdevkit/VOC2012/JPEGImages/2007_002212.jpg')
img,cor=MyRandomCrop((250,350))(img)
print cor
plt.imshow(img)
plt.show()
