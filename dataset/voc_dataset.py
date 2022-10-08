import os
import wget
import glob
import torch
import random
import tarfile
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch.utils.data as data
from xml.etree.ElementTree import parse
from matplotlib.patches import Rectangle
from utils import bar_custom, voc_color_array
from dataset.detection_transforms import mosaic
from config import device


def download_voc(root_dir='D:\data\\voc', remove_compressed_file=True):

    voc_2012_train_url = 'https://pjreddie.com/media/files/VOCtrainval_11-May-2012.tar'
    voc_2007_train_url = 'https://pjreddie.com/media/files/VOCtrainval_06-Nov-2007.tar'
    voc_2007_test_url = 'https://pjreddie.com/media/files/VOCtest_06-Nov-2007.tar'

    os.makedirs(root_dir, exist_ok=True)

    """Download the VOC data if it doesn't exist in processed_folder already."""
    if (os.path.exists(os.path.join(root_dir, 'VOCtrainval_11-May-2012')) and
        os.path.exists(os.path.join(root_dir, 'VOCtrainval_06-Nov-2007')) and
        os.path.exists(os.path.join(root_dir, 'VOCtest_06-Nov-2007'))):
        print("Already exist!")
        return

    print("Download...")

    wget.download(url=voc_2012_train_url, out=root_dir, bar=bar_custom)
    print('')
    wget.download(url=voc_2007_train_url, out=root_dir, bar=bar_custom)
    print('')
    wget.download(url=voc_2007_test_url, out=root_dir, bar=bar_custom)
    print('')

    os.makedirs(os.path.join(root_dir, 'VOCtrainval_11-May-2012'), exist_ok=True)
    os.makedirs(os.path.join(root_dir, 'VOCtrainval_06-Nov-2007'), exist_ok=True)
    os.makedirs(os.path.join(root_dir, 'VOCtest_06-Nov-2007'), exist_ok=True)

    print("Extract...")

    with tarfile.open(os.path.join(root_dir, 'VOCtrainval_11-May-2012.tar')) as tar:
        def is_within_directory(directory, target):
            
            abs_directory = os.path.abspath(directory)
            abs_target = os.path.abspath(target)
        
            prefix = os.path.commonprefix([abs_directory, abs_target])
            
            return prefix == abs_directory
        
        def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
        
            for member in tar.getmembers():
                member_path = os.path.join(path, member.name)
                if not is_within_directory(path, member_path):
                    raise Exception("Attempted Path Traversal in Tar File")
        
            tar.extractall(path, members, numeric_owner=numeric_owner) 
            
        
        safe_extract(tar, os.path.join(root_dir,"VOCtrainval_11-May-2012"))
    with tarfile.open(os.path.join(root_dir, 'VOCtrainval_06-Nov-2007.tar')) as tar:
        def is_within_directory(directory, target):
            
            abs_directory = os.path.abspath(directory)
            abs_target = os.path.abspath(target)
        
            prefix = os.path.commonprefix([abs_directory, abs_target])
            
            return prefix == abs_directory
        
        def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
        
            for member in tar.getmembers():
                member_path = os.path.join(path, member.name)
                if not is_within_directory(path, member_path):
                    raise Exception("Attempted Path Traversal in Tar File")
        
            tar.extractall(path, members, numeric_owner=numeric_owner) 
            
        
        safe_extract(tar, os.path.join(root_dir,"VOCtrainval_06-Nov-2007"))
    with tarfile.open(os.path.join(root_dir, 'VOCtest_06-Nov-2007.tar')) as tar:
        def is_within_directory(directory, target):
            
            abs_directory = os.path.abspath(directory)
            abs_target = os.path.abspath(target)
        
            prefix = os.path.commonprefix([abs_directory, abs_target])
            
            return prefix == abs_directory
        
        def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
        
            for member in tar.getmembers():
                member_path = os.path.join(path, member.name)
                if not is_within_directory(path, member_path):
                    raise Exception("Attempted Path Traversal in Tar File")
        
            tar.extractall(path, members, numeric_owner=numeric_owner) 
            
        
        safe_extract(tar, os.path.join(root_dir,"VOCtest_06-Nov-2007"))

    # remove tars
    if remove_compressed_file:
        root_zip_list = glob.glob(os.path.join(root_dir, '*.tar'))  # in root_dir remove *.zip
        for root_zip in root_zip_list:
            os.remove(root_zip)
        print("Remove *.tars")

    print("Done!")


class VOC_Dataset(data.Dataset):

    # not background for coco
    class_names = ('aeroplane', 'bicycle', 'bird', 'boat',
                   'bottle', 'bus', 'car', 'cat', 'chair',
                   'cow', 'diningtable', 'dog', 'horse',
                   'motorbike', 'person', 'pottedplant',
                   'sheep', 'sofa', 'train', 'tvmonitor')

    """
    voc dataset
    """
    def __init__(self,
                 root='D:\data\\voc',
                 split='train',
                 download=True,
                 transform=None,
                 visualization=True):
        super(VOC_Dataset, self).__init__()

        self.root = root
        # -------------------------- set split --------------------------
        assert split in ['train', 'test']
        self.split = split

        # -------------------------- download --------------------------
        self.download = download
        if self.download:
            download_voc(root_dir=root)

        # -------------------------- transform --------------------------
        self.transform = transform

        # -------------------------- visualization --------------------------
        self.visualization = visualization

        # -------------------------- data setting --------------------------
        self.data_list = []
        self.img_list = []
        self.anno_list = []

        for i in os.listdir(self.root):
            if i.find('.tar') == -1 and i.find(self.split) != -1:  # split 이 포함된 data - except .tar 제외
                self.data_list.append(i)

        for data_ in self.data_list:

            self.img_list.extend(glob.glob(os.path.join(os.path.join(self.root, data_), '*/*/JPEGImages/*.jpg')))
            self.anno_list.extend(glob.glob(os.path.join(os.path.join(self.root, data_), '*/*/Annotations/*.xml')))

        self.img_list = sorted(self.img_list)
        self.anno_list = sorted(self.anno_list)

        self.class_idx_dict = {class_name: i for i, class_name in enumerate(self.class_names)}     # class name : idx
        self.idx_class_dict = {i: class_name for i, class_name in enumerate(self.class_names)}     # idx : class name

    def __getitem__(self, idx):

        # load img
        image = Image.open(self.img_list[idx]).convert('RGB')
        # load labels
        boxes, labels = self.parse_voc(self.anno_list[idx])

        # load img name for string
        img_name = os.path.basename(self.anno_list[idx]).split('.')[0]
        img_name_to_ascii = [ord(c) for c in img_name]

        # load img width and height
        img_width, img_height = float(image.size[0]), float(image.size[1])

        boxes = torch.FloatTensor(boxes)
        labels = torch.LongTensor(labels)  # 0 ~ 19
        img_name = torch.FloatTensor([img_name_to_ascii])
        additional_info = torch.FloatTensor([img_width, img_height])

        # --------------------------- for transform ---------------------------
        if random.random() > 0.5:
            image, boxes, labels = self.mosaic_transform(image, boxes, labels)

        if self.transform is not None:
            image, boxes, labels = self.transform(image, boxes, labels)

        if self.visualization:
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])

            # tensor to img
            img_vis = np.array(image.permute(1, 2, 0), np.float32)  # C, W, H
            img_vis *= std
            img_vis += mean
            img_vis = np.clip(img_vis, 0, 1)

            plt.figure('input')
            plt.imshow(img_vis)
            print('num objects : {}'.format(len(boxes)))

            for i in range(len(boxes)):

                new_h_scale = new_w_scale = 1
                # box_normalization of DetResize
                if self.transform.transforms[-2].box_normalization:
                    new_h_scale, new_w_scale = image.size()[1:]

                x1 = boxes[i][0] * new_w_scale
                y1 = boxes[i][1] * new_h_scale
                x2 = boxes[i][2] * new_w_scale
                y2 = boxes[i][3] * new_h_scale

                # class
                plt.text(x=x1 - 5,
                         y=y1 - 5,
                         s=str(self.idx_class_dict[labels[i].item()]),   # FIXME
                         bbox=dict(boxstyle='round4',
                                   facecolor=voc_color_array[labels[i]],
                                   alpha=0.9))

                # bounding box
                plt.gca().add_patch(Rectangle(xy=(x1, y1),
                                              width=x2 - x1,
                                              height=y2 - y1,
                                              linewidth=1,
                                              edgecolor=voc_color_array[labels[i]],
                                              facecolor='none'))

            plt.show()
        if self.split == "test":
            return image, boxes, labels, img_name, additional_info

        return image, boxes, labels

    def parse_voc(self, xml_file_path):

        tree = parse(xml_file_path)
        root = tree.getroot()

        boxes = []
        labels = []

        for obj in root.iter("object"):

            # 'name' tag 에서 멈추기
            name = obj.find('./name')
            class_name = name.text.lower().strip()
            labels.append(self.class_idx_dict[class_name])

            # bbox tag 에서 멈추기
            bbox = obj.find('./bndbox')
            x_min = bbox.find('./xmin')
            y_min = bbox.find('./ymin')
            x_max = bbox.find('./xmax')
            y_max = bbox.find('./ymax')

            # from str to int
            x_min = float(x_min.text) - 1
            y_min = float(y_min.text) - 1
            x_max = float(x_max.text) - 1
            y_max = float(y_max.text) - 1

            boxes.append([x_min, y_min, x_max, y_max])

        return (np.array(boxes, dtype=np.float32),
                np.array(labels, dtype=np.int64))

    def collate_fn(self, batch):
        """
        :param batch: an iterable of N sets from __getitem__()
        :return: a tensor of images, lists of varying-size tensors of bounding boxes, labels, and difficulties
        """
        images = list()
        boxes = list()
        labels = list()
        img_name = list()
        additional_info = list()

        for b in batch:
            images.append(b[0])
            boxes.append(b[1])
            labels.append(b[2])
            if self.split == "test":
                img_name.append(b[3])
                additional_info.append(b[4])

        images = torch.stack(images, dim=0)
        if self.split == "test":
            return images, boxes, labels, img_name, additional_info
        return images, boxes, labels

    def __len__(self):
        return len(self.img_list)

    def mosaic_transform(self, image, boxes, labels):
        # 1. index 구하기
        idx_mosaic_1 = random.randint(0, self.__len__() - 1)
        idx_mosaic_2 = random.randint(0, self.__len__() - 1)
        idx_mosaic_3 = random.randint(0, self.__len__() - 1)

        # 2. image open 하기
        file_path_1 = self.img_list[idx_mosaic_1]
        file_path_2 = self.img_list[idx_mosaic_2]
        file_path_3 = self.img_list[idx_mosaic_3]
        new_image_1 = Image.open(file_path_1).convert('RGB')
        new_image_2 = Image.open(file_path_2).convert('RGB')
        new_image_3 = Image.open(file_path_3).convert('RGB')

        new_boxes_1, new_labels_1 = self.parse_voc(self.anno_list[idx_mosaic_1])
        new_boxes_1 = torch.FloatTensor(new_boxes_1)
        new_labels_1 = torch.LongTensor(new_labels_1)  # 0 ~ 19

        new_boxes_2, new_labels_2 = self.parse_voc(self.anno_list[idx_mosaic_2])
        new_boxes_2 = torch.FloatTensor(new_boxes_2)
        new_labels_2 = torch.LongTensor(new_labels_2)  # 0 ~ 19

        new_boxes_3, new_labels_3 = self.parse_voc(self.anno_list[idx_mosaic_3])
        new_boxes_3 = torch.FloatTensor(new_boxes_3)
        new_labels_3 = torch.LongTensor(new_labels_3)  # 0 ~ 19

        new_image = image
        new_boxes = boxes
        new_labels = labels

        images_ = (new_image, new_image_1, new_image_2, new_image_3)
        boxes_ = (new_boxes, new_boxes_1, new_boxes_2, new_boxes_3)
        labels_ = (new_labels, new_labels_1, new_labels_2, new_labels_3)
        new_image, new_boxes, new_labels = mosaic(images_, boxes_, labels_, 600)

        return new_image, new_boxes, new_labels


if __name__ == "__main__":
    import torchvision.transforms as transforms
    import dataset.detection_transforms as det_transforms

    # train_transform
    ubuntu_root = "/home/cvmlserver3/Sungmin/data/voc"
    window_root = 'D:\data\\voc'
    root = window_root

    transform_train = det_transforms.DetCompose([
        # ------------- for Tensor augmentation -------------
        det_transforms.DetRandomPhotoDistortion(),
        det_transforms.DetRandomHorizontalFlip(),
        det_transforms.DetToTensor(),
        # ------------- for Tensor augmentation -------------
        det_transforms.DetRandomZoomOut(max_scale=3),
        det_transforms.DetRandomZoomIn(),
        det_transforms.DetResize(size=(416, 416), box_normalization=True),
        det_transforms.DetNormalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
    ])

    transform_test = det_transforms.DetCompose([
        det_transforms.DetToTensor(),
        det_transforms.DetResize(size=800, max_size=1333, box_normalization=True),
        det_transforms.DetNormalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
    ])

    train_set = VOC_Dataset(root,
                            split='test',
                            download=True,
                            transform=transform_train,
                            visualization=True)

    train_loader = torch.utils.data.DataLoader(train_set,
                                               batch_size=1,
                                               collate_fn=train_set.collate_fn,
                                               shuffle=False,
                                               num_workers=0,
                                               pin_memory=True)

    for i, data in enumerate(train_loader):

        images = data[0]
        boxes = data[1]
        labels = data[2]

        images = images.to(device)
        boxes = [b.to(device) for b in boxes]
        labels = [l.to(device) for l in labels]
        print(labels)

