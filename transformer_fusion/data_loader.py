import os
import numpy as np
import cv2
from sklearn.preprocessing import normalize
from torchvision import transforms
import numbers
import random
from PIL import Image, ImageOps
from torch.utils.data import Dataset
import pickle


def pil_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            img = img.crop((0, 0, img.size[0] / 2, img.size[1]))
            return img.convert('RGB')


def frame_count(video, video_path):
    frames = []
    for v in video_path:
        mp4 = cv2.VideoCapture(video + v)  # 读取视频
        frame_count = mp4.get(7)
        frames.append(int(frame_count))
        # frames.append(int(np.ceil(frame_count/25)))
    return frames


class RandomCrop(object):
    def __init__(self, size, seq_length=16, padding=0):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.seq_length = seq_length
        self.padding = padding
        self.count = 0

    def __call__(self, img):

        if self.padding > 0:
            img = ImageOps.expand(img, border=self.padding, fill=0)

        w, h = img.size
        th, tw = self.size
        if w == tw and h == th:
            return img

        random.seed(self.count // self.seq_length)
        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)
        # print(self.count, x1, y1)
        self.count += 1
        return img.crop((x1, y1, x1 + tw, y1 + th))


class RandomHorizontalFlip(object):
    def __init__(self, seq_length=16):
        self.count = 0
        self.seq_length = seq_length

    def __call__(self, img):
        seed = self.count // self.seq_length
        random.seed(seed)
        prob = random.random()
        self.count += 1
        # print(self.count, seed, prob)
        if prob < 0.5:
            return img.transpose(Image.FLIP_LEFT_RIGHT)
        return img


class CholecDataset(Dataset):
    def __init__(self,
                 image,
                 file_paths,
                 file_labels,
                 kdata,
                 transform=None,
                 loader=pil_loader):
        self.file_paths = file_paths
        self.file_labels_phase = file_labels
        self.kdata = kdata
        self.image = image
        self.transform = transform
        self.loader = loader

    def __getitem__(self, index):
        img_names = self.image + self.file_paths[index]
        labels_phase = self.file_labels_phase[index]
        data = self.kdata[index]
        imgs = self.loader(img_names)
        if self.transform is not None:
            imgs = self.transform(imgs)

        return imgs, labels_phase, data

    def __len__(self):
        return len(self.file_paths)


def get_data(args):
    '''
    with open(data_path, 'rb') as f:
        train_test_paths_labels = pickle.load(f)
    '''

    train_path = os.listdir(args.video)
    test_path = os.listdir(args.video.replace('train', 'test'))
    train_num = len(train_path)
    test_num = len(test_path)
    train_ind = list(range(train_num))
    test_ind = list(range(test_num))
    np.random.seed(0)
    np.random.shuffle(train_ind)
    np.random.shuffle(test_ind)
    train_anno = open(args.annotation, 'r')
    test_anno = open(args.annotation.replace('train', 'test'), 'r')
    train_image = os.listdir(args.image)
    test_image = os.listdir(args.image.replace('train', 'test'))
    train_label = train_anno.readlines()
    test_label = test_anno.readlines()
    train_data = os.listdir(args.data)
    test_data = os.listdir(args.data.replace('train', 'test'))
    train_kdata = []
    for d in train_data:
        data = open(args.data + d, 'r').readlines()
        data = [da.replace('\n', '').split('\t') for da in data]
        train_kdata = train_kdata + data
    train_frames = frame_count(args.video, train_path)
    train_start_ind = [0]
    for i in range(1, train_num + 1):
        train_start_ind.append(sum(train_frames[0:i]))
    test_kdata = []
    for d in test_data:
        data = open(args.data.replace('train', 'test') + d, 'r').readlines()
        data = [da.replace('\n', '').split('\t') for da in data]
        test_kdata = test_kdata + data
    test_frames = frame_count(args.video.replace('train', 'test'), test_path)
    train_start_ind = [0]
    for i in range(1, train_num + 1):
        train_start_ind.append(sum(train_frames[0:i]))
    test_start_ind = [0]
    for i in range(1, test_num + 1):
        test_start_ind.append(sum(test_frames[0:i]))

    train_phase = [l.split('\t')[1] for l in train_label]
    phase_dict = {
        'Idle': 0,
        'Needle holding': 1,
        'Suture making': 2,
        'Suture handling': 3,
        '1 knot': 4,
        '2 knot': 5,
        '3 knot': 6,
        'Idle Step': 4
    }
    train_phases = [phase_dict[l] for l in train_phase]
    test_phase = [l.split('\t')[1] for l in test_label]
    test_phases = [phase_dict[l] for l in test_phase]

    train_num_each = train_frames
    test_num_each = test_frames
    train_phases = np.asarray(train_phases, dtype=np.int64)
    test_phases = np.asarray(test_phases, dtype=np.int64)
    train_kdata = np.asarray(train_kdata, dtype=np.float32)
    test_kdata = np.asarray(test_kdata, dtype=np.float32)
    train_kdata = normalize(train_kdata, axis=0, norm='max')
    test_kdata = normalize(test_kdata, axis=0, norm='max')

    if args.flip == 0:
        train_transforms = transforms.Compose([
            transforms.Resize([256, 256]),
            RandomCrop(224, args.seq),
            transforms.ToTensor(),
            #transforms.Normalize([0.3456, 0.2281, 0.2233], [0.2528, 0.2135, 0.2104])
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
    elif args.flip == 1:
        train_transforms = transforms.Compose([
            transforms.Resize([256, 256]),
            RandomCrop(224, args.seq),
            RandomHorizontalFlip(args.seq),
            transforms.ToTensor(),
            #transforms.Normalize([0.3456, 0.2281, 0.2233], [0.2528, 0.2135, 0.2104])
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
    test_transforms = transforms.Compose([
        transforms.Resize([224, 224]),
        transforms.ToTensor(),
        #transforms.Normalize([0.3456, 0.2281, 0.2233], [0.2528, 0.2135, 0.2104])
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    train_dataset = CholecDataset(args.image, train_image, train_phases,
                                  train_kdata, train_transforms)
    test_dataset = CholecDataset(args.image.replace('train',
                                                    'test'), test_image,
                                 test_phases, test_kdata, test_transforms)

    return train_dataset, train_num_each, test_dataset, test_num_each