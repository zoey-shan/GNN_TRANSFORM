from transformer import Transformer
from sklearn.preprocessing import normalize
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import torch.nn.init as init
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from torch.nn import DataParallel
import os
from PIL import Image, ImageOps, ImageDraw
import time
import pickle
import numpy as np
from torchvision.transforms import Lambda
import argparse
import copy
import random
import numbers
import math
import cv2
from torch.utils.data import Sampler
from torch.nn import functional as F

parser = argparse.ArgumentParser(description='lstm training')
parser.add_argument(
    '-g', '--gpu', default=[0], nargs='+', type=int, help='index of gpu to use, default 2')
parser.add_argument('-s', '--seq', default=16, type=int,
                    help='sequence length, default 4')
parser.add_argument('-t', '--train', default=64, type=int,
                    help='train batch size, default 100')
parser.add_argument('-v', '--val', default=32, type=int,
                    help='valid batch size, default 8')
parser.add_argument('-o', '--opt', default=1, type=int,
                    help='0 for sgd 1 for adam, default 1')
parser.add_argument('-m', '--multi', default=1, type=int,
                    help='0 for single opt, 1 for multi opt, default 1')
parser.add_argument('-e', '--epo', default=100, type=int,
                    help='epochs to train and val, default 25')
parser.add_argument('-w', '--work', default=2, type=int,
                    help='num of workers to use, default 2')
parser.add_argument('-f', '--flip', default=0, type=int,
                    help='0 for not flip, 1 for flip, default 0')
parser.add_argument('-c', '--crop', default=1, type=int,
                    help='0 rand, 1 cent, 5 five_crop, 10 ten_crop, default 1')
parser.add_argument('-l', '--lr', default=1e-4, type=float,
                    help='learning rate for optimizer, default 1e-3')
parser.add_argument('--momentum', default=0.9, type=float,
                    help='momentum for sgd, default 0.9')
parser.add_argument('--weightdecay', default=0, type=float,
                    help='weight decay for sgd, default 0')
parser.add_argument('--dampening', default=0, type=float,
                    help='dampening for sgd, default 0')
parser.add_argument('--nesterov', default=False, type=bool,
                    help='nesterov momentum, default False')
parser.add_argument('--sgdadjust', default=1, type=int,
                    help='sgd method adjust lr 0 for step 1 for min, default 1')
parser.add_argument('--sgdstep', default=5, type=int,
                    help='number of steps to adjust lr for sgd, default 5')
parser.add_argument('--sgdgamma', default=0.1, type=float,
                    help='gamma of steps to adjust lr for sgd, default 0.1')

parser.add_argument('--annotation', default='annotation_train.txt',
                    type=str, help='annotation path')
parser.add_argument(
    '--video', default='F:/Feature/MISAW_train/Video/', type=str, help='Video path')
parser.add_argument('--image', default='F:/Feature/MISAW_train/Images/',
                    type=str, help='Video to images save path')
parser.add_argument('--data', default='F:/Feature/MISAW_train/Kinematic/',
                    type=str, help='Video to kdata save path')
parser.add_argument('--srate', default=5, type=int, help='sample rate')

args = parser.parse_args()

gpu_usg = ",".join(list(map(str, args.gpu)))
sequence_length = args.seq
train_batch_size = args.train
val_batch_size = args.val
optimizer_choice = args.opt
multi_optim = args.multi
epochs = args.epo
workers = args.work
use_flip = args.flip
crop_type = args.crop
learning_rate = args.lr
momentum = args.momentum
weight_decay = args.weightdecay
dampening = args.dampening
use_nesterov = args.nesterov

sgd_adjust_lr = args.sgdadjust
sgd_step = args.sgdstep
sgd_gamma = args.sgdgamma

annotation = args.annotation
video = args.video
image = args.image
data = args.data
srate = args.srate

os.environ["CUDA_VISIBLE_DEVICES"] = gpu_usg
num_gpu = torch.cuda.device_count()
use_gpu = torch.cuda.is_available()


def pil_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            img = img.crop((0, 0, img.size[0] / 2, img.size[1]))
            return img.convert('RGB')


class RandomCrop(object):
    def __init__(self, size, padding=0):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.padding = padding
        self.count = 0

    def __call__(self, img):

        if self.padding > 0:
            img = ImageOps.expand(img, border=self.padding, fill=0)

        w, h = img.size
        th, tw = self.size
        if w == tw and h == th:
            return img

        random.seed(self.count // sequence_length)
        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)
        # print(self.count, x1, y1)
        self.count += 1
        return img.crop((x1, y1, x1 + tw, y1 + th))


class RandomHorizontalFlip(object):
    def __init__(self):
        self.count = 0

    def __call__(self, img):
        seed = self.count // sequence_length
        random.seed(seed)
        prob = random.random()
        self.count += 1
        # print(self.count, seed, prob)
        if prob < 0.5:
            return img.transpose(Image.FLIP_LEFT_RIGHT)
        return img


class CholecDataset(Dataset):
    def __init__(self, image, file_paths, file_labels, kdata,
                 transform=None, loader=pil_loader):
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


def get_useful_start_idx(sequence_length, list_each_length):
    count = 0
    idx = []
    for i in range(len(list_each_length)):
        for j in range(count, count + (list_each_length[i] + 1 - sequence_length * srate), sequence_length * srate):
            idx.append(j)
        count += list_each_length[i]
    return idx


def frame_count(video, video_path):
    frames = []
    for v in video_path:
        mp4 = cv2.VideoCapture(video + v)  # 读取视频
        frame_count = mp4.get(7)
        frames.append(int(frame_count))
        # frames.append(int(np.ceil(frame_count/25)))
    return frames


def get_data():
    '''
    with open(data_path, 'rb') as f:
        train_test_paths_labels = pickle.load(f)
    '''

    train_path = os.listdir(video)
    test_path = os.listdir(video.replace('train', 'test'))
    train_num = len(train_path)
    test_num = len(test_path)
    train_ind = list(range(train_num))
    test_ind = list(range(test_num))
    np.random.seed(0)
    np.random.shuffle(train_ind)
    np.random.shuffle(test_ind)
    train_anno = open(annotation, 'r')
    test_anno = open(annotation.replace('train', 'test'), 'r')
    train_image = os.listdir(image)
    test_image = os.listdir(image.replace('train', 'test'))
    train_label = train_anno.readlines()
    test_label = test_anno.readlines()
    train_data = os.listdir(args.data)
    test_data = os.listdir(args.data.replace('train', 'test'))
    train_kdata = []
    for d in train_data:
        data = open(args.data + d, 'r').readlines()
        data = [da.replace('\n', '').split('\t') for da in data]
        train_kdata = train_kdata + data
    train_frames = frame_count(video, train_path)
    train_start_ind = [0]
    for i in range(1, train_num+1):
        train_start_ind.append(sum(train_frames[0:i]))
    test_kdata = []
    for d in test_data:
        data = open(args.data.replace('train', 'test') + d, 'r').readlines()
        data = [da.replace('\n', '').split('\t') for da in data]
        test_kdata = test_kdata + data
    test_frames = frame_count(video.replace('train', 'test'), test_path)
    train_start_ind = [0]
    for i in range(1, train_num+1):
        train_start_ind.append(sum(train_frames[0:i]))
    test_start_ind = [0]
    for i in range(1, test_num+1):
        test_start_ind.append(sum(test_frames[0:i]))

    train_phase = [l.split('\t')[1] for l in train_label]
    phase_dict = {'Idle': 0, 'Needle holding': 1, 'Suture making': 2,
                  'Suture handling': 3, '1 knot': 4, '2 knot': 5, '3 knot': 6, 'Idle Step': 4}
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
    '''
    print('train_paths  : {:6d}'.format(len(train_paths)))
    print('train_labels : {:6d}'.format(len(train_phase)))
    print('valid_paths  : {:6d}'.format(len(val_paths)))
    print('valid_labels : {:6d}'.format(len(val_phase)))
    print('test_paths   : {:6d}'.format(len(test_paths)))
    print('test_labels  : {:6d}'.format(len(test_phase)))
    '''

    if use_flip == 0:
        train_transforms = transforms.Compose([
            transforms.Resize([256, 256]),
            RandomCrop(224),
            transforms.ToTensor(),
            #transforms.Normalize([0.3456, 0.2281, 0.2233], [0.2528, 0.2135, 0.2104])
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                                 0.229, 0.224, 0.225])
        ])
    elif use_flip == 1:
        train_transforms = transforms.Compose([
            transforms.Resize([256, 256]),
            RandomCrop(224),
            RandomHorizontalFlip(),
            transforms.ToTensor(),
            #transforms.Normalize([0.3456, 0.2281, 0.2233], [0.2528, 0.2135, 0.2104])
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                                 0.229, 0.224, 0.225])
        ])
    test_transforms = transforms.Compose([
        transforms.Resize([224, 224]),
        transforms.ToTensor(),
        #transforms.Normalize([0.3456, 0.2281, 0.2233], [0.2528, 0.2135, 0.2104])
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225])
    ])

    train_dataset = CholecDataset(
        image, train_image, train_phases, train_kdata, train_transforms)
    test_dataset = CholecDataset(image.replace(
        'train', 'test'), test_image, test_phases, test_kdata, test_transforms)

    return train_dataset, train_num_each, test_dataset, test_num_each


# 序列采样sampler
class SeqSampler(Sampler):
    def __init__(self, data_source, idx):
        super().__init__(data_source)
        self.data_source = data_source
        self.idx = idx

    def __iter__(self):
        return iter(self.idx)

    def __len__(self):
        return len(self.idx)


default_colors = ['#005fa7', '#884c96', '#d762a0', '#b6bb12', '#00adc4']


def draw_ribbon(data, output_path=None, data_height=30, data_width=500, padding=10, colors=default_colors):
    num_series = len(data)
    num_colors = len(colors)
    height = data_height * num_series + padding * (num_series + 1)
    width = data_width + padding * 2

    img = Image.new('RGB', (width, height))
    canvas = ImageDraw.Draw(img)
    canvas.rectangle((0, 0, width, height), fill='#ffffff')
    for i, series in enumerate(data):
        total = sum(series)
        x = padding
        y = data_height * i + padding * (i + 1)
        for j, d in enumerate(series):
            d_width = d / total * data_width
            canvas.rectangle((x, y, x + d_width, y + data_height),
                             fill=colors[j % num_colors])
            x += d_width
    if not output_path:
        img.show()
    else:
        img.save(output_path)


def train_model(train_dataset, train_num_each, val_dataset, val_num_each):
    num_train = len(train_dataset)
    num_val = len(val_dataset)

    train_useful_start_idx = get_useful_start_idx(
        sequence_length, train_num_each)
    val_useful_start_idx = get_useful_start_idx(sequence_length, val_num_each)

    num_train_we_use = len(train_useful_start_idx) // num_gpu * num_gpu
    num_val_we_use = len(val_useful_start_idx) // num_gpu * num_gpu
    # num_train_we_use = 8000
    # num_val_we_use = 800

    # 训练数据开始位置
    train_we_use_start_idx = train_useful_start_idx[0:num_train_we_use]
    val_we_use_start_idx = val_useful_start_idx[0:num_val_we_use]

    np.random.seed(0)
    np.random.shuffle(train_we_use_start_idx)
    train_idx = []
    for i in range(num_train_we_use):
        for j in range(sequence_length):
            # 训练数据位置，每一张图是一个数据
            train_idx.append(train_we_use_start_idx[i] + j * srate)

    val_idx = []
    for i in range(num_val_we_use):
        for j in range(sequence_length):
            val_idx.append(val_we_use_start_idx[i] + j * srate)

    num_train_all = float(len(train_idx))
    num_val_all = float(len(val_idx))

    val_loader = DataLoader(
        val_dataset,
        batch_size=val_batch_size,
        # sampler=val_idx,
        sampler=SeqSampler(val_dataset, val_idx),
        num_workers=workers,
        pin_memory=False
    )
    model = Transformer(7, sequence_length)
    if use_gpu:
        model = model.cuda()

    #model = DataParallel(model)
    criterion = nn.CrossEntropyLoss()
    '''
    if multi_optim == 0:
        if optimizer_choice == 0:
            optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, dampening=dampening,
                                  weight_decay=weight_decay, nesterov=use_nesterov)
            if sgd_adjust_lr == 0:
                exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=sgd_adjust_lr, gamma=sgd_gamma)
            elif sgd_adjust_lr == 1:
                exp_lr_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
        elif optimizer_choice == 1:
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    elif multi_optim == 1:
        if optimizer_choice == 0:
            optimizer = optim.SGD([
                {'params': model.module.share.parameters()},
                {'params': model.module.lstm.parameters(), 'lr': learning_rate},
                {'params': model.module.fc.parameters(), 'lr': learning_rate},
            ], lr=learning_rate / 10, momentum=momentum, dampening=dampening,
                weight_decay=weight_decay, nesterov=use_nesterov)
            if sgd_adjust_lr == 0:
                exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=sgd_adjust_lr, gamma=sgd_gamma)
            elif sgd_adjust_lr == 1:
                exp_lr_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
        elif optimizer_choice == 1:
            optimizer = optim.Adam([
                {'params': model.module.share.parameters()},
                {'params': model.module.lstm.parameters(), 'lr': learning_rate},
                {'params': model.module.fc.parameters(), 'lr': learning_rate},
            ], lr=learning_rate / 10)
    '''
    # if multi_optim == 0:
    #     if optimizer_choice == 0:
    #         optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, dampening=dampening,
    #                               weight_decay=weight_decay, nesterov=use_nesterov)
    #         if sgd_adjust_lr == 0:
    #             exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=sgd_adjust_lr, gamma=sgd_gamma)
    #         elif sgd_adjust_lr == 1:
    #             exp_lr_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
    #     elif optimizer_choice == 1:
    #         optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    # elif multi_optim == 1:
    #     if optimizer_choice == 0:
    #         optimizer = optim.SGD([
    #             {'params': model.module.share.parameters()},
    #             {'params': model.module.lstm.parameters(), 'lr': learning_rate},
    #             {'params': model.module.fc.parameters(), 'lr': learning_rate},
    #         ], lr=learning_rate / 10, momentum=momentum, dampening=dampening,
    #             weight_decay=weight_decay, nesterov=use_nesterov)
    #         if sgd_adjust_lr == 0:
    #             exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=sgd_adjust_lr, gamma=sgd_gamma)
    #         elif sgd_adjust_lr == 1:
    #             exp_lr_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
    #     elif optimizer_choice == 1:
    #         optimizer = optim.Adam([
    #             {'params': model.module.share.parameters()},
    #             {'params': model.module.lstm.parameters(), 'lr': learning_rate},
    #             {'params': model.module.fc.parameters(), 'lr': learning_rate},
    #         ], lr=learning_rate / 10)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.8)

    best_model_wts = copy.deepcopy(model.state_dict())
    best_val_accuracy = 0.0
    correspond_train_acc = 0.0

    record_np = np.zeros([epochs, 3])

    for epoch in range(epochs):
        np.random.seed(epoch)
        np.random.shuffle(train_we_use_start_idx)
        train_idx = []
        for i in range(num_train_we_use):
            for j in range(sequence_length):
                train_idx.append(train_we_use_start_idx[i] + j * srate)

        train_loader = DataLoader(
            train_dataset,
            batch_size=train_batch_size,
            sampler=SeqSampler(train_dataset, train_idx),
            num_workers=workers,
            pin_memory=False
        )

        model.train()
        train_loss = 0.0
        train_corrects = 0
        train_start_time = time.time()
        num = 0
        train_num = 0
        for data in train_loader:
            num = num + 1
            inputs, labels_phase, kdata = data
            if use_gpu:
                inputs = Variable(inputs.cuda())
                labels = Variable(labels_phase.cuda())
                kdatas = Variable(kdata.cuda())
            else:
                inputs = Variable(inputs)
                labels = Variable(labels_phase)
                kdatas = Variable(kdata)
            optimizer.zero_grad()
            outputs, f = model.forward(inputs, kdatas)
            outputs = F.softmax(outputs, dim=1)
            _, preds = torch.max(outputs.data, 1)
            print(f'{num}-th batch')
            print(f'{epoch} predict {preds}')
            print(f'{epoch} label {labels}')

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.data
            train_corrects += torch.sum(preds == labels.data)
            train_num += labels.shape[0]
            print(f'batch accuracy {train_corrects.cpu().numpy() / train_num}')
            if train_corrects.cpu().numpy() / train_num > 0.8:
                torch.save(copy.deepcopy(model.state_dict()),
                           str(epoch)+'.pth')

        train_elapsed_time = time.time() - train_start_time
        train_accuracy = train_corrects / train_num
        train_average_loss = train_loss / num_train_all

        # begin eval
        num_labels = 10
        if use_gpu:
            pred_counter = torch.zeros(num_labels).cuda()
            label_counter = torch.zeros(num_labels).cuda()
        else:
            pred_counter = torch.zeros(num_labels)
            label_counter = torch.zeros(num_labels)
        # pred_counter = torch.zeros(num_labels)
        # label_counter = torch.zeros(num_labels)
        model.eval()
        val_corrects = 0
        val_start_time = time.time()
        val_num = 0
        v_cn = 0
        with torch.no_grad():
            for data in val_loader:
                inputs, labels_phase, kdata = data
                # labels_phase = labels_phase[(sequence_length - 1)::sequence_length]
                # kdata = kdata[(sequence_length - 1)::sequence_length]
                v_cn += 1
                if use_gpu:
                    inputs = Variable(inputs.cuda())
                    labels = Variable(labels_phase.cuda())
                    kdatas = Variable(kdata.cuda())
                else:
                    inputs = Variable(inputs)
                    labels = Variable(labels_phase)
                    kdatas = Variable(kdata)

                outputs, f = model.forward(inputs, kdatas)
                # outputs = outputs[sequence_length - 1::sequence_length]
                _, preds = torch.max(outputs.data, 1)
                cur_pred_counter = torch.bincount(preds, minlength=num_labels)
                cur_label_counter = torch.bincount(
                    labels, minlength=num_labels)
                pred_counter += cur_pred_counter
                label_counter += cur_label_counter
                # print(f'{epoch} val predict {preds}')
                # print(f'{epoch} val label {labels}')
                print(f'{v_cn} val batch')
                val_corrects += torch.sum(preds == labels.data)
                val_num += labels.shape[0]
                val_elapsed_time = time.time() - val_start_time
                val_accuracy = val_corrects / val_num

        print('epoch: {:4d}'
              ' train in: {:2.0f}m{:2.0f}s'
              ' train loss: {:4.4f}'
              ' train accu: {:.4f}'
              ' valid in: {:2.0f}m{:2.0f}s'
              ' valid accu: {:.4f}'
              .format(epoch,
                      train_elapsed_time // 60,
                      train_elapsed_time % 60,
                      train_average_loss,
                      train_accuracy,
                      val_elapsed_time // 60,
                      val_elapsed_time % 60,
                      val_accuracy))

        # if optimizer_choice == 0:
        #     if sgd_adjust_lr == 0:
        #         exp_lr_scheduler.step()
        #     elif sgd_adjust_lr == 1:
        #         exp_lr_scheduler.step(train_average_loss) # val_average_loss

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            correspond_train_acc = train_accuracy
            best_model_wts = copy.deepcopy(model.state_dict())
        if val_accuracy == best_val_accuracy:
            if train_accuracy > correspond_train_acc:
                correspond_train_acc = train_accuracy
                best_model_wts = copy.deepcopy(model.state_dict())

        record_np[epoch, 0] = train_accuracy
        record_np[epoch, 1] = train_average_loss
        record_np[epoch, 2] = val_accuracy
        np.save(str(epoch) + '.npy', record_np)
        scheduler.step()

    draw_ribbon([pred_counter.tolist(), label_counter.tolist()],
                output_path='ribbon_trans.png')
    print('best accuracy: {:.4f} cor train accu: {:.4f}'.format(
        best_val_accuracy, correspond_train_acc))

    save_val = int("{:4.0f}".format(best_val_accuracy * 10000))
    save_train = int("{:4.0f}".format(correspond_train_acc * 10000))
    model_name = "attention" \
                 + "_epoch_" + str(epochs) \
                 + "_length_" + str(sequence_length) \
                 + "_opt_" + str(optimizer_choice) \
                 + "_mulopt_" + str(multi_optim) \
                 + "_flip_" + str(use_flip) \
                 + "_crop_" + str(crop_type) \
                 + "_batch_" + str(train_batch_size) \
                 + "_train_" + str(save_train) \
                 + "_val_" + str(save_val) \
                 + ".pth"

    torch.save(best_model_wts, model_name)

    record_name = "attention" \
                  + "_epoch_" + str(epochs) \
                  + "_length_" + str(sequence_length) \
                  + "_opt_" + str(optimizer_choice) \
                  + "_mulopt_" + str(multi_optim) \
                  + "_flip_" + str(use_flip) \
                  + "_crop_" + str(crop_type) \
                  + "_batch_" + str(train_batch_size) \
                  + "_train_" + str(save_train) \
                  + "_val_" + str(save_val) \
                  + ".npy"
    np.save(record_name, record_np)


def main():
    train_dataset, train_num_each, test_dataset, test_num_each = get_data()
    train_model(train_dataset, train_num_each, test_dataset, test_num_each)


if __name__ == "__main__":
    main()

print()
