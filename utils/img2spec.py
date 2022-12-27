from torch.utils.data import Dataset, DataLoader
from torch import nn
from torch import optim
import torch
import glob
import matplotlib.font_manager as fm
import torch.nn.functional as F
from PIL import Image, ImageFont, ImageDraw
import random
import numpy as np
import pandas as pd
import os
from skimage import io, color, data, filters
from skimage import morphology as mo
from skimage import transform as tr
from sklearn.linear_model import LinearRegression


class LabelDataset(Dataset):  # 需要继承data.Dataset
    def __init__(self, size=100000, label_size=[60, 50, 40, 35, 30, 25, 20]):
        self.tag_set = ['0', '1', '2', '3', '4',
                        '5', '6', '7', '8', '9', '.', ' ', '-']
        self.font_files = glob.glob('./font_family/*')
        self.x_list = [20, 25, 30]
        self.y_list = [0, 1, 2, 3, 4, 5, 6]
        self.size = size
        self.label_size = label_size

    def __getitem__(self, index):
        x = self.random_select(self.x_list)
        y = self.random_select(self.y_list)
        fn = self.random_select(self.font_files)
        tag = self.random_select(self.tag_set)
        index = self.tag_set.index(tag)
        fm_ = fm.FontProperties(fname=fn)
        tag_im = self.gentag(tag, x, y, 70, 70, '', fm_)
        tag_array = np.array(tag_im)
        tag_array = tag_array == [255, 255, 255]
        tag_array = np.all(tag_array, axis=2)
        tag_array = 1-tag_array.astype(np.float32)
        return torch.tensor(tag_array).float(), torch.tensor([index]).long()

    def __len__(self):
        # You should change 0 to the total size of your dataset.
        return self.size

    def gentag(self, tag, x, y, image_l=250, image_h=40, file='', fm_=fm.FontProperties(family='DejaVu Sans')):
        size = self.random_select(self.label_size)
        text = tag
        im = Image.new("RGB", (image_l, image_h), (255, 255, 255))
        dr = ImageDraw.Draw(im)
        font = ImageFont.truetype(fm.findfont(fm_), size)
        dr.text((x, y), text, font=font, fill="#000000")
        if file != '':
            im.save(file)
        return im

    @staticmethod
    def random_select(items):
        rand_idx = random.randint(0, len(items)-1)
        return items[rand_idx]


class MyCNN(nn.Module):

    def __init__(self, numclasses=3):
        super(MyCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 128, kernel_size=3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(128, 128, kernel_size=3,
                               padding=2, dilation=2, bias=False, groups=2)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3,
                               padding=4, dilation=4, bias=False, groups=2)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3,
                               padding=4, dilation=4, bias=False, groups=2)
        self.pool = nn.AdaptiveMaxPool2d((1, 1))
        self.lin = nn.Conv2d(128, numclasses, kernel_size=1)
        #self.lin2 = nn.Linear(128,numclasses)

    def forward(self, x):

        x = x.unsqueeze(1)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = self.lin(x)
        x = x.squeeze(2)
        x = x.squeeze(2)
        return x
    
    
def train_number_model(label_size = [60, 50, 40, 35, 30, 25, 20],
                train_label_set_size = 150000,
                test_label_set_size = 10000,
                device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
                learning_rate = 1e-3,
                criterion = nn.CrossEntropyLoss(),
                output_path='./model/recog_label_model.pth'):
    
    print(device)
    train_label_set = LabelDataset(train_label_set_size, label_size)
    test_label_set = LabelDataset(test_label_set_size, label_size)
    train_load = DataLoader(train_label_set, batch_size=256, shuffle=True)
    test_load = DataLoader(test_label_set, batch_size=256, shuffle=True)
    recog_label_model = MyCNN(len(train_label_set.tag_set))
    
    recog_label_model.to(device)
    optimizer = optim.Adam(recog_label_model.parameters(), lr=learning_rate)

    loss_hist = []
    all_acc = []
    for epoch in range(50):
        all_pred = []
        all_tag = []
        all_loss = []
        for idx, tmp_load in enumerate(train_load):
            optimizer.zero_grad()
            input_tensor = tmp_load[0].to(device)
            tag_tensor = tmp_load[1].to(device).squeeze()
            output = recog_label_model(input_tensor)

            predict = output.detach().cpu().topk(1)[1].view(-1).numpy()
            target = tag_tensor.detach().cpu().numpy()

            loss = criterion(output, tag_tensor)
            loss.backward()
            optimizer.step()
            loss_hist.append(loss)
            all_loss.append(loss.detach().cpu())
            all_pred.append(predict)
            all_tag.append(target)
            # break
        all_pred = np.concatenate(all_pred)
        all_tag = np.concatenate(all_tag)
        accuracy = np.mean(all_tag == all_pred)
        all_acc.append(accuracy)
        print('Epoch: %3d Loss: %.4f, Accuracy: %.4f' %
              (epoch, np.mean(all_loss), accuracy))
        torch.save(recog_label_model.state_dict(), output_path)
        
        
class Img2Spec():
    def __init__(self, img_path: str, model_path: str, spec_type: str, out_put_path: str = None, value_threshold : float = 0.05):
        self.img_path = img_path
        self.out_put_path = out_put_path if out_put_path else img_path
        self.spec_type = spec_type
        self.number_model = self.set_number_recognize_model(model_path)
        self.img_list = self.get_img_list()
        use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda:0" if use_cuda else "cpu")
        self.value_threshold = value_threshold

    def set_number_recognize_model(self, model_path):
        model = MyCNN(13)
        # model = nn.DataParallel(model)
        state_dict = torch.load(model_path)
        model.load_state_dict(state_dict)
        model = model.cuda().eval()
        return model

    class singleImg2Spec():
        def __init__(self, img: np.ndarray, model, device, value_threshold):
            # type 1 是有xlabel，有x坐标，有ylabel，有y坐标，无上框，无右框的类型
            # type 2 是无xlabel，有x坐标，无ylabel，无y坐标，无上框，无右框的类型
            self.img_type = "1"
            self.original_img = img
            self.value_threshold = value_threshold
            try:
                self.img_shape = img.shape
            except AttributeError:
                print("ERROR! Parameter <img> should be a numpy array")
            self.thresholded_img = self.thresholding()
            try:
                self.y_splits = self.get_y_label_splits()
                self.x_splits = self.get_x_label_splits()
                self._y_label_area = self.set_y_label_area()
                self._x_label_area = self.set_x_label_area()
                self._content_area = self.set_content_area()
            except:
                self.img_type = '2'
                self.x_splits = self.get_x_label_splits()
                self._x_label_area = self.set_x_label_area()
                self._content_area = self.set_content_area()
                # io.imsave('test.png', self._content_area)
            self.number_model = model
            self.device = device
            self.tag_set = ['0', '1', '2', '3', '4',
                            '5', '6', '7', '8', '9', '.', ' ', '-']

        # 二值化
        def thresholding(self):
            # io.imshow(self.original_img)
            try:
                img_gray = color.rgb2gray(self.original_img[:,:,:3])
            except:
                img_gray = self.original_img
            thresh = filters.threshold_li(img_gray)
            binary = img_gray > thresh
            return binary

        # 获得x坐标在全图中的纵向区域
        def get_x_label_splits(self):
            vertical_distribution = np.sum(~self.thresholded_img, axis=1)
            # print(vertical_distribution)
            x_splits = []
            item = []
            for i in range(len(vertical_distribution)):
                if len(item) > 0 and vertical_distribution[i] == 0:
                    x_splits.append(item)
                    item = []
                try:
                    if vertical_distribution[i] > 0:
                        item.append(i)
                except:
                    if vertical_distribution[i].any() > 0:
                        item.append(i)

            return x_splits

        def set_x_label_area(self):
            if self.img_type == "1":
                return ~self.thresholded_img[self.x_splits[-2], :]
            if self.img_type == "2":
                return ~self.thresholded_img[self.x_splits[-1], :]

        def get_x_label_area(self):
            return self._x_label_area

        # 获得y坐标在全图中的横向区域
        def get_y_label_splits(self):
            horizon_distribution = np.sum(~self.thresholded_img, axis=0)
            y_splits = []
            item = []
            for i in range(len(horizon_distribution)):
                if len(item) > 0 and i == len(horizon_distribution)-1:
                    y_splits.append(item)
                    item = []
                if len(item) > 0 and horizon_distribution[i] == 0:
                    y_splits.append(item)
                    item = []
                try:
                    if horizon_distribution[i] > 0:
                        item.append(i)
                except:
                    if horizon_distribution[i].any() > 0:
                        item.append(i)

            return y_splits

        def set_y_label_area(self):
            if self.img_type == "1":
                return ~self.thresholded_img[:, self.y_splits[1]]

        def get_y_label_area(self):
            return self._y_label_area

        # 获得实际数据区域
        def set_content_area(self):
            if self.img_type == "1":
                # print(self.y_splits)
                content = self.thresholded_img[:self.x_splits[-3][-1], :]
                content = content[:, self.y_splits[2][0]:]
                return ~content
            if self.img_type == "2":
                content = self.thresholded_img[:self.x_splits[-2][-1], :]
                return ~content

        def get_content_area(self):
            io.imsave("temp.png", self._content_area)
            return self._content_area

        @staticmethod
        def resizefig(array, x_bias, y_bias, canvas_size=1000):
            x, y = array.shape
            canvas = np.zeros((canvas_size, canvas_size))

            canvas[x_bias:x+x_bias, y_bias:y+y_bias] = array
            return canvas

        def get_x_numbers(self):
            kernel = mo.disk(3)
            dilated = mo.binary_dilation(self._x_label_area, kernel)
            # split xLabel
            horizon_distribution = np.sum(dilated, axis=0)
            x_label_splits = []
            item = []
            for i in range(len(horizon_distribution)):
                if len(item) > 0 and horizon_distribution[i] == 0:
                    x_label_splits.append(item)
                    item = []
                if horizon_distribution[i] > 0:
                    item.append(i)
            # x_label_splits : the numbers x list
            xLabels = []
            for i in x_label_splits:
                xLabels.append(self._x_label_area[:, i])

            x_Numbers = []
            for i in xLabels:
                x_numbers = []
                horizon_distribution = np.sum(i, axis=0)
                item = []
                for j in range(len(horizon_distribution)):
                    if len(item) > 0 and horizon_distribution[j] == 0:
                        x_numbers.append(item)
                        item = []
                    if horizon_distribution[j] > 0:
                        item.append(j)
                x_Numbers.append(x_numbers)
            x_Numbers_resized = []
            for i in range(len(x_Numbers)):
                x_numbers_resized = []
                for j in range(len(x_Numbers[i])):
                    x_numbers_resized.append(self.resizefig(tr.rescale(
                        xLabels[i][:, x_Numbers[i][j]], 2), 10, 10, 70))
                x_Numbers_resized.append(x_numbers_resized)
            x_recognized = []
            for i in x_Numbers_resized:
                x_recognized_number = []
                for j in i:
                    input_tensor = torch.tensor(np.expand_dims(
                        j, 0), dtype=torch.float32).to(self.device)
                    output = self.number_model(input_tensor)
                    predict = output.detach().cpu().topk(1)[1].view(-1).numpy()
                    x_recognized_number.append(self.tag_set[predict[0]])
                digit = "".join(x_recognized_number)
                # print(digit)
                x_recognized.append(digit)
            x_label_formated = []
            for i in range(len(x_recognized)):
                try:
                    x_label_formated.append([[x_label_splits[i][0], self.x_splits[-2][0]],
                                             [x_label_splits[i][-1],
                                                 self.x_splits[-2][0]],
                                             [x_label_splits[i][0],
                                                 self.x_splits[-2][-1]],
                                             [x_label_splits[i][-1],
                                                 self.x_splits[-2][-1]],
                                             eval(x_recognized[i])])
                except:
                    x_label_formated.append([[x_label_splits[i][0], self.x_splits[-2][0]],
                                             [x_label_splits[i][-1],
                                                 self.x_splits[-2][0]],
                                             [x_label_splits[i][0],
                                                 self.x_splits[-2][-1]],
                                             [x_label_splits[i][-1],
                                                 self.x_splits[-2][-1]],
                                             0])
            return x_label_formated

        def get_y_numbers(self):
            y_rotated = tr.rotate(self._y_label_area, angle=-90, resize=True)
            y_rotated = np.array(y_rotated, dtype=bool)
            kernel = mo.disk(2)
            dilated = mo.binary_dilation(y_rotated, kernel)
            # split yLabel
            horizon_distribution = np.sum(dilated, axis=0)
            y_label_splits = []
            item = []
            for i in range(len(horizon_distribution)):
                if len(item) > 0 and horizon_distribution[i] == 0:
                    y_label_splits.append(item)
                    item = []
                if horizon_distribution[i] > 0:
                    item.append(i)
            # y_label_splits : the numbers y list
            yLabels = []
            for i in y_label_splits:
                yLabels.append(y_rotated[:, i])

            y_Numbers = []
            for i in yLabels:
                y_numbers = []
                horizon_distribution = np.sum(i, axis=0)
                item = []
                for j in range(len(horizon_distribution)):
                    if len(item) > 0 and horizon_distribution[j] == 0:
                        y_numbers.append(item)
                        item = []
                    if horizon_distribution[j] > 0:
                        item.append(j)
                y_Numbers.append(y_numbers)
            y_Numbers_resized = []
            for i in range(len(y_Numbers)):
                y_numbers_resized = []
                for j in range(len(y_Numbers[i])):
                    y_numbers_resized.append(self.resizefig(tr.rescale(
                        yLabels[i][:, y_Numbers[i][j]], 2), 10, 10, 70))
                y_Numbers_resized.append(y_numbers_resized)
            y_recognized = []
            for i in y_Numbers_resized:
                y_recognized_number = []
                for j in i:
                    input_tensor = torch.tensor(np.expand_dims(
                        j, 0), dtype=torch.float32).to(self.device)
                    output = self.number_model(input_tensor)
                    predict = output.detach().cpu().topk(1)[1].view(-1).numpy()
                    y_recognized_number.append(self.tag_set[predict[0]])
                digit = "".join(y_recognized_number)
                # print(digit)
                y_recognized.append(digit)
            y_label_formated = []
            for i in range(len(y_recognized)-1, -1, -1):
                c, d = self.thresholded_img.shape
                b, a = c-y_label_splits[i][0], c-y_label_splits[i][-1]
                y_label_formated.append([[a, self.y_splits[1][0]],
                                        [b, self.y_splits[1][0]],
                                        [a, self.y_splits[1][-1]],
                                        [b, self.y_splits[1][-1]],
                                        eval(y_recognized[i])])
            return y_label_formated

        def get_spec(self):
            # io.imsave("temp.png", self.thresholded_img)
            if self.img_type == '1':
                horizon_distribution = np.sum(self._content_area, axis=0)
                y_label_bulge = []
                temp = horizon_distribution[0]
                for i in range(len(horizon_distribution)):
                    if horizon_distribution[i] == temp:
                        y_label_bulge.append(i)
                    else:
                        break
                # print(y_label_bulge)
                y_label_formated = self.get_y_numbers()
                vertical_distribution = np.sum(
                    self._content_area[:, y_label_bulge], axis=1)
                y_label_bulge_y = []
                j = 0
                for i in range(len(vertical_distribution)):
                    if vertical_distribution[i] and vertical_distribution[i-1] == 0 and j < len(y_label_formated):
                        y_label_bulge_y.append(i)
                        bottom = i
                        j += 1

                # print("y_label_formated", y_label_formated)
                # print("y_label_bulge_y", y_label_bulge_y)
                y_coordinates = {}
                i = 0
                j = 0
                while True:
                    if y_label_formated[i][0][0] <= y_label_bulge_y[j] <= y_label_formated[i][1][0]:
                        y_coordinates[y_label_bulge_y[j]
                                      ] = y_label_formated[i][-1]
                        i += 1

                    j += 1
                    if j > len(y_label_bulge_y) - 1:
                        break
                vertical_distribution = np.sum(self._content_area, axis=1)
                x_label_bulge = []
                temp = vertical_distribution[-1]
                for i in range(len(vertical_distribution)-1, -1, -1):
                    if vertical_distribution[i] == temp:
                        x_label_bulge.append(i)
                    else:
                        break
                horizon_distribution = np.sum(
                    self._content_area[x_label_bulge, :], axis=0)
                x_label_bulge_x = []
                for i in range(len(horizon_distribution)):
                    if horizon_distribution[i] and horizon_distribution[i-1] == 0:
                        x_label_bulge_x.append(i)
                x_label_formated = self.get_x_numbers()
                # print("x_label_bulge_x", x_label_bulge_x)
                # print("x_label_formated", x_label_formated)
                x_coordinates = {}
                i = 0
                j = 0
                while True:
                    if x_label_formated[i][0][0] <= x_label_bulge_x[j]+self.y_splits[2][0] <= x_label_formated[i][1][0]:
                        x_coordinates[x_label_bulge_x[j]
                                      ] = x_label_formated[i][-1]
                        i += 1
                    j += 1
                    if i >= len(x_label_formated)-1:
                        break

                x = np.array(list(x_coordinates.keys())).reshape((-1, 1))
                y = np.array(list(x_coordinates.values()))
                # print(y)
                x_label_model = LinearRegression()
                x_label_model.fit(x, y)
                x = np.array(list(y_coordinates.keys())).reshape((-1, 1))
                y = np.array(list(y_coordinates.values()))
                # print(y)
                y_label_model = LinearRegression()
                y_label_model.fit(x, y)
                temp = np.sum(self._content_area, axis=0) - 1
                a, b = self._content_area.shape
                for i in range(len(temp)):
                    if temp[i] < 2:
                        start = i
                        break
                # print(bottom)
                # io.imsave('temp.png', self._content_area)
                content_list = []
                for i in range(start, b):
                    for j in range(a):
                        if self._content_area[j][i]:
                            x, y = x_label_model.predict(np.array(i).reshape(
                                (-1, 1)))[0], y_label_model.predict(np.array(j).reshape((-1, 1)))[0]
                            # print(j)
                            if abs(y) > 1e-1 and j < bottom:
                                content_list.append([x, y])
                            break
                return content_list
            elif self.img_type == '2':
                content_list = []
                vertical_distribution = np.sum(self._content_area, axis=1)
                x_label_bulge = []
                temp = vertical_distribution[-1]
                for i in range(len(vertical_distribution)-1, -1, -1):
                    if vertical_distribution[i] == temp:
                        x_label_bulge.append(i)
                    else:
                        break
                horizon_distribution = np.sum(
                    self._content_area[x_label_bulge, :], axis=0)
                x_label_bulge_x = []
                for i in range(len(horizon_distribution)):
                    if horizon_distribution[i] and horizon_distribution[i-1] == 0:
                        x_label_bulge_x.append(i)
                x_label_formated = self.get_x_numbers()
                # print("x_label_bulge_x", x_label_bulge_x)
                # print("x_label_formated", x_label_formated)
                x_coordinates = {}
                i = 0
                j = 0
                while True:
                    if x_label_formated[i][0][0] <= x_label_bulge_x[j] <= x_label_formated[i][1][0]:
                        x_coordinates[x_label_bulge_x[j]
                                      ] = x_label_formated[i][-1]
                        i += 1

                    j += 1
                    if i > len(x_label_formated)-1 or j > len(x_label_bulge_x)-1:
                        break
                x = np.array(list(x_coordinates.keys())).reshape((-1, 1))
                y = np.array(list(x_coordinates.values()))
                # print(y)
                x_label_model = LinearRegression()
                x_label_model.fit(x, y)
                # print(x_coordinates)

                content_splits = []
                item = []
                for i in range(len(vertical_distribution)):
                    if len(item) > 0 and i == len(vertical_distribution)-1:
                        content_splits.append(item)
                        item = []
                    if len(item) > 0 and vertical_distribution[i] == 0:
                        content_splits.append(item)
                        item = []
                    if vertical_distribution[i] > 0:
                        item.append(i)
                content = self._content_area[content_splits[0], :]
                # io.imsave('temp.png', content)
                a, b = content.shape
                for i in range(b):
                    for j in range(a):
                        if content[j][i]:
                            x, y = x_label_model.predict(
                                np.array(i).reshape((-1, 1)))[0], (a - j) / a
                            if abs(y) > self.value_threshold:
                                content_list.append([x, y])
                            break
                return content_list

    def get_img_list(self):
        files_name = os.listdir(self.img_path)
        img_list = []
        for file_name in files_name:
            if not os.path.isdir(file_name):
                if file_name.split('.')[-1] in ['png', 'jpg', 'gif']:
                    img_list.append('{}/{}'.format(self.img_path, file_name))
        return img_list

    def img_process(self):
        for img_name in self.img_list:
            img = io.imread(img_name)
            spec_name = img_name.split('/')[-1].split('.')[0]
            # print(img_name)
            try:
            # print('processing,', img_name)
                processor = self.singleImg2Spec(
                    img=img, model=self.number_model, device=self.device, value_threshold=self.value_threshold)
                self.save_spec_2_csv(processor.get_spec(), spec_name)

                print('success,', img_name)
            except:
                print("ERROR, `{}` is not standardized".format(img_name))

    def save_spec_2_csv(self, spec, name):
        data = pd.DataFrame(spec)
        # print(data)
        data.columns = ['x', 'y']
        data.to_csv(
            "{}/{}_{}.csv".format(self.out_put_path, name, self.spec_type))