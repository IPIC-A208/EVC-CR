import sys
import numpy as np
from yuv_io import *
# import matplotlib.pyplot as plt
import torch





def progress_bar(num, total, width=40):
    rate = num / total
    rate_num = int(rate * width)
    r = '\r[%s%s] %d%%%s%d' % ("=" * rate_num, " " * (width - rate_num), int(rate * 100), ' done of ', total)
    sys.stdout.write(r)
    sys.stdout.flush()


def generate(im_input, im_label, im_size, patch_size, step):
    [hgt, wdt] = im_size
    count_h = int((hgt - patch_size) / step + 1)
    count_w = int((wdt - patch_size) / step + 1)

    input_data = []
    label_data = []
    count = 0

    start_h = 0
    for h in range(count_h):
        start_w = 0
        for w in range(count_w):
            patch_input = im_input[start_h:start_h + patch_size, start_w:start_w + patch_size]
            patch_label = im_label[start_h:start_h + patch_size, start_w:start_w + patch_size]
            input_data.append(patch_input)
            label_data.append(patch_label)
            count += 1
            start_w += step
        start_h += step

    start_h = 0
    for h in range(count_h):
        patch_input = im_input[start_h:start_h + patch_size, - patch_size:]
        patch_label = im_label[start_h:start_h + patch_size, - patch_size:]
        input_data.append(patch_input)
        label_data.append(patch_label)
        count += 1
        start_h += step

    start_w = 0
    for w in range(count_w):
        patch_input = im_input[- patch_size:, start_w:start_w + patch_size]
        patch_label = im_label[- patch_size:, start_w:start_w + patch_size]
        input_data.append(patch_input)
        label_data.append(patch_label)
        count += 1
        start_w += step

    patch_input = im_input[- patch_size:, - patch_size:]
    patch_label = im_label[- patch_size:, - patch_size:]
    input_data.append(patch_input)
    label_data.append(patch_label)
    count += 1

    # return np.expand_dims(np.array(input_data), axis=3), np.expand_dims(np.array(label_data), axis=3), count
    return input_data, label_data, count

def generate_v2(im_input, im_label, im_size, patch_hgt, patch_wdt):
    [hgt, wdt] = im_input.shape
    count_h = int((hgt - patch_hgt) / patch_hgt + 1)
    count_w = int((wdt - patch_wdt) / patch_wdt + 1)

    input_data = []
    label_data = []

    start_h = 0
    for h in range(count_h):
        star_w = 0
        for w in range(count_w):
            patch_input = im_input[start_h:start_h + patch_hgt, star_w:star_w + patch_wdt]
            patch_label = im_label[start_h:start_h + patch_hgt, star_w:star_w + patch_wdt]
            input_data.append(patch_input)
            label_data.append(patch_label)
            star_w += patch_wdt
        start_h += patch_hgt
    count = count_h * count_w
    return input_data, label_data, count

def cut_frame(input_path, label_path, frame_size, frame_num, start_frame, patch_size, step):
    (height, width) = frame_size
    im_input, _, _ = YUVread(input_path, [height, width], frame_num, start_frame)
    im_label, _, _ = YUVread(label_path, [height, width], frame_num, start_frame)

    frame_number = im_input.shape[0]
    input_data, label_data = [], []
    total_count = 0
    for index in range(frame_number):
        input, label, count = generate(im_input[index], im_label[index], [height, width], patch_size, step)
        input_data.extend(input)
        label_data.extend(label)
        total_count += count
    input_data = np.expand_dims(np.array(input_data), axis=3)
    label_data = np.expand_dims(np.array(label_data), axis=3)
    return input_data, label_data, total_count

def cut_frame_v2(input_path, label_path, frame_size, frame_num, start_frame, patch_hgt, patch_wdt):
    (height, width) = frame_size
    im_input, _, _ = YUVread(input_path, [height, width], frame_num, start_frame)
    im_label, _, _ = YUVread(label_path, [height, width], frame_num, start_frame)

    frame_number = im_input.shape[0]
    input_data, label_data = [], []
    total_count = 0
    for index in range(frame_number):
        input, label, count = generate_v2(im_input[index], im_label[index], [height, width],  patch_hgt, patch_wdt)
        input_data.extend(input)
        label_data.extend(label)
        total_count += count
    input_data = np.expand_dims(np.array(input_data), axis=3)
    label_data = np.expand_dims(np.array(label_data), axis=3)
    return input_data, label_data, total_count


if __name__ == '__main__':
    input_path = './data/videos/BasketballDrive_1920x1080_50_000to000_QP22_IP_rec.yuv'
    label_path = './data/videos/BasketballDrive_1920x1080_50_000to000.yuv'


    height = 1080
    width = 1920

    initial_epoch = 0
    n_epoch = 1
    batch_size = 1
    input_data, label_data, total_count = cut_frame_v2(input_path, label_path, [height, width], 540, 960)
    print(input_data.shape)
    print(label_data.shape)
    print(total_count)


    print(input_data[0].squeeze().shape)
    print(label_data[0].squeeze().shape)
    plt.figure()
    plt.imshow(input_data[0].squeeze())
    plt.show()
    plt.figure()
    plt.imshow(label_data[0].squeeze())
    plt.show()

    # im_input, _, _ = YUVread(input_path, [height, width])
    # im_label, _, _ = YUVread(label_path, [height, width])
    #
    # frame_number = im_input.shape[0]
    # input_data, label_data = [], []
    # total_count = 0
    # for index in range(frame_number):
    #     input, label, count = generate(im_input[index], im_label[index], [height, width], 41, 36)
    #     print(np.array(input).shape)
    #     print(np.array(label).shape)
    #     input_data.extend(input)
    #     label_data.extend(label)
    #     total_count += count
    # input_data = np.expand_dims(np.array(input_data), axis=3)
    # label_data = np.expand_dims(np.array(label_data), axis=3)
    # print(input_data.shape)
    # print(label_data.shape)
    # print(total_count)



