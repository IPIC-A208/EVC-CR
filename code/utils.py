import numpy as np
import cv2
import os
import time
import pickle
import random


def print_info(info_list):
    for info in info_list:
        for key, value in info.items():
            print(key, ':', value)


def record_info(info_list, file_path):
    with open(file_path, 'a') as file:
        file.write('\n\n*** log_time ' + get_time_str() + ' ***\n')
        for info in info_list:
            for key, value in info.items():
                file.write(str(key) + ' : ' + str(value) + '\n')


def progress_bar(done_num, total_num, length=40):
    if total_num == 0:
        print('Error: Total number is 0.')
        return
    percent = done_num / total_num
    show = int(length * percent)
    bar = '|' + '=' * show + ' ' * (length - show) + '| (' + \
          str(int(percent * 100)) + '%) ' + str(done_num) + ' of ' + str(total_num) + ' ' * 10
    print(bar, end='\r')


def get_time_str():
    return time.strftime("%Y%m%d%H%M%S", time.localtime())


# def cv_imread(file_path, channel='BGR'):
#     """
#     mode: 'BGR' or 'RGB'
#     """
#     img = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), -1)
#     if channel == 'RGB':
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     return img


# def cv_imwrite(output_path, img, channel='BGR', ext='.png'):
#     """
#     image: 8-bit single-channel or 3-channel (with 'BGR' channel order) images
#     """
#     if channel == 'RGB':
#         img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
#     cv2.imencode(ext, img)[1].tofile(output_path)


def YUVread(file, size, frame_num=None, start_frame=0, mode='420'):
    """
    Only for 4:2:0 and 4:4:4 for now.

    :param file: yuv file
    :param size: [height, width]
    :param frame_num: The number of frames you want to read, and it shouldn't smaller than the frame number of original
        yuv file. Defult is None, means read from start_frame to the end of file.
    :param start_frame: which frame begin from. Default is 0.
    :param mode: yuv file mode, '420' or '444 planar'
    :return: byte_type y, u, v with a shape of [frame_num, height, width] of each
    """
    [height, width] = size
    if mode == '420':
        frame_size = int(height * width / 2 * 3)
    else:
        frame_size = int(height * width * 3)
    all_y = np.uint8([])
    all_u = np.uint8([])
    all_v = np.uint8([])
    # with open(path, 'rb') as file:
    file.seek(frame_size * start_frame)
    if frame_num is None:
        frame_num = 0
        while True:
            if mode == '420':
                y = np.uint8(list(file.read(height * width)))
                u = np.uint8(list(file.read(height * width >> 2)))
                v = np.uint8(list(file.read(height * width >> 2)))
            else:
                y = np.uint8(list(file.read(height * width)))
                u = np.uint8(list(file.read(height * width)))
                v = np.uint8(list(file.read(height * width)))
            if y.shape == (0,):
                break
            all_y = np.concatenate([all_y, y])
            all_u = np.concatenate([all_u, u])
            all_v = np.concatenate([all_v, v])
            frame_num += 1
    else:
        for fn in range(frame_num):
            if mode == '420':
                y = np.uint8(list(file.read(height * width)))
                u = np.uint8(list(file.read(height * width >> 2)))
                v = np.uint8(list(file.read(height * width >> 2)))
            else:
                y = np.uint8(list(file.read(height * width)))
                u = np.uint8(list(file.read(height * width)))
                v = np.uint8(list(file.read(height * width)))
            if y.shape == (0,):
                break
            all_y = np.concatenate([all_y, y])
            all_u = np.concatenate([all_u, u])
            all_v = np.concatenate([all_v, v])

    all_y = np.reshape(all_y, [frame_num, height, width])
    if mode == '420':
        all_u = np.reshape(all_u, [frame_num, height >> 1, width >> 1])
        all_v = np.reshape(all_v, [frame_num, height >> 1, width >> 1])
    else:
        all_u = np.reshape(all_u, [frame_num, height, width])
        all_v = np.reshape(all_v, [frame_num, height, width])

    return all_y, all_u, all_v


def Yread(path, size, frame_num=None, start_frame=0):
    """
    Assuming that the file contains only the Y component.

    :param path: yuv file path
    :param size: [height, width]
    :param frame_num: The number of frames you want to read, and it shouldn't smaller than the frame number of original
        yuv file. Defult is None, means read from start_frame to the end of file.
    :param start_frame: Which frame begin from. Default is 0.
    :return: byte_type y with a shape of [frame_num, height, width]
    """
    [height, width] = size
    frame_size = int(height * width)
    all_y = np.uint8([])
    with open(path, 'rb') as file:
        file.seek(frame_size * start_frame)
        if frame_num is None:
            frame_num = 0
            while True:
                y = np.uint8(list(file.read(height * width)))
                if y.shape == (0,):
                    break
                all_y = np.concatenate([all_y, y])
                frame_num += 1
        else:
            for fn in range(frame_num):
                y = np.uint8(list(file.read(height * width)))
                if y.shape == (0,):
                    break
                all_y = np.concatenate([all_y, y])

    all_y = np.reshape(all_y, [frame_num, height, width])

    return all_y


def YUVwrite(y, u, v, path):
    """
    Ndarray to file. If '444', write by YUV444 planar mode.

    :param y: y with a shape of [frame_num, height, width] or [height, width]
    :param u: u with a shape of [frame_num, height, width] or [height, width]
    :param v: v with a shape of [frame_num, height, width] or [height, width]
    :param path: save path
    """
    if len(np.shape(y)) == 3:
        frame_num = np.shape(y)[0]
        with open(path, 'wb') as file:
            for fn in range(frame_num):
                file.write(y[fn].tobytes())
                file.write(u[fn].tobytes())
                file.write(v[fn].tobytes())
    else:
        with open(path, 'wb') as file:
            file.write(y.tobytes())
            file.write(u.tobytes())
            file.write(v.tobytes())


def Ywrite(y, path):
    """
    Only Y channel.

    :param y: y with a shape of [frame_num, height, width] or [height, width]
    :param path: save path
    """
    if len(np.shape(y)) == 3:
        frame_num = np.shape(y)[0]
        with open(path, 'wb') as file:
            for fn in range(frame_num):
                file.write(y[fn].tobytes())
    else:
        with open(path, 'wb') as file:
            file.write(y.tobytes())


def YUVcut(y, u, v, new_size, new_frame_num=None, start_frame=0, start_point=(0, 0)):
    """
    Cut frames/patches from yuv. Only for 4:2:0 or 4:4:4.

    :param y: y
    :param u: u
    :param v: v
    :param new_size: [height, width]
    :param new_frame_num: How many frames you want to get.
    :param start_frame: Begin from which frame. Default is 0.
    :param start_point: The left_up point of new patch. Default is (0, 0)
    :return: cut yuv
    """
    [new_height, new_width] = new_size
    [sh, sw] = start_point
    if new_frame_num is None:
        new_frame_num = np.shape(y)[0] - start_frame

    if np.shape(y)[1] == np.shape(u)[1]:  # 444
        new_y = y[start_frame:start_frame + new_frame_num, sh:sh + new_height, sw:sw + new_width]
        new_u = u[start_frame:start_frame + new_frame_num, sh:sh + new_height, sw:sw + new_width]
        new_v = v[start_frame:start_frame + new_frame_num, sh:sh + new_height, sw:sw + new_width]
    else:  # 420
        if new_height % 2 != 0:
            new_height += 1
        if new_width % 2 != 0:
            new_width += 1
        if sh % 2 != 0:
            sh += 1
        if sw % 2 != 0:
            sw += 1
        new_y = y[start_frame:start_frame + new_frame_num, sh:sh + new_height, sw:sw + new_width]
        new_u = u[start_frame:start_frame + new_frame_num, (sh >> 1):((sh >> 1) + (new_height >> 1)),
                (sw >> 1):((sw >> 1) + (new_width >> 1))]
        new_v = v[start_frame:start_frame + new_frame_num, (sh >> 1):((sh >> 1) + (new_height >> 1)),
                (sw >> 1):((sw >> 1) + (new_width >> 1))]

    return new_y, new_u, new_v


# def YUV_change_mode(y, u, v, direction='420to444'):
#     """
#     derection: '420to444' or '444to420'
#     """
#     if direction == '420to444':
#         u = np.array([cv2.resize(ch, (u.shape[2] * 2, u.shape[1] * 2), interpolation=cv2.INTER_CUBIC) for ch in u])
#         v = np.array([cv2.resize(ch, (v.shape[2] * 2, v.shape[1] * 2), interpolation=cv2.INTER_CUBIC) for ch in v])
#     if direction == '444to420':
#         u = np.array([cv2.resize(ch, (u.shape[2] // 2, u.shape[1] // 2), interpolation=cv2.INTER_CUBIC) for ch in u])
#         v = np.array([cv2.resize(ch, (v.shape[2] // 2, v.shape[1] // 2), interpolation=cv2.INTER_CUBIC) for ch in v])
#     return y, u, v


# def save_YUV_img(y, u, v, output_path, mode='420', ext='.png'):
#     if mode == '420':
#         y, u, v = YUV_change_mode(y, u, v, '420to444')
#     if y.shape[0] == 1:
#         img = cv2.cvtColor(np.concatenate([y[0, :, :, np.newaxis], u[0, :, :, np.newaxis], v[0, :, :, np.newaxis]], 2),
#                            cv2.COLOR_YUV2BGR)
#         cv_imwrite(output_path, img, ext=ext)
#     else:
#         path = os.path.splitext(output_path)[0]
#         for fn in range(y.shape[0]):
#             img = cv2.cvtColor(
#                 np.concatenate([y[fn, :, :, np.newaxis], u[fn, :, :, np.newaxis], v[fn, :, :, np.newaxis]], 2),
#                 cv2.COLOR_YUV2BGR)
#             cv_imwrite(path + '_' + str(fn) + ext, img, ext=ext)


def calculate_variables(var_list, print_vars=False):
    all_num = 0
    all_size = 0
    for var in var_list:
        num = 1
        for d in var.shape.as_list():
            num *= d
        all_num += num
        all_size += var.dtype.size * num
        if print_vars:
            print(var.dtype.name, var.dtype.size, var.name, num)
    print()
    print('Variables number : %d' % all_num)
    print('Variables  size  : %d B' % (all_size))
    print('Variables  size  : %d b' % (all_size * 8))
    print('Variables  size  : %f KB' % (all_size / 1024))
    print('Variables  size  : %f Kb' % (all_size * 8 / 1024))
    print('Variables  size  : %f MB' % (all_size / 1024 / 1024))
    print('Variables  size  : %f Mb' % (all_size * 8 / 1024 / 1024))


