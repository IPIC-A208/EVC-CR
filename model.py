import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
from Dolores import *
from dataloader import FrameDataset, MultiFrameDataset
import time

from utils import print_info
from utils import record_info
from utils import get_time_str
from utils import YUVread

from cut_frame import *

from tensorboardX import SummaryWriter



class MyDolores:
    def __init__(self, model, count_B, channel, QP, video_name, codec_mode, neighbor_frames,height, width,frame_num, start_frame, use_BN_at_begin=True, use_BN_in_ru=True, use_BN_at_end=True):

        # Run flags:
        self.MODE_TRAIN = 'TRAIN'
        self.MODE_EVAL = 'EVAL'
        self.MODE_PREDICT = 'PREDICT'
        self.info_top = {}

        # Hyper-parameters:
        self.model = model
        self.count_B = count_B
        self.channel = channel
        self.recurrent_time = 9
        self.use_BN_at_begin = use_BN_at_begin
        self.use_BN_in_ru = use_BN_in_ru
        self.use_BN_at_end = use_BN_at_end
        self.codec_mode = codec_mode
        self.neighbor_frames = neighbor_frames
        self.net_name = self.model + '_B' + str(self.count_B) + 'C' + str(self.channel) + '_' + self.codec_mode
        self.info_top['net_name'] = self.net_name
        self.info_top['use_BN_at_begin'] = str(self.use_BN_at_begin)
        self.info_top['use_BN_in_ru'] = str(self.use_BN_in_ru)
        self.info_top['use_BN_at_end'] = str(self.use_BN_at_end)

        # Video information:
        self.QP = QP
        self.video_name = video_name
        self.height = height
        self.width = width
        self.frame_num = frame_num
        self.start_frame = start_frame
        self.info_top['video_name'] = self.video_name
        self.info_top['video_size'] = str(self.width) + 'x' + str(self.height)
        self.info_top['QP'] = str(self.QP)


        print('\n\n********** MyDolores **********')
        print_info([self.info_top])
        print('********** *********** **********')

    def checkpoint(self, model, epoch, optimizer, psnr_gain, backup_dir):
        checkpoint_name = 'epoch_{}_psnrgain_{:.6f}_{}_QP{}_start{}_frame{}.pth'.format (epoch, psnr_gain,self.video_name,self.QP, self.start_frame,self.frame_num)

        # state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}
        state = {'state': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}
        torch.save(state, backup_dir + '/' + checkpoint_name)
        print("Checkpoint saved to {}".format(checkpoint_name))


    def train(self,
              time_str,
              patch_mode=None,
              patch_size=60,
              step=36,
              patch_height=1080,
              patch_width=1920,
              train_batch_size=64,
              max_epoch=2400,
              learning_rate=0.001,
              resume=None
              ):

        time_str = time_str or get_time_str()

        log_dir = os.path.join('./logs', self.net_name, self.video_name + '_QP' + str(self.QP), time_str)
        backup_dir = os.path.join('./checkpoints', self.net_name, self.video_name + '_QP' + str(self.QP), time_str)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        if not os.path.exists(backup_dir):
            os.makedirs(backup_dir)

        writer = SummaryWriter(logdir=log_dir)

        train_input_frame = './data/HM_compressed/' + self.video_name + '_QP' + str(self.QP) + '_' + self.codec_mode + '_rec_HM.yuv'
        train_label_frame = './data/raw/' + self.video_name + '.yuv'

        print('\n')
        print('===> Loading datasets')
        
        im_input, _, _ = YUVread(train_input_frame, [self.height, self.width],self.frame_num,self.start_frame)
        im_label, _, _ = YUVread(train_label_frame, [self.height, self.width],self.frame_num,self.start_frame)

        frame_num = im_input.shape[0]
        if patch_mode == 'small':
            train_set = MultiFrameDataset(rec_y=im_input, label_y=im_label, totalFrames=frame_num, nFrames=self.neighbor_frames, width=self.width, height=self.height, width_cut=patch_size, height_cut=patch_size)
            total_count = train_set.__len__()
        else:
            train_set = MultiFrameDataset(rec_y=im_input, label_y=im_label, totalFrames=frame_num, nFrames=self.neighbor_frames, width=self.width, height=self.height,width_cut=self.width, height_cut=self.height)
            total_count = train_set.__len__()

        if patch_mode == 'small':
            training_data_loader = DataLoader(train_set, batch_size=train_batch_size, shuffle=True, num_workers=4)
        else:
            training_data_loader = DataLoader(train_set, batch_size=train_batch_size, shuffle=False, num_workers=4)
        print('===> Done\n')

        print('===> Building model ')

        model = CRNN(input_channel=1, base_channel=self.channel, neighbor_frames=self.neighbor_frames, use_norm_at_begin=self.use_BN_at_begin, use_norm_in_ru=self.use_BN_in_ru, use_norm_at_end=self.use_BN_at_end)
        calculate_variables(model, print_vars=False)
        model = model.cuda()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-8)
        l1_loss_fn = nn.L1Loss()
        l2_loss_fn = nn.MSELoss(reduction='elementwise_mean')
        l1_loss_fn = l1_loss_fn.cuda()
        l2_loss_fn = l2_loss_fn.cuda()
        print('===> Done\n')


        print('===> Try resume from checkpoint')
        if resume != 'none':
            checkpoint = torch.load( resume)
            model.load_state_dict(checkpoint['state'])
            if patch_mode == 'large':
                start_epoch = 1
            else:
                optimizer.load_state_dict(checkpoint['optimizer'])
                start_epoch = checkpoint['epoch']
            print(resume.split('_')[-7])
            psnr_gain_max = float(resume.split('_')[-7])
            print('===> Load checkpoint')
        else:
            start_epoch = 1
            psnr_gain_max = 0.0
            print('===> Start from scratch')

        # info:
        self.info_train = {}
        if resume != 'none':
            self.info_train['checkpoint_to_load'] =  resume
        self.info_train['time_str'] = time_str
        self.info_train['max_epoch'] = max_epoch
        self.info_train['learning_rate'] = str(learning_rate)
        self.info_train['num_of_patches'] = str(total_count)
        if patch_mode == 'small':
            self.info_train['patch_size'] = str(patch_size) + 'x' + str(patch_size)
        else:
            self.info_train['patch_size'] = str(patch_height) + 'x' + str(patch_width)
        self.info_train['train_batch_size'] = str(train_batch_size)
        self.info_train['log_dir'] = log_dir
        self.info_train['backup_dir'] = backup_dir
        self.info_train['train_input'] = train_input_frame
        self.info_train['train_label'] = train_label_frame
        self.info_train['loss_function'] = 'L1-absolute_difference'
        print('\n\n********** Train **********')
        print_info([self.info_train])
        print('********** ***** **********')
        record_info([self.info_top, self.info_train], os.path.join(backup_dir, 'info.txt'))
        record_info([self.info_top, self.info_train], os.path.join(log_dir, 'info.txt'))

        count = 0

        for epoch in range(start_epoch, max_epoch+1):
            # global psnr_gain_max
            model.train()
            psnr_gain = 0.0
            total_psnr_before = 0.0
            for iteration, batch in enumerate(training_data_loader):
                batch_input, batch_neighor, batch_label = batch[0], batch[1], batch[2]
                batch_input = batch_input.cuda()
                batch_neighor = batch_neighor.cuda()
                batch_label = batch_label.cuda()
                batch_output = model(batch_input, batch_neighor)
                mse_loss_before = l2_loss_fn(batch_input, batch_label)
                l1_loss = l1_loss_fn(batch_output, batch_label)
                mse_loss = l2_loss_fn(batch_output, batch_label)
                optimizer.zero_grad()
                l1_loss.backward()
                optimizer.step()

                with torch.no_grad():
                    psnr_before = np.multiply(10.0, np.log(1.0 * 1.0 / mse_loss_before.cpu()) / np.log(10.0))
                    psnr = np.multiply(10.0, np.log(1.0 * 1.0 / mse_loss.cpu()) / np.log(10.0))
                    psnr_gain += (psnr - psnr_before)

                    print(
                        "Train(%.10s:QP%.2d):> Epoch[%.4d](%.3d/%.3d)==  lr: %.8f  train-loss: %.10f  train_PSNR: %.6f  PSNR_before: %.6f  PSNR_gain: %.6f" %
                        (self.video_name,self.QP,epoch, iteration + 1, len(training_data_loader), optimizer.param_groups[0]['lr'], mse_loss.cpu(), psnr,
                         psnr_before, psnr - psnr_before))
                    total_psnr_before += psnr_before
                writer.add_scalar('Train_loss', l1_loss.cpu(), count)
                writer.add_scalar('Train_PSNR', psnr, count)

            total_psnr_before = total_psnr_before / (len(training_data_loader))
            print(total_psnr_before)
            psnr_gain = psnr_gain / (len(training_data_loader))
            self.checkpoint(model, epoch, optimizer, psnr_gain_max, backup_dir=backup_dir)

            if epoch % 50 == 0:
                self.checkpoint(model, epoch, optimizer, psnr_gain, backup_dir=backup_dir)

            if self.QP in [22, 27]:
                if (epoch + 1) == 50 or (epoch + 1) == 300:
                    for param_group in optimizer.param_groups:
                        param_group['lr'] /= 10
                    print('Learning rate decay: lr={}'.format(optimizer.param_groups[0]['lr']))
            else:
                if (epoch + 1) == 100 or (epoch + 1) == 300:
                    for param_group in optimizer.param_groups:
                        param_group['lr'] /= 10
                    print('Learning rate decay: lr={}'.format(optimizer.param_groups[0]['lr']))

    def evaluate(self,
                    ckpt,
                    height=1080,
                    width=1920,
                    need_output=False):


        output_path = './data/videos/' + self.video_name + '_QP' + str(self.QP) + '_' + self.codec_mode + '_rec_HM_CRNN.yuv'

        input_path = './data/HM_compressed/' + self.video_name + '_QP' + str(self.QP) + '_' + self.codec_mode + '_rec_HM.yuv'
        label_path = './data/raw/' + self.video_name + '.yuv'

        print('\n')
        print('===> Loading datasets')
        im_input, U, V = YUVread(input_path, [self.height, self.width],self.frame_num,self.start_frame)
        im_label, _, _ = YUVread(label_path, [self.height, self.width],self.frame_num,self.start_frame)
        frame_num = im_input.shape[0]
        eval_set = MultiFrameDataset(rec_y=im_input, label_y=im_label, totalFrames=frame_num,nFrames=self.neighbor_frames, width=self.width, height=self.height, width_cut=self.width, height_cut=self.height)
        eval_data_loader = DataLoader(dataset=eval_set, num_workers=0, drop_last=False, batch_size=1, shuffle=False)
        print('===> Done\n')

        print('===> Building model ')
        
        model = CRNN(input_channel=1, base_channel=self.channel, neighbor_frames=self.neighbor_frames, use_norm_at_begin=self.use_BN_at_begin, use_norm_in_ru=self.use_BN_in_ru, use_norm_at_end=self.use_BN_at_end)
        model.cuda()
        calculate_variables(model, print_vars=False)
        print('===> Done\n')

        print('===> Initialize and prepare...')
        checkpoint = torch.load(ckpt)
        model.load_state_dict(checkpoint['state'])
        print('===> Done\n')

        # info:
        self.info_evaluate = {}
        self.info_evaluate['image_size'] = str(height) + 'x' + str(width)
        self.info_evaluate['input_data'] = input_path
        self.info_evaluate['label_data'] = label_path
        self.info_evaluate['checkpoint'] = ckpt
        if need_output:
            self.info_evaluate['output_path'] = output_path
        print('\n\n********** evaluate **********')
        print_info([self.info_evaluate])
        print('********** ******** **********')

        loss_fn = nn.MSELoss(reduction='elementwise_mean')
        times_diff = []  
        with torch.no_grad():
            model.eval()
            predictions = []
            total_psnr_before, total_psnr_after, total_gain = 0, 0, 0
            # start_time = time.time()
            for n_count, batch in enumerate(eval_data_loader):
                batch_input, batch_neighor, batch_label = batch[0], batch[1], batch[2]
                batch_input = batch_input.cuda()
                batch_neighor = batch_neighor.cuda()
                batch_label = batch_label.cuda()
                start_time = time.time()
                batch_output = model(batch_input, batch_neighor)
                end_time = time.time()
                time_diff = end_time - start_time
                times_diff.append(time_diff)

                MSE_before = loss_fn(batch_input, batch_label)
                MSE_after = loss_fn(batch_output, batch_label)
                prediction = np.transpose(batch_output.cpu().numpy(), (0, 2, 3, 1))
                predictions.extend(prediction)
                PSNR_before = np.multiply(10.0, np.log(1.0 * 1.0 / MSE_before.cpu()) / np.log(10.0))
                PSNR_after = np.multiply(10.0, np.log(1.0 * 1.0 / MSE_after.cpu()) / np.log(10.0))

                PSNR_gain = PSNR_after - PSNR_before

                print('Frame%2d/%2d   PSNR_before : %.4f  PSNR_after : %.4f  PSNR_gain: %.4f time: %.6f ms' % ((n_count + 1), len(eval_data_loader), PSNR_before, PSNR_after, PSNR_gain, (end_time-start_time)*1000))
                total_psnr_before += PSNR_before
                total_psnr_after += PSNR_after
                total_gain += PSNR_gain

            time_diff_avg = np.mean(np.array(times_diff))
            print('*' * 50)
            print('Average PSNR before Restoring : %.4f' % (total_psnr_before / len(eval_data_loader)))
            print('Average PSNR after Restoring : %.4f' % (total_psnr_after / len(eval_data_loader)))
            print('Average PSNR Gain : %.4f' % (total_gain / len(eval_data_loader)))
            print('Total Time : %.6f ms' % (time_diff_avg*1000))
            if need_output:
                print('\n** Write results...')
                Y = np.array(predictions).squeeze()
                Y = np.clip(np.round(255.0 * Y), 0, 255).astype(np.uint8)
                YUVwrite(Y, U, V, output_path)
                print('Output path:', output_path)
                print('Done.\n')
