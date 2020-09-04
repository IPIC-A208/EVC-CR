import argparse
import torch
from torch.utils.data import DataLoader
from utils import YUVread
from model import Model
from ORNN import ORNN
from data import MDataYUV
import time
import yaml

version = float(torch.__version__[0:3])
if version >= 1.1:
    from torch.utils.tensorboard import SummaryWriter
else:
    from tensorboardX import SummaryWriter





def get_model(args):
	return ORNN(block=args.block, neighbor_frames=args.neighbor, channel=args.channel)
		
	

def get_data(args):
	rec_path='../data/'+args.data+'/'+args.sequence+'_QP'+args.qp+'_rec.yuv'
	label_path='../data/original/'+args.sequence+'.yuv'
	rec_file = open(rec_path, 'rb')
	label_file = open(label_path, 'rb')
	rec_y, _, _ = YUVread(rec_file, [args.heigh, args.width], frame_num=args.frame, start_frame=0)
	label_y, _, _ = YUVread(label_file, [args.heigh, args.width], frame_num=args.frame, start_frame=0)
	return MDataYUV(rec_y=rec_y, label_y=label_y, nFrames=args.neighbor, width=args.width, heigh=args.heigh, totalFrames=args.frame, width_cut=args.width_cut, heigh_cut=args.heigh_cut)



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=1.0e-02, help='learning rate')
    parser.add_argument('--resume', type=str, default='none', help='resume model')
    parser.add_argument('--qp', type=str, default='22', help='training qp')
    parser.add_argument('--decay', type=int, default=80, help='learning rate decay')
    parser.add_argument('--epoch', type=int, default=200, help='all epoch want to run')
    parser.add_argument('--frame', type=int, default=200, help='frames need to test')
    parser.add_argument('--epoch_start', type=int, default=0, help='start epoch')
    parser.add_argument('--batch_size', type=int, default=64, help='batch')
    parser.add_argument('--model', type=str, default='ORNN', help='model name')
    parser.add_argument('--width_cut', type=int, default=64, help='width cut')
    parser.add_argument('--heigh_cut', type=int, default=64, help='heigh cut')
    parser.add_argument('--width', type=int, default=1920, help='width')
    parser.add_argument('--heigh', type=int, default=1080, help='heigh')
    parser.add_argument('--data', type=str, default='HM', help='HM, H266, X265, X264, NVENC')
    parser.add_argument('--sequence', type=str, default='BasketballDrive_1920_1080', help='sequence name')
    parser.add_argument('--neighbor', type=int, default=2, help='neighbor frame number')
    parser.add_argument('--block', type=int, default=4, help='blocks of Model')
    parser.add_argument('--channel', type=int, default=16, help='channels of Model')
    parser.add_argument('--optim', type=str, default='Adam', help='optimizer')
    # parser.add_argument('--block', type=int, default=16, help='channels of Model')
    args = parser.parse_args()
    frame_num = args.frame
    width, heigh = args.width, args.heigh
    sequence = args.sequence

    model_ = get_model(args)

    # checkpoint difference
    if args.optim != 'Adam':
        args.model = args.model + '_' + args.optim
    

    model = Model(lr=args.lr, model=model_, optim=args.optim)
    model.print_network()
    model.epoch = args.epoch_start + 1

    if args.resume != 'none':
        model.resume_network(args.resume)
        print('checkpoint {} has resumed'.format(args.resume))
 
    DataYUV = get_data(args)
    DataLoaderYUV = DataLoader(DataYUV, batch_size=int(args.batch_size), shuffle=True, num_workers=0)
    writer = SummaryWriter('tensorboard/{}_{}_record-QP{}-allFrame{}-{}'.format(args.model, sequence, args.qp, frame_num, time.strftime('%Y%m%d%H%M%S', time.localtime())))

    for epoch in range(args.epoch_start+1, args.epoch):
        for iteration, (input_, neighbor, label_) in enumerate(DataLoaderYUV):
            # print(input_.shape, neighbor.shape,label_.shape)
            model.feed_data(input_, neighbor, label_)
            model.optimize_parameters()
            model.log(iteration, epoch, len(DataLoaderYUV), args, writer=writer)

        if epoch % args.decay == 0:
            model.adajust_lr(0.1)

        






if __name__ == '__main__':
    main()


