import argparse
import os

from model import MyDolores

parser = argparse.ArgumentParser()
parser.add_argument('-g', '--GPU', default='0', help='the GPU No. to use')
subparsers = parser.add_subparsers(dest='mode')

# train mode:
parser_train = subparsers.add_parser('train', help='Train the net. Type "python run.py train -h" for more information.')
parser_train.add_argument('-m', '--model', default='FRCNN', type=str, help='choose model, like:FRCNN, FRCNN_vconv(default FRCNN)')
parser_train.add_argument('-b', '--B', required=True, type=int, help='the number of blocks')
parser_train.add_argument('-c', '--C', required=True, type=int, help='the depth feature maps')
parser_train.add_argument('-v', '--video', required=True, help='video name, like: "BasketballDrive_1920x1080_50_000to049"')
parser_train.add_argument('-q', '--qp', required=True, type=int, help='codex QP')
parser_train.add_argument('-t', '--time_str', help='the old time stamp, from which backup_dir to continue')
parser_train.add_argument('--codec_mode', default='IP', type=str, help='IP for lowdelay P; RA for RandomAccess; AI for AllIntra')
parser_train.add_argument('--patch_mode', default='small', type=str, help='patch size,small for 41x41,large for 1080x1920')
parser_train.add_argument('--height', required=True, type=int, help='the height of input image/video')
parser_train.add_argument('--width', required=True, type=int, help='the width of input image/video')
parser_train.add_argument('--frame_num', required=True, type=int, help='frame number of input image/video')
parser_train.add_argument('--start_frame', required=True, type=int, help='the first frame of input image/video')
parser_train.add_argument('--neighbor_frames', required=True, type=int, help='reference frame number of input image/video')
parser_train.add_argument('--train_batch', default=64, type=int, help='train batch size (default: 200)')
parser_train.add_argument('--max_epoch', type=int, default=2000, help='number of epochs to train for')
parser_train.add_argument('--lr', default=0.001, type=float, help='learning rate (default: 0.001)')
parser_train.add_argument('--resume', default='none', type=str, help='resume modle')
parser_train.add_argument('--no_BN_begin', action='store_true', default=False, help='if you do NOT want to use BatchNormalization layer at the beginning block')
parser_train.add_argument('--no_BN_ru', action='store_true', default=False, help='if you do NOT want to use BatchNormalization layer in each residual unit')
parser_train.add_argument('--no_BN_end', action='store_true', default=False, help='if you do NOT want to use BatchNormalization layer at the ending block ')



# evaluate mode:
parser_evaluate = subparsers.add_parser('evaluate', help='Evaluate the net. Type "python run.py evaluate -h" for more information.')
parser_evaluate.add_argument('-m', '--model', default='FRCNN', type=str, help='choose model, like:FRCNN, FRCNN_vconv(default FRCNN)')
parser_evaluate.add_argument('-b', '--B', required=True, type=int, help='the number of blocks')
parser_evaluate.add_argument('-c', '--C', required=True, type=int, help='the depth feature maps')
parser_evaluate.add_argument('-v', '--video', required=True, help='video name, like: "BasketballDrive_1920x1080_50_000to049"')
parser_evaluate.add_argument('--codec_mode', default='IP', type=str, help='IP for lowdelay P; RA for RandomAccess; AI for AllIntra')
parser_evaluate.add_argument('-q', '--qp', required=True, type=int, help='codex QP')
parser_evaluate.add_argument('--height', required=True, type=int, help='the height of input image/video')
parser_evaluate.add_argument('--width', required=True, type=int, help='the width of input image/video')
parser_evaluate.add_argument('--frame_num', required=True, type=int, help='frame number of input image/video')
parser_evaluate.add_argument('--start_frame', required=True, type=int, help='the first frame of input image/video')
parser_evaluate.add_argument('--neighbor_frames', required=True, type=int, help='reference frame number of input image/video')
parser_evaluate.add_argument('--ckpt', required=True, help='the path of checkpoint')
parser_evaluate.add_argument('-o', '--output', action='store_true', default=False, help='whether to output the results')
parser_evaluate.add_argument('--no_BN_begin', action='store_true', default=False, help='if you do NOT want to use BatchNormalization layer at the beginning block')
parser_evaluate.add_argument('--no_BN_ru', action='store_true', default=False, help='if you do NOT want to use BatchNormalization layer in each residual unit')
parser_evaluate.add_argument('--no_BN_end', action='store_true', default=False, help='if you do NOT want to use BatchNormalization layer at the ending block ')

args = parser.parse_args()
if args.mode is None:
    parser.print_help()
    exit()

os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = args.GPU
print('\n** GPU selection:', os.environ["CUDA_VISIBLE_DEVICES"])

dolores = MyDolores(
    model=args.model,
    count_B=args.B,
    channel=args.C,
    QP=args.qp,
    height=args.height,
    width=args.width,
    video_name=args.video,
    codec_mode=args.codec_mode,
    frame_num=args.frame_num,
    start_frame=args.start_frame,
    neighbor_frames = args.neighbor_frames,
    use_BN_at_begin=(not args.no_BN_begin),
    use_BN_in_ru=(not args.no_BN_ru),
    use_BN_at_end=(not args.no_BN_end)
)


if args.mode == 'train':
    dolores.train(
        patch_mode=args.patch_mode,
        time_str=args.time_str,
        max_epoch=args.max_epoch,
        learning_rate=args.lr,
        resume=args.resume
    )

else:
    dolores.evaluate(
       ckpt=args.ckpt,
       height=args.height,
       width=args.width,
       need_output=args.output
   )