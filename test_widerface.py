from __future__ import print_function

import argparse
import sys
import os
import datetime
import numpy as np
import cv2
from rcnn.logger import logger
#from rcnn.config import config, default, generate_config
#from rcnn.tools.test_rcnn import test_rcnn
#from rcnn.tools.test_rpn import test_rpn
from rcnn.processing.bbox_transform import nonlinear_pred, clip_boxes, landmark_pred
from rcnn.processing.generate_anchor import generate_anchors_fpn, anchors_plane
from rcnn.processing.nms import gpu_nms_wrapper
from rcnn.processing.bbox_transform import bbox_overlaps
from rcnn.dataset import retinaface
from retinaface import RetinaFace


os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'


def parse_args():
    parser = argparse.ArgumentParser(
        description='Test widerface by retinaface detector')
    # general
    parser.add_argument('--network',
                        help='network name',
                        default='net3',
                        type=str)
    parser.add_argument('--dataset',
                        help='dataset name',
                        default='retinaface',
                        type=str)
    parser.add_argument('--image-set',
                        help='image_set name',
                        default='val',
                        type=str)
    parser.add_argument('--root-path',
                        help='output data folder',
                        default='./data',
                        type=str)
    parser.add_argument('--dataset-path',
                        help='dataset path',
                        default='./data/retinaface',
                        type=str)
    parser.add_argument('--gpu',
                        help='GPU device to test with',
                        default=0,
                        type=int)
    # testing
    parser.add_argument('--prefix',
                        help='model to test with',
                        default='',
                        type=str)
    parser.add_argument('--epoch',
                        help='model to test with',
                        default=0,
                        type=int)
    parser.add_argument('--output',
                        help='output folder',
                        default='./wout',
                        type=str)
    parser.add_argument('--nocrop', help='', action='store_true')
    parser.add_argument('--thresh',
                        help='valid detection threshold',
                        default=0.02,
                        type=float)
    parser.add_argument('--mode',
                        help='test mode, 0 for fast, 1 for accurate',
                        default=1,
                        type=int)
    #parser.add_argument('--pyramid', help='enable pyramid test', action='store_true')
    #parser.add_argument('--bbox-vote', help='', action='store_true')
    parser.add_argument('--part', help='', default=0, type=int)
    parser.add_argument('--parts', help='', default=1, type=int)
    args = parser.parse_args()
    return args


detector = None
args = None
imgid = -1


def get_boxes(roi, pyramid):
    global imgid

    im = roi['img_arr']
    do_flip = False
    if not pyramid:     # args.mode=0
        # target_size = 1200
        # max_size = 1600
        #do_flip = True
        # target_size = 1504
        # max_size = 2000
        target_size = 1600
        max_size = 2150
        im_shape = im.shape
        im_size_min = np.min(im_shape[0:2])
        im_size_max = np.max(im_shape[0:2])
        im_scale = float(target_size) / float(im_size_min)
        # prevent bigger axis from being more than max_size:
        if np.round(im_scale * im_size_max) > max_size:
            im_scale = float(max_size) / float(im_size_max)
        scales = [im_scale]
    else:
        do_flip = True
        TEST_SCALES = [500, 800, 1100, 1400, 1700]
        target_size = 800
        max_size = 1200
        im_shape = im.shape
        im_size_min = np.min(im_shape[0:2])
        im_size_max = np.max(im_shape[0:2])
        im_scale = float(target_size) / float(im_size_min)
        # prevent bigger axis from being more than max_size:
        if np.round(im_scale * im_size_max) > max_size:
            im_scale = float(max_size) / float(im_size_max)
        scales = [
            float(scale) / target_size * im_scale for scale in TEST_SCALES
        ]

    # do inference
    boxes, landmarks = detector.detect(im,
                                       threshold=args.thresh,
                                       scales=scales,
                                       do_flip=do_flip)
    # print(boxes.shape, landmarks.shape)

    # if imgid < 0, save test image
    if imgid >= 0 and imgid < 100:
        font = cv2.FONT_HERSHEY_SIMPLEX
        for i in range(boxes.shape[0]):
            box = boxes[i]
            ibox = box[0:4].copy().astype(np.int32)
            cv2.rectangle(im, (ibox[0], ibox[1]), (ibox[2], ibox[3]),
                          (255, 0, 0), 2)
            print('box', ibox)
            # if len(ibox)>5:
            #  for l in range(5):
            #    pp = (ibox[5+l*2], ibox[6+l*2])
            #    cv2.circle(im, (pp[0], pp[1]), 1, (0, 0, 255), 1)
            # blur = box[4]
            # k = "%.3f" % blur
            # cv2.putText(im, k, (ibox[0] + 2, ibox[1] + 14), font, 0.6,
            #             (0, 255, 0), 2)
            # landmarks = box[6:21].reshape( (5,3) )
            if landmarks is not None:
                for l in range(5):
                    color = (0, 255, 0)
                    landmark = landmarks[i][l]
                    pp = (int(landmark[0]), int(landmark[1]))
                    if landmark[1] - 0.5 < 0.0:
                        color = (0, 0, 255)
                    cv2.circle(im, (pp[0], pp[1]), 1, color, 2)
        filename = './testimages/%d.jpg' % imgid
        cv2.imwrite(filename, im)
        print(filename, 'wrote')
        imgid += 1

    return boxes


def test(args):
    print(f'** test with args: {args}')

    # Initialize
    global detector
    output_folder = args.output
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    # Get Network
    detector = RetinaFace(args.prefix, args.epoch, args.gpu,
                          network=args.network,
                          nocrop=args.nocrop,
                          vote=args.bbox_vote)

    # Create Dataset class
    imdb = eval(args.dataset)(args.image_set, args.root_path, args.dataset_path)
    roidb = imdb.gt_roidb()

    overall = [0.0, 0.0, 0]
    overall_05 = [0.0, 0.0, 0]

    num_pos = 0
    print(f'** roidb size: {len(roidb)}')

    # Metric
    iouv = np.linspace(0.5, 0.95, 10)
    output_file = os.path.join(output_folder,
                               (datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")) + '.txt')
    for i in range(len(roidb)):
        if i % args.parts != args.part:
            continue

        roi = roidb[i]
        boxes = get_boxes(roi, args.pyramid)
        if 'boxes' in roi and roi['boxes'].size > 0 and boxes.size > 0:
            gt_boxes = roi['boxes'].copy()
            num_pos += gt_boxes.shape[0]

            overlaps = bbox_overlaps(boxes.astype(np.float), gt_boxes.astype(np.float))
            # print(gt_boxes.shape, boxes.shape, overlaps.shape, file=sys.stderr)
            _gt_overlaps = np.zeros((gt_boxes.shape[0]))

            pred_overlaps = overlaps.max(axis=1)
            true_overlaps = overlaps.max(axis=0)

            if boxes.shape[0] > 0:
                with open(output_file, 'a') as f:
                    for iou in iouv:
                        ap_tp = (pred_overlaps > iou).sum()
                        ap_fp = len(pred_overlaps) - ap_tp

                        ar_tp = (true_overlaps > iou).sum()
                        ar_fn = len(true_overlaps) - ar_tp

                        precision = ap_tp / len(pred_overlaps)
                        recall = ar_tp / len(true_overlaps)

                        if iou == 0.5:
                            overall_05[0] += precision
                            overall_05[1] += recall
                            overall_05[2] += 1

                            map50 = overall_05[0] / overall_05[2]
                            mar50 = overall_05[1] / overall_05[2]

                            name = '/'.join(roidb[i]['image'].split('/')[-2:])
                            f.write(f"[{i}] {name} ")
                            f.write(
                                f"precision: {precision:.4f} / recall: {recall:.4f} / mAP 0.5: {map50:.4f} / mAR 0.5: {mar50:.4f}\n")

                        overall[0] += precision
                        overall[1] += recall
                        overall[2] += 1

                    map5095 = overall[0] / overall[2]
                    mar5095 = overall[1] / overall[2]
                    print(f"{[i]} mAP 0.5: {map50:.4f} / mAR 0.5: {mar50:.4f} / mAP 0.5:0.95: {map5095:.4f} / mAR 0.5:0.95: {mar5095:.4f}",
                          file=sys.stderr)

                    # Write inference result to txt file
                    if i % 10 == 0 or i == len(roidb) - 1:
                        f.write(
                            f"{[i]} mAP 0.5: {map50:.4f} / mAR 0.5: {mar50:.4f} / mAP 0.5:0.95: {map5095:.4f} / mAR 0.5:0.95: {mar5095:.4f}\n")

        else:
            print('[%d]' % i, 'detect %d faces' % boxes.shape[0])


def main():
    global args
    args = parse_args()
    args.pyramid = False
    args.bbox_vote = False
    if args.mode == 1:
        args.pyramid = True
        args.bbox_vote = True
    elif args.mode == 2:
        args.pyramid = True
        args.bbox_vote = False
    logger.info(' ** Called with argument: %s' % args)
    test(args)


if __name__ == '__main__':
    main()
