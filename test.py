import os
import sys
import cv2
import numpy as np
import argparse
import logging
from datetime import datetime
from tqdm import tqdm
from retinaface import RetinaFace


logging.basicConfig(level=logging.INFO)
logging.getLogger("TEST_MXNET").setLevel(logging.INFO)
log = logging.getLogger("TEST_MXNET")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--symbol', type=str, help='mxnet model symbol')
    parser.add_argument('-e', '--epoch', type=int, help='mxnet model epoch')
    parser.add_argument('-f', '--image_file', type=str, help='image file for inference test', default="t1.jpg")
    parser.add_argument('-t', '--threshold', type=float, help='detection threshold', default=0.5)
    parser.add_argument('-n', '--num', type=int, help='inference test num', default=1000)
    args = parser.parse_args()

    thresh = args.threshold
    scales = [1024, 1980]

    # MXNET Retinaface network model
    gpuid = 0
    detector = RetinaFace(args.symbol, args.epoch, gpuid, 'net3')

    # Input process
    img = cv2.imread(args.image_file)
    log.info(f"** original image shape: {img.shape}")
    im_shape = img.shape
    target_size = scales[0]
    max_size = scales[1]
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])
    #im_scale = 1.0
    #if im_size_min>target_size or im_size_max>max_size:
    im_scale = float(target_size) / float(im_size_min)
    # prevent bigger axis from being more than max_size:
    if np.round(im_scale * im_size_max) > max_size:
        im_scale = float(max_size) / float(im_size_max)

    log.info(f'** im_scale: {im_scale}')

    scales = [im_scale]
    flip = False
    input_tensor, im_info = detector.preprocess_img(img, scales=scales[0], do_flip=flip)
    net_out = detector.infer(input_tensor)
    faces, landmarks = detector.postprocess(net_out, im_info, scales[0], threshold=thresh, do_flip=flip)
    log.info(f"** face: {faces.shape} / landmarks: {landmarks.shape}")

    # Do inference test
    num = args.num
    log.info(f"** start inference time check {num} tries.")
    pbar = tqdm(range(num), desc='Face detection tests')
    t0 = datetime.now()
    for _ in pbar:
        input_tensor, im_info = detector.preprocess_img(img, scales=scales[0], do_flip=flip)
        net_out = detector.infer(input_tensor)
        faces, landmarks = detector.postprocess(net_out, im_info, scales[0], threshold=thresh, do_flip=flip)
    t1 = datetime.now() - t0
    log.info(f"** {num} times Inference Whole Time: {t1}, {num} times Inference Average Time: {t1 / num}")

    if faces is not None:
        log.info(f'** find {faces.shape[0]} faces')
        for i in range(faces.shape[0]):
            #print('score', faces[i][4])
            box = faces[i].astype(np.int32)
            #color = (255,0,0)
            color = (0, 0, 255)
            cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), color, 2)
            if landmarks is not None:
                landmark5 = landmarks[i].astype(np.int32)
                #print(landmark.shape)
                for l in range(landmark5.shape[0]):
                    color = (0, 0, 255)
                    if l == 0 or l == 3:
                        color = (0, 255, 0)
                    cv2.circle(img, (landmark5[l][0], landmark5[l][1]), 1, color, 2)

        filename = f'{args.symbol}-{args.epoch:04d}.jpg'
        log.info(f'** writing {filename}')
        cv2.imwrite(filename, img)

    sys.exit(0)

if __name__ == '__main__':
    main()