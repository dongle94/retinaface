import os
import sys
from datetime import datetime
from tqdm import tqdm
import onnxruntime as ort
import numpy as np
import cv2
import argparse
import logging
from rcnn.processing.generate_anchor import generate_anchors_fpn, anchors_plane
from rcnn.processing.nms import gpu_nms_wrapper, cpu_nms_wrapper


logging.basicConfig(level=logging.INFO)
logging.getLogger("TEST_ONNX").setLevel(logging.INFO)
log = logging.getLogger("TEST_ONNX")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, help='frozen model path', default='./retinaface-0000.onnx')
    parser.add_argument('-f', '--image_file', type=str, help='image file for inference test',
                        default="t1.jpg")
    parser.add_argument('-t', '--threshold', type=float, help='detection threshold', default=0.5)
    parser.add_argument('-n', '--num', type=int, help='inference test num', default=1000)
    args = parser.parse_args()

    EP_list = [
        ('CUDAExecutionProvider', {
         'device_id': 0,
         # 'gpu_mem_limit': 2 * 1024 * 1024 * 1024,
         'arena_extend_strategy': 'kNextPowerOfTwo',
         'cudnn_conv_algo_search': 'EXHAUSTIVE',     # EXHAUSTIVE, HEURISTIC, DEFAULT
         'do_copy_in_default_stream': True,
        }),
        'CPUExecutionProvider',
    ]

    # Make Onnx runtime Session
    sess = ort.InferenceSession(args.input, providers=EP_list)
    input_name = sess.get_inputs()[0].name
    input_shape = sess.get_inputs()[0].shape
    log.info(f"** Input Tensor name / shape : {input_name}, {input_shape}, {sess.get_inputs()[0].type}")
    output_names = [output.name for output in sess.get_outputs()]
    output_shapes = [output.shape for output in sess.get_outputs()]
    for _name, _shape in zip(output_names, output_shapes):
        log.info(f"** Output Tensors name / shape : {_name}, {_shape}, {sess.get_inputs()[0].type}")

    # Check current device
    log.info(f"** Current Inference Device: {ort.get_device()}")

    # imread test image
    _img = cv2.imread(args.image_file)
    img, im_info, im_scale = preprocess(_img)

    # Warm Up Inference
    #ret = detect_faces(img, 0.8, sess)
    log.info(f"** input_shape: {img.shape}")
    pred_onnx = sess.run(output_names, {input_name: img})
    faces, landmarks = postprocess(pred_onnx, im_info, im_scale, args.threshold)
    log.info(f"** output_shape - face: {[face.shape for face in faces]}")
    log.info(f"** output_shape - landmark: {[landmark.shape for landmark in landmarks]}")

    # Do inference test
    num = args.num
    log.info(f"** start inference time check {num} tries.")
    pbar = tqdm(range(num), desc='Face detection tests')
    t0 = datetime.now()
    for _ in pbar:
        img, im_info, im_scale = preprocess(_img)
        pred_onnx = sess.run(output_names, {input_name: img})
        faces, landmarks = postprocess(pred_onnx, im_info, im_scale, args.threshold)
    t1 = datetime.now() - t0
    log.info(f"** {num} times Inference Whole Time: {t1}, {num} times Inference Average Time: {t1 / num}")

    # Post process Image
    img = img[0].transpose(1, 2, 0)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    if faces is not None:
        log.info(f"** find {faces.shape[0]} faces")
        for i in range(faces.shape[0]):
            # print('score', faces[i][4])
            box = faces[i].astype(np.int32)
            # color = (255,0,0)
            color = (0, 0, 255.0)
            cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), color, 2)
            if landmarks is not None:
                landmark5 = landmarks[i].astype(np.int32)
                # print(landmark.shape)
                for l in range(landmark5.shape[0]):
                    color = (0, 0, 255)
                    if l == 0 or l == 3:
                        color = (0, 255, 0)
                    cv2.circle(img, (landmark5[l][0], landmark5[l][1]), 1, color, 2)

        filename = f'{os.path.splitext(args.input)[0]}-onnx.jpg'
        log.info(f"** writing {filename}")
        cv2.imwrite(filename, img)

    sys.exit(0)


def preprocess(img):
    img, im_scale, add_w, add_h = resize_image(img, [640, 640])

    scales = [640, 640]
    im_shape = img.shape
    target_size = scales[0]
    max_size = scales[1]
    im_size_min, im_size_max = np.min(im_shape[0:2]), np.max(im_shape[0:2])
    im_scale = float(target_size) / float(im_size_min)
    if np.round(im_scale * im_size_max) > max_size:
        im_scale = float(max_size) / float(im_size_max)

    pixel_means = [0.0, 0.0, 0.0]
    pixel_stds = [1.0, 1.0, 1.0]
    pixel_scale = 1.0
    if im_scale != 1.0:
        im = cv2.resize(img, None, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    else:
        im = img.copy()
    im = im.astype(np.float32)
    im_tensor = np.zeros((1, 3, im.shape[0], im.shape[1])).astype(np.float32)
    for i in range(3):
        im_tensor[0, i, :, :] = (im[:, :, 2 - i] / pixel_scale - pixel_means[2 - i]) / pixel_stds[2 - i]
    return im_tensor, im.shape[0:2], im_scale


def resize_image(img, sizes):
    img_h, img_w = img.shape[0:2]
    standard_scale = sizes[1] / sizes[0]

    if img_w / img_h > standard_scale:
        target_scale = sizes[1] / img_w
        target_w = sizes[1]
        target_h = int(img_h * target_scale)
        target_h = target_h if target_h % 2 == 0 else target_h + 1
        add_w = 0
        add_h = int((sizes[0] - target_h) / 2)
    else:   # img_w / img_h < standard_scale
        target_scale = sizes[0] / img_h
        target_h = sizes[0]
        target_w = int(img_w * target_scale)
        target_w = target_w if target_w % 2 == 0 else target_w + 1

        add_w = int((sizes[1] - target_w) / 2)
        add_h = 0

    im_scale = target_scale

    img = cv2.resize(
        img,
        dsize=(target_w, target_h),
        dst=None,
        fx=im_scale,
        fy=im_scale,
        interpolation=cv2.INTER_LINEAR
    )

    if add_w == 0:
        img = np.pad(img, ((add_h, add_h), (0, 0), (0, 0)), 'constant')
    else:   # add_h == 0
        img = np.pad(img, ((0, 0), (add_w, add_w), (0, 0)), 'constant')
    return img, im_scale, add_w, add_h


def postprocess(net_out, im_info, im_scale, threshold=0.9):
    threshold = threshold
    nms_threshold = 0.4
    decay4 = 0.5

    if ort.get_device() == 'GPU':
        nms = gpu_nms_wrapper(nms_threshold, 0)
    else:
        nms = cpu_nms_wrapper(nms_threshold)

    _feat_stride_fpn = [32, 16, 8]

    _anchors_fpn = {
        'stride32': np.array([[-248., -248., 263., 263.], [-120., -120., 135., 135.]], dtype=np.float32),
        'stride16': np.array([[-56., -56., 71., 71.], [-24., -24., 39., 39.]], dtype=np.float32),
        'stride8': np.array([[-8., -8., 23., 23.], [0., 0., 15., 15.]], dtype=np.float32)

    }

    _num_anchors = {'stride32': 2, 'stride16': 2, 'stride8': 2}

    proposals_list = []
    scores_list = []
    landmarks_list = []

    # net_out = [elt.numpy() for elt in net_out]
    sym_idx = 0

    for _idx, s in enumerate(_feat_stride_fpn):
        _key = 'stride%s' % s
        stride = int(s)

        scores = net_out[sym_idx]
        scores = scores[:,  _num_anchors['stride%s' % s]:, :, :]

        bbox_deltas = net_out[sym_idx + 1]
        height, width = bbox_deltas.shape[2], bbox_deltas.shape[3]

        A = _num_anchors['stride%s' % s]
        K = height * width
        anchors_fpn = _anchors_fpn['stride%s' % s]
        anchors = anchors_plane(height, width, stride,
                                anchors_fpn)
        anchors = anchors.reshape((K * A, 4))
        scores = scores.transpose((0, 2, 3, 1))
        scores = scores.reshape((-1, 1))

        bbox_deltas = bbox_deltas.transpose((0, 2, 3, 1))
        bbox_stds = [1.0, 1.0, 1.0, 1.0]
        bbox_deltas = bbox_deltas
        bbox_pred_len = bbox_deltas.shape[3] // A
        bbox_deltas = bbox_deltas.reshape((-1, bbox_pred_len))
        bbox_deltas[:, 0::4] = bbox_deltas[:, 0::4] * bbox_stds[0]
        bbox_deltas[:, 1::4] = bbox_deltas[:, 1::4] * bbox_stds[1]
        bbox_deltas[:, 2::4] = bbox_deltas[:, 2::4] * bbox_stds[2]
        bbox_deltas[:, 3::4] = bbox_deltas[:, 3::4] * bbox_stds[3]
        proposals = bbox_pred(anchors, bbox_deltas)

        proposals = clip_boxes(proposals, im_info[:2])

        if s == 4 and decay4 < 1.0:
            scores *= decay4

        scores_ravel = scores.ravel()
        order = np.where(scores_ravel >= threshold)[0]

        proposals = proposals[order, :]
        scores = scores[order]

        proposals[:, 0:4] /= im_scale
        proposals_list.append(proposals)
        scores_list.append(scores)

        landmark_deltas = net_out[sym_idx + 2]
        landmark_pred_len = landmark_deltas.shape[1] // A
        landmark_deltas = landmark_deltas.transpose((0, 2, 3, 1))
        landmark_deltas = landmark_deltas.reshape((-1, 5, landmark_pred_len // 5))
        landmarks = landmark_pred(anchors, landmark_deltas)
        landmarks = landmarks[order, :]

        landmarks[:, :, 0:2] /= im_scale
        landmarks_list.append(landmarks)
        sym_idx += 3

    proposals = np.vstack(proposals_list)
    if proposals.shape[0] == 0:
        landmarks = np.zeros((0, 5, 2))
        return np.zeros((0, 5)), landmarks
    scores = np.vstack(scores_list)
    scores_ravel = scores.ravel()
    order = scores_ravel.argsort()[::-1]

    proposals = proposals[order, :]
    scores = scores[order]
    landmarks = np.vstack(landmarks_list)
    landmarks = landmarks[order].astype(np.float32, copy=False)

    pre_det = np.hstack((proposals[:, 0:4], scores)).astype(np.float32, copy=False)

    # nms = cpu_nms_wrapper(nms_threshold)
    # keep = nms(pre_det)
    keep = nms(pre_det)

    det = np.hstack((pre_det, proposals[:, 4:]))
    det = det[keep, :]
    landmarks = landmarks[keep]

    return det, landmarks


#this function is copied from the following code snippet: https://github.com/StanislasBertrand/RetinaFace-tf2/blob/master/retinaface.py
def bbox_pred(boxes, box_deltas):
    if boxes.shape[0] == 0:
        return np.zeros((0, box_deltas.shape[1]))

    boxes = boxes.astype(np.float32, copy=False)
    widths = boxes[:, 2] - boxes[:, 0] + 1.0
    heights = boxes[:, 3] - boxes[:, 1] + 1.0
    ctr_x = boxes[:, 0] + 0.5 * (widths - 1.0)
    ctr_y = boxes[:, 1] + 0.5 * (heights - 1.0)

    dx = box_deltas[:, 0:1]
    dy = box_deltas[:, 1:2]
    dw = box_deltas[:, 2:3]
    dh = box_deltas[:, 3:4]

    pred_ctr_x = dx * widths[:, np.newaxis] + ctr_x[:, np.newaxis]
    pred_ctr_y = dy * heights[:, np.newaxis] + ctr_y[:, np.newaxis]
    pred_w = np.exp(dw) * widths[:, np.newaxis]
    pred_h = np.exp(dh) * heights[:, np.newaxis]

    pred_boxes = np.zeros(box_deltas.shape)
    # x1
    pred_boxes[:, 0:1] = pred_ctr_x - 0.5 * (pred_w - 1.0)
    # y1
    pred_boxes[:, 1:2] = pred_ctr_y - 0.5 * (pred_h - 1.0)
    # x2
    pred_boxes[:, 2:3] = pred_ctr_x + 0.5 * (pred_w - 1.0)
    # y2
    pred_boxes[:, 3:4] = pred_ctr_y + 0.5 * (pred_h - 1.0)

    if box_deltas.shape[1]>4:
        pred_boxes[:,4:] = box_deltas[:,4:]

    return pred_boxes

# This function copied from the following code snippet: https://github.com/StanislasBertrand/RetinaFace-tf2/blob/master/retinaface.py
def landmark_pred(boxes, landmark_deltas):
    if boxes.shape[0] == 0:
      return np.zeros((0, landmark_deltas.shape[1]))
    boxes = boxes.astype(np.float32, copy=False)
    widths = boxes[:, 2] - boxes[:, 0] + 1.0
    heights = boxes[:, 3] - boxes[:, 1] + 1.0
    ctr_x = boxes[:, 0] + 0.5 * (widths - 1.0)
    ctr_y = boxes[:, 1] + 0.5 * (heights - 1.0)
    pred = landmark_deltas.copy()
    for i in range(5):
        pred[:,i,0] = landmark_deltas[:,i,0]*widths + ctr_x
        pred[:,i,1] = landmark_deltas[:,i,1]*heights + ctr_y
    return pred

# This function copied from rcnn module of retinaface-tf2 project: https://github.com/StanislasBertrand/RetinaFace-tf2/blob/master/rcnn/processing/bbox_transform.py
def clip_boxes(boxes, im_shape):
    # x1 >= 0
    boxes[:, 0::4] = np.maximum(np.minimum(boxes[:, 0::4], im_shape[1] - 1), 0)
    # y1 >= 0
    boxes[:, 1::4] = np.maximum(np.minimum(boxes[:, 1::4], im_shape[0] - 1), 0)
    # x2 < im_shape[1]
    boxes[:, 2::4] = np.maximum(np.minimum(boxes[:, 2::4], im_shape[1] - 1), 0)
    # y2 < im_shape[0]
    boxes[:, 3::4] = np.maximum(np.minimum(boxes[:, 3::4], im_shape[0] - 1), 0)
    return boxes



#this function is mainly based on the following code snippet: https://github.com/StanislasBertrand/RetinaFace-tf2/blob/master/rcnn/cython/cpu_nms.pyx
#Fast R-CNN by Ross Girshick
def cpu_nms(dets, threshold):
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    ndets = dets.shape[0]
    suppressed = np.zeros((ndets), dtype=np.int)

    keep = []
    for _i in range(ndets):
        i = order[_i]
        if suppressed[i] == 1:
            continue
        keep.append(i)
        ix1 = x1[i]; iy1 = y1[i]; ix2 = x2[i]; iy2 = y2[i]
        iarea = areas[i]
        for _j in range(_i + 1, ndets):
            j = order[_j]
            if suppressed[j] == 1:
                continue
            xx1 = max(ix1, x1[j]); yy1 = max(iy1, y1[j]); xx2 = min(ix2, x2[j]); yy2 = min(iy2, y2[j])
            w = max(0.0, xx2 - xx1 + 1); h = max(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (iarea + areas[j] - inter)
            if ovr >= threshold:
                suppressed[j] = 1

    return keep

if __name__ == '__main__':
    main()
