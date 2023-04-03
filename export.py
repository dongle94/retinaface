import argparse
import logging
import os
import sys
import numpy as np
import mxnet as mx
import cv2

import onnx
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import cupy as cp

from PIL import Image
import random


logging.basicConfig(level=logging.INFO)
logging.getLogger("Exporter").setLevel(logging.INFO)
log = logging.getLogger("Exporter")


class ImageBatcher:
    """
    Creates batches of pre-processed images.
    """

    def __init__(self, input, shape, dtype, max_num_images=None, exact_batches=False, preprocessor="Retinaface", shuffle_files=False):
        """
        :param input: The input directory to read images from.
        :param shape: The tensor shape of the batch to prepare, either in NCHW or NHWC format.
        :param dtype: The (numpy) datatype to cast the batched data to.
        :param max_num_images: The maximum number of images to read from the directory.
        :param exact_batches: This defines how to handle a number of images that is not an exact multiple of the batch
        size. If false, it will pad the final batch with zeros to reach the batch size. If true, it will *remove* the
        last few images in excess of a batch size multiple, to guarantee batches are exact (useful for calibration).
        :param preprocessor: Set the preprocessor to use, depending on which network is being used.
        :param shuffle_files: Shuffle the list of files before batching.
        """
        # Find images in the given input path
        input = os.path.realpath(input)
        self.images = []

        extensions = [".jpg", ".jpeg", ".png", ".bmp"]

        def is_image(path):
            return os.path.isfile(path) and os.path.splitext(path)[1].lower() in extensions

        if os.path.isdir(input):
            self.images = [os.path.join(input, f) for f in os.listdir(input) if is_image(os.path.join(input, f))]
            self.images.sort()
            if shuffle_files:
                random.seed(47)
                random.shuffle(self.images)
        elif os.path.isfile(input):
            if is_image(input):
                self.images.append(input)
        self.num_images = len(self.images)
        if self.num_images < 1:
            print("No valid {} images found in {}".format("/".join(extensions), input))
            sys.exit(1)

        # Handle Tensor Shape
        self.dtype = dtype
        self.shape = shape
        assert len(self.shape) == 4
        self.batch_size = shape[0]
        assert self.batch_size > 0
        self.format = None
        self.width = -1
        self.height = -1
        if self.shape[1] == 3:
            self.format = "NCHW"
            self.height = self.shape[2]
            self.width = self.shape[3]
        elif self.shape[3] == 3:
            self.format = "NHWC"
            self.height = self.shape[1]
            self.width = self.shape[2]
        assert all([self.format, self.width > 0, self.height > 0])

        # Adapt the number of images as needed
        if max_num_images and 0 < max_num_images < len(self.images):
            self.num_images = max_num_images
        if exact_batches:
            self.num_images = self.batch_size * (self.num_images // self.batch_size)
        if self.num_images < 1:
            print("Not enough images to create batches")
            sys.exit(1)
        self.images = self.images[0:self.num_images]

        # Subdivide the list of images into batches
        self.num_batches = 1 + int((self.num_images - 1) / self.batch_size)
        self.batches = []
        for i in range(self.num_batches):
            start = i * self.batch_size
            end = min(start + self.batch_size, self.num_images)
            self.batches.append(self.images[start:end])

        # Indices
        self.image_index = 0
        self.batch_index = 0

        self.preprocessor = preprocessor

    def preprocess_image(self, image_path):
        """
        The image preprocessor loads an image from disk and prepares it as needed for batching. This includes padding,
        resizing, normalization, data type casting, and transposing.
        This Image Batcher implements one algorithm for now:
        * EfficientDet: Resizes and pads the image to fit the input size.
        :param image_path: The path to the image on disk to load.
        :return: Two values: A numpy array holding the image sample, ready to be contacatenated into the rest of the
        batch, and the resize scale used, if any.
        """

        def resize_pad(image, pad_color=(0, 0, 0)):
            """
            A subroutine to implement padding and resizing. This will resize the image to fit fully within the input
            size, and pads the remaining bottom-right portions with the value provided.
            :param image: The PIL image object
            :pad_color: The RGB values to use for the padded area. Default: Black/Zeros.
            :return: Two values: The PIL image object already padded and cropped, and the resize scale used.
            """
            width, height = image.size
            width_scale = width / self.width
            height_scale = height / self.height
            scale = 1.0 / max(width_scale, height_scale)
            image = image.resize((round(width * scale), round(height * scale)), resample=Image.BILINEAR)
            pad = Image.new("RGB", (self.width, self.height))
            pad.paste(pad_color, [0, 0, self.width, self.height])
            pad.paste(image)
            return pad, scale

        scale = None
        image = Image.open(image_path)
        image = image.convert(mode="RGB")
        if self.preprocessor == "Retinaface":
            # For EfficientNet V2: Resize & Pad with ImageNet mean values and keep as [0,255] Normalization
            image, scale = resize_pad(image, (124, 116, 104))
            image = np.asarray(image, dtype=self.dtype)
            # [0-1] Normalization, Mean subtraction and Std Dev scaling are part of the EfficientDet graph, so
            # no need to do it during preprocessing here
        else:
            print("Preprocessing method {} not supported".format(self.preprocessor))
            sys.exit(1)
        if self.format == "NCHW":
            image = np.transpose(image, (2, 0, 1))
        return image, scale

    def get_batch(self):
        """
        Retrieve the batches. This is a generator object, so you can use it within a loop as:
        for batch, images in batcher.get_batch():
           ...
        Or outside of a batch with the next() function.
        :return: A generator yielding three items per iteration: a numpy array holding a batch of images, the list of
        paths to the images loaded within this batch, and the list of resize scales for each image in the batch.
        """
        for i, batch_images in enumerate(self.batches):
            batch_data = np.zeros(self.shape, dtype=self.dtype)
            batch_scales = [None] * len(batch_images)
            for i, image in enumerate(batch_images):
                self.image_index += 1
                batch_data[i], batch_scales[i] = self.preprocess_image(image)
            self.batch_index += 1
            yield batch_data, batch_images, batch_scales


class EngineCalibrator(trt.IInt8EntropyCalibrator2):
    """
    Implements the INT8 Entropy Calibrator 2.
    """
    def __init__(self, cache_file):
        """
        :param cache_file: The location of the cache file.
        """
        super().__init__()
        self.cache_file = cache_file
        self.image_batcher = None
        self.batch_allocation = None
        self.batch_generator = None

    def set_image_batcher(self, image_batcher: ImageBatcher):
        """
        Define the image batcher to use, if any. If using only the cache file, an image batcher doesn't need
        to be defined.
        :param image_batcher: The ImageBatcher object
        """
        self.image_batcher = image_batcher
        size = int(np.dtype(self.image_batcher.dtype).itemsize * np.prod(self.image_batcher.shape))
        self.batch_allocation = cuda.mem_alloc(size)
        self.batch_generator = self.image_batcher.get_batch()
    def get_batch_size(self):
        """
        Overrides from trt.IInt8EntropyCalibrator2.
        Get the batch size to use for calibration.
        :return: Batch size.
        """
        if self.image_batcher:
            return self.image_batcher.batch_size
        return 1

    def get_batch(self, names):
        """
        Overrides from trt.IInt8EntropyCalibrator2.
        Get the next batch to use for calibration, as a list of device memory pointers.
        :param names: The names of the inputs, if useful to define the order of inputs.
        :return: A list of int-casted memory pointers.
        """
        if not self.image_batcher:
            return None
        try:
            batch, _, _ = next(self.batch_generator)
            log.info("Calibrating image {} / {}".format(self.image_batcher.image_index, self.image_batcher.num_images))
            cuda.memcpy_htod(self.batch_allocation, np.ascontiguousarray(batch))
            return [int(self.batch_allocation)]
        except StopIteration:
            log.info("Finished calibration batches")
            return None

    def read_calibration_cache(self):
        """
        Overrides from trt.IInt8EntropyCalibrator2.
        Read the calibration cache file stored on disk, if it exists.
        :return: The contents of the cache file, if any.
        """
        if self.cache_file is not None and os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                log.info("Using calibration cache file: {}".format(self.cache_file))
                return f.read()

    def write_calibration_cache(self, cache):
        """
        Overrides from trt.IInt8EntropyCalibrator2.
        Store the calibration cache to a file on disk.
        :param cache: The contents of the calibration cache to store.
        """
        if self.cache_file is None:
            return
        with open(self.cache_file, "wb") as f:
            log.info("** Writing calibration cache data to: {}".format(self.cache_file))
            f.write(cache)


def convert(opt):
    input_shape = [(1, 3, 640, 640)]
    input_type = [np.float32]
    onnx_file = f"{opt.params.split('-')[0]}.onnx"

    if os.path.exists(onnx_file):
        converted_onnx = onnx_file
        print(f"** ONNX file already exist.")
    else:
        print(f"** Convert {onnx_file} starting")
        converted_onnx = mx.onnx.export_model(opt.symbol, opt.params, input_shape, input_type, onnx_file)

        onnx_model = onnx.load(converted_onnx)
        if onnx_model.ir_version < 4:
            print("** Model with ir_version below 4 requires to include initilizer in graph input")
        else:
            print("** Remove model initializer from inputs")
            inputs = onnx_model.graph.input
            name_to_input = {}
            for input in inputs:
                name_to_input[input.name] = input

            for initializer in onnx_model.graph.initializer:
                if initializer.name in name_to_input:
                    inputs.remove(name_to_input[initializer.name])

            onnx.save(onnx_model, converted_onnx)

    if opt.type == 'onnx':
        print(f"** Convert {converted_onnx} Succecss.")
        return
    elif opt.type == 'tensorrt':
        import tensorrt as trt
        trt_engine_path = f"{opt.params.split('-')[0]}.engine"

        TRT_LOGGER = trt.Logger(trt.Logger.INFO)
        trt.init_libnvinfer_plugins(TRT_LOGGER, 'RetinaFace')

        EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        with trt.Builder(TRT_LOGGER) as builder, builder.create_network(EXPLICIT_BATCH) as network, \
                builder.create_builder_config() as config, trt.OnnxParser(network, TRT_LOGGER) as parser:
            # builder.max_batch_size = batch_size       # DEPRECATION

            with open(converted_onnx, 'rb') as model:
                parser.parse(model.read())
            print(f"** num_input:{network.num_inputs} / num_output:{network.num_outputs}")
            for i in range(network.num_inputs):
                print(
                    f"** input {i}: {network.get_input(0).name} / {network.get_input(0).shape} / {network.get_input(0).dtype}")
            for i in range(network.num_outputs):
                print(
                    f"** output {i}: {network.get_output(i).name} / {network.get_output(i).shape} / {network.get_output(i).dtype}")

            profile = builder.create_optimization_profile()
            opt_shape = network.get_input(0).shape
            if len(opt_shape) == 4 and opt.batch_size != 1:
                opt_shape = (opt.batch_size, opt_shape[1], opt_shape[2], opt_shape[3])
            profile.set_shape(network.get_input(0).name, opt_shape, opt_shape, opt_shape)
            print(f"** Optimization profile: {network.get_input(0).name} / {opt_shape} / {opt_shape} / {opt_shape}")
            print("** num_layers: ", network.num_layers)

            config.add_optimization_profile(profile)
            config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << opt.workspace_size)
            config.set_tactic_sources(1 << int(trt.TacticSource.CUBLAS) | 1 << int(trt.TacticSource.CUBLAS_LT) | 1 << int(trt.TacticSource.CUDNN))
            if opt.fp16:
                config.set_flag(trt.BuilderFlag.FP16)
            elif opt.int8:
                config.set_flag(trt.BuilderFlag.INT8)

                inputs = [network.get_input(i) for i in range(network.num_inputs)]

                calib_cache = opt.calib_cache
                #calib = RetinaFaceCalibrator(os.path.abspath(opt.calib_input), cache_file=calibration_cache)
                calib = EngineCalibrator(cache_file=calib_cache)
                config.int8_calibrator = calib
                if calib_cache is None or not os.path.exists(calib_cache):
                    calib_shape = [opt.calib_batch_size] + list(inputs[0].shape[1:])
                    calib_dtype = trt.nptype(inputs[0].dtype)
                    print(calib_shape, calib_dtype)
                    config.int8_calibrator.set_image_batcher(
                        ImageBatcher(input=opt.calib_input,
                                     shape=calib_shape,
                                     dtype=calib_dtype,
                                     max_num_images=opt.calib_num_images,
                                     exact_batches=True,
                                     shuffle_files=True)
                    )

            plan = builder.build_serialized_network(network, config)
            with open(trt_engine_path, "wb") as f:
                f.write(plan)


def parse_args():
    parser = argparse.ArgumentParser(description='Export RetinaFace')
    parser.add_argument('--symbol',
                        help='network checkpoint symbol file',
                        required=True,
                        type=str)
    parser.add_argument('--params',
                        help='network checkpoint params file',
                        required=True,
                        type=str)
    parser.add_argument('--type',
                        help='export optimization framework',
                        choices=['onnx', 'tensorrt'])
    parser.add_argument('-b', '--batch_size',
                        help='tensorrt engine batch size',
                        type=int,
                        default=1)
    parser.add_argument('-w', '--workspace_size',
                        help='tensorrt engine workspace size',
                        type=int,
                        default=30)
    parser.add_argument('--calib_input',
                        help='tensorrt engine int8 calibration data location',
                        type=str)
    parser.add_argument("--calib_cache",
                        default=None,
                        help="The file path for INT8 calibration cache to use, default: ./calibration.cache")
    parser.add_argument("--calib_num_images",
                        default=5000,
                        type=int,
                        help="The maximum number of images to use for calibration, default: 5000")
    parser.add_argument("--calib_batch_size",
                        default=16,
                        type=int,
                        help="The batch size for the calibration process, default: 16")
    parser.add_argument('--fp16', action='store_true', help='float point 16bits. half precision')
    parser.add_argument('--int8', action='store_true', help='int8 Quantaization')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    log.info(f"** Parameters: {args}")
    convert(args)