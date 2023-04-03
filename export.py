import argparse
import numpy as np
import mxnet as mx
import os

import onnx
import tensorrt as trt
#import pycuda.driver as cuda
#import pycuda.autoinit
import cupy as cp
import cv2

class RetinaFaceCalibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, training_data, cache_file="", batch_size=32):
        trt.IInt8EntropyCalibrator2.__init__(self)

        self.dataset = mx.gluon.data.vision.datasets.ImageFolderDataset(training_data)
        self.cache_file = cache_file
        self.batch_size = batch_size
        self.current_index = 0

        # shape = self.engine.get_binding_shape(binding)
        # dtype = trt.nptype(self.engine.get_binding_dtype(binding))
        # cuda_mem = cp.zeros(shape=shape, dtype=dtype)
        self.data = cp.zeros(shape=(self.batch_size, 3, 640, 640), dtype=cp.int8)
        #self.data = np.zeros((self.batch_size, 3, 640, 640))
        # self.device_input = cuda.mem_alloc(self.data[0].nbytes)
        self.device_input = self.data.data.ptr
        #print(self.dataset[0])
        # exit()
        # for k, (images, targets) in enumerate(self.dataset):
        #     if k >= self.max_batches:
        #         break
        #     self.data[k] = images.numpy()

    def get_batch_size(self):
        return self.batch_size

    def get_batch(self, names):
        """Get a batch of input for calibration.
            Args:
                names (List[str]): list of file names
            Returns:
                list of device memory pointers set to the memory containing
                each network input data, or an empty list if there are no more
                batches for calibration
        """
        print('** Calibration starting...')
        try:
            if self.current_index + self.batch_size > self.data.shape[0]:
                return None
            current_batch = int(self.current_index / self.batch_size)
            # if current_batch % self.batch_size == 0:
            print("** Calibrating batch {:}, containing {:} images".format(current_batch, self.batch_size))

            # Assume self.batches is a generator that provides batch data.

            batch = self.dataset.take(self.batch_size)
            for i in range(self.batch_size):
                img = cv2.resize(batch[i][0].asnumpy(), (640, 640), interpolation=cv2.INTER_LINEAR)
                img = cp.asarray(img)
                img = cp.transpose(img, [2, 0, 1])
                img = img.astype(cp.int8)
                self.data[i] = img
            # Assume that self.device_input is a device buffer allocated by the constructor.
            #cuda.memcpy_htod(self.device_input, batch)
            self.current_index += self.batch_size
            return [int(self.device_input)]
        except StopIteration:
            # When we're out of batches, we return either [] or None.
            # This signals to TensorRT that there is no calibration data remaining.
            return None

    def read_calibration_cache(self):
        """Load a calibration cache. If there is a cache, use it instead of calibrating again. Otherwise, implicitly return None.
            Returns:
                a cache object or None if there is no data
        """
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                return f.read()
        return None

    def write_calibration_cache(self, cache):
        """Save a calibration cache.
           Args:
               cache (memoryview): the calibration cache to write
        """
        print('[Write calibration cache file]')
        with open(self.cache_file, "wb") as f:
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
            print("Model with ir_version below 4 requires to include initilizer in graph input")
        else:
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
                calibration_cache = "retinaface_calibration.cache"
                calib = RetinaFaceCalibrator(os.path.abspath(opt.calib_data), cache_file=calibration_cache)
                config.set_flag(trt.BuilderFlag.INT8)
                config.int8_calibrator = calib

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
    parser.add_argument('--calib_data',
                        help='tensorrt engine int8 calibration data location',
                        type=str)
    parser.add_argument('--fp16', action='store_true', help='float point 16bits. half precision')
    parser.add_argument('--int8', action='store_true', help='int8 Quantaization')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    print(f"** Parameters: {args}")
    convert(args)