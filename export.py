import argparse
import numpy as np
import mxnet as mx
import os

class RetinaFaceCalibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self):
        pass
# class MNISTEntropyCalibrator(trt.IInt8EntropyCalibrator2):
#     def __init__(self, training_data, cache_file, batch_size=64):
#         # Whenever you specify a custom constructor for a TensorRT class,
#         # you MUST call the constructor of the parent explicitly.
#         trt.IInt8EntropyCalibrator2.__init__(self)
#
#         self.cache_file = cache_file
#
#         # Every time get_batch is called, the next batch of size batch_size will be copied to the device and returned.
#         self.data = load_mnist_data(training_data)
#         self.batch_size = batch_size
#         self.current_index = 0
#
#         # Allocate enough memory for a whole batch.
#         self.device_input = cuda.mem_alloc(self.data[0].nbytes * self.batch_size)
#
#     def get_batch_size(self):
#         return self.batch_size
#
#     # TensorRT passes along the names of the engine bindings to the get_batch function.
#     # You don't necessarily have to use them, but they can be useful to understand the order of
#     # the inputs. The bindings list is expected to have the same ordering as 'names'.
#     def get_batch(self, names):
#         if self.current_index + self.batch_size > self.data.shape[0]:
#             return None
#
#         current_batch = int(self.current_index / self.batch_size)
#         if current_batch % 10 == 0:
#             print("Calibrating batch {:}, containing {:} images".format(current_batch, self.batch_size))
#
#         batch = self.data[self.current_index: self.current_index + self.batch_size].ravel()
#         cuda.memcpy_htod(self.device_input, batch)
#         self.current_index += self.batch_size
#         return [self.device_input]
#
#     def read_calibration_cache(self):
#         # If there is a cache, use it instead of calibrating again. Otherwise, implicitly return None.
#         if os.path.exists(self.cache_file):
#             with open(self.cache_file, "rb") as f:
#                 return f.read()
#
#     def write_calibration_cache(self, cache):
#         with open(self.cache_file, "wb") as f:
#             f.write(cache)

def convert(opt):
    input_shape = [(1, 3, 640, 640)]
    input_type = [np.float32]
    onnx_file = f"./{opt.params.split('-')[0]}.onnx"

    if os.path.exists(onnx_file):
        converted_onnx = onnx_file
        print(f"** ONNX file already exist.")
    else:
        converted_onnx = mx.onnx.export_model(opt.symbol, opt.params, input_shape, input_type, onnx_file)
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
    parser.add_argument('--fp16', action='store_true', help='float point 16bits. half precision')
    parser.add_argument('--int8', action='store_true', help='int8 Quantaization')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    print(f"** Parameters: {args}")
    convert(args)