import os
import sys
import cv2
import time
import ctypes
import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda

import uff
import tensorrt as trt
import graphsurgeon as gs


# initialize
TRT_LOGGER = trt.Logger(trt.Logger.INFO)
trt.init_libnvinfer_plugins(TRT_LOGGER, '')
runtime = trt.Runtime(TRT_LOGGER)

# compile model into TensorRT
uff_model = uff.from_tensorflow_frozen_model('frozen_model.pb', ['dense_1/BiasAdd'], output_filename='tmp.uff')

with trt.Builder(TRT_LOGGER) as builder, builder.create_network() as network, trt.UffParser() as parser:
    builder.max_workspace_size = 1 << 28
    builder.max_batch_size = 1

    parser.register_input('conv2d_input', (1, 100, 400))
    parser.register_output('dense_1/BiasAdd')
    parser.parse('tmp.uff', network)
    engine = builder.build_cuda_engine(network)

    buf = engine.serialize()

    with open('model.bin', 'wb') as f:
        f.write(buf)
