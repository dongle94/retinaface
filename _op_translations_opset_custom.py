#  Copyright (c) 2017, NVIDIA CORPORATION. All rights reserved.
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions
#  are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
#  EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
#  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
#  PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
#  CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
#  EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
#  PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
#  PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
#  OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
#  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# Based on
#  https://github.com/NVIDIA/mxnet_to_onnx/blob/master/mx2onnx_converter/
# mx2onnx_converter_functions.py

# coding: utf-8
# pylint: disable=too-many-locals,no-else-return,too-many-lines
# pylint: disable=anomalous-backslash-in-string,eval-used
# pylint: disable=too-many-function-args
"""
Conversion Functions for common layers.
Add new functions here with a decorator.
"""

from .._export_onnx import MXNetGraph as mx_op
try:
    import onnx
except ImportError:
    onnx = None


def get_inputs(node, kwargs):
    """Helper function to get inputs"""
    name = node["name"]
    outputs_lookup = kwargs["outputs_lookup"]
    inputs = node["inputs"]
    attrs = node.get("attrs", {})
    input_nodes = []
    for ip in inputs:
        input_node_name = outputs_lookup[ip[0]][ip[1]].name
        input_nodes.append(input_node_name)

    return name, input_nodes, attrs



@mx_op.register("SoftmaxActivation")
def convert_SoftmaxActivation(node, **kwargs):
    """Map MXNet's SoftmaxActivation operator attributes to onnx's Softmax operator
    and return the created node.
    """
    print(node)
    name, input_nodes, attrs = get_inputs(node, kwargs)

    axis = int(attrs.get("axis", 1))

    softmax_node = onnx.helper.make_node(
        "Softmax",
        input_nodes,
        [name],
        axis=axis,
        name=name
    )

    return [softmax_node]