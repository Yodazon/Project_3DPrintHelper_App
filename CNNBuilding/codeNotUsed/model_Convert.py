#To convert .onnx files to tensorflow

import onnx

from onnx_tf.backend import prepare

onnx_model = onnx.load("CNN_Model.onnx")  # load onnx model
tf_rep = prepare(onnx_model)  # prepare tf representation
#tf_rep.export_graph("CNN_Model_APp.js")  # export the model

# Input nodes to the model
print('inputs:', tf_rep.inputs)

# Output nodes from the model
print('outputs:', tf_rep.outputs)

# All nodes in the model
print('tensor_dict:')
print(tf_rep.tensor_dict)