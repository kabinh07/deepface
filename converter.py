import tensorflow as tf
from tensorflow.python.compiler.tensorrt import trt_convert as trt
from deepface.modules.modeling import build_model
import tf2onnx

# Load the Keras model
model = build_model(
    task= "facial_recognition", 
    model_name= "Facenet512"
).model

import tensorflow as tf
print("TensorFlow version:", tf.__version__)
print("GPU available:", tf.config.list_physical_devices('GPU'))

onnx_model_path = "data/facenet.onnx"
spec = (tf.TensorSpec((None, 160, 160, 3), tf.float32, name="input"),)
onnx_model, _ = tf2onnx.convert.from_keras(model, input_signature=spec, opset=13)

with open(onnx_model_path, "wb") as f:
    f.write(onnx_model.SerializeToString())

# model.model.save("data/saved_model")

# # Convert the model to TensorRT
# converter = trt.TrtGraphConverterV2(input_saved_model_dir="data/saved_model")
# trt_model = converter.convert()

# # Save the optimized model
# converter.save("data/trt_saved_model")
# print("TensorRT model saved successfully!")