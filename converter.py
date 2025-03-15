import tensorflow as tf
from tensorflow.python.compiler.tensorrt import trt_convert as trt
from deepface.modules.modeling import build_model
import tf2onnx
import torch
from onnx2torch import convert
import onnx
from onnxsim import simplify

# Load the Keras model
model = build_model(
    task="facial_recognition", 
    model_name="Facenet512"
).model

# Convert to ONNX
onnx_model_path = "data/facenet512.onnx"
spec = (tf.TensorSpec((1, 160, 160, 3), tf.float32, name="input"),)  # Adjust input shape if necessary
onnx_model, _ = tf2onnx.convert.from_keras(model, input_signature=spec, opset=13)

# Save ONNX model
with open(onnx_model_path, "wb") as f:
    f.write(onnx_model.SerializeToString())

print(f"Model converted and saved as {onnx_model_path}")

# Load ONNX model
onnx_model = onnx.load("data/facenet512.onnx")

# # Simplify model
# model_simp, check = simplify(onnx_model)

# if check:
#     print("Successfully simplified ONNX model")
#     onnx.save(model_simp, "data/facenet512_simplified.onnx")
# else:
#     print("Simplification failed")

# # Load ONNX model
# onnx_model = onnx.load("data/facenet512_simplified.onnx")

# Convert to PyTorch model
torch_model = convert(onnx_model)

# Save PyTorch model
torch.save(torch_model, "data/facenet512_pytorch.pth")

print("ONNX model successfully converted to PyTorch")

# Load the torch model
torch_model = torch.load("data/facenet512_pytorch.pth")

# Dummy inp1ut with the expected shape (adjust as needed)
dummy_input = torch.randn(1, 160, 160, 3)

# Convert PyTorch model to JIT
jit_model = torch.jit.trace(torch_model, dummy_input)

# Save JIT model
jit_model.save("data/facenet512_jit_traced.pt")

print("TorchScript (JIT) traced model saved as facenet512_jit_traced.pt")