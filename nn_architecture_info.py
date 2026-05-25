import onnx

model = onnx.load("models/onnx_whisper/1/model.onnx")


for inp in model.graph.input:
	print(inp.name)

for inp in model.graph.output:
	print(inp.name)
