import onnx
from onnxsim import simplify

# load your predefined ONNX model
model = onnx.load('model/just_reshape.onnx')
#model = onnx.load('onnx/output_file.onnx')
input_shapes = {}

input_shape = ['input:1,3,4,5']

if input_shape is not None:
        for x in input_shape:
            if ':' not in x:
                input_shapes[None] = list(map(int, x.split(',')))
            else:
                pieces = x.split(':')
                # for the input name like input:0
                name, shape = ':'.join(
                    pieces[:-1]), list(map(int, pieces[-1].split(',')))
                input_shapes[name] = shape

# convert model
#model_simp, check = simplify(model)


## 本版本不支持导出dynamic_input_shape的简化模型 ，pip安装的最新版支持此功能
model_simp, check = simplify(model, check_n= 0, perform_optimization = True,skip_fuse_bn= False, input_shapes = input_shapes, skipped_optimizers = None, skip_shape_inference=False) 

assert check, "Simplified ONNX model could not be validated"

onnx.save(model_simp, 'model/simplifier2.onnx')

# use model_simp as a standard ONNX model object