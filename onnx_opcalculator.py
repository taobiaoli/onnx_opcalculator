import string
import onnx
import onnxruntime as rt
import numpy as np
from onnx import numpy_helper
import csv
import argparse

def calculate_params(model: onnx.ModelProto) -> int:
    onnx_weights = model.graph.initializer
    params = 0

    for onnx_w in onnx_weights:
        try:
            weight = numpy_helper.to_array(onnx_w)
            params += np.prod(weight.shape)
        except Exception as _:
            pass

    return params


def onnx_node_attributes_to_dict(args):
    """
    Parse ONNX attributes to Python dictionary
    :param args: ONNX attributes object
    :return: Python dictionary
    """
    def onnx_attribute_to_dict(onnx_attr):
        """
        Parse ONNX attribute
        :param onnx_attr: ONNX attribute
        :return: Python data type
        """
        if onnx_attr.HasField('t'):
            return numpy_helper.to_array(getattr(onnx_attr, 't'))

        for attr_type in ['f', 'i', 's']:
            if onnx_attr.HasField(attr_type):
                return getattr(onnx_attr, attr_type)

        for attr_type in ['floats', 'ints', 'strings']:
            if getattr(onnx_attr, attr_type):
                return list(getattr(onnx_attr, attr_type))
    return {arg.name: onnx_attribute_to_dict(arg) for arg in args}


def calculate_macs(model: onnx.ModelProto,csv_file_name):
    # shape inference for node
    infered_model = onnx.shape_inference.infer_shapes(model)
    onnx_nodes = infered_model.graph.node
    #onnx.save(infered_model, 'infer_shape.onnx')
    node_name_dict = {}
    input_shape_list = []
    output_shape_list = []
    # get input node shape
    for index,input_shape in enumerate(infered_model.graph.input[0].type.tensor_type.shape.dim):
        input_shape_list.append(input_shape.dim_value)
    node_name_dict.update({infered_model.graph.input[0].name:np.array(input_shape_list)})
    # get output node shape
    for index,output_shape in enumerate(infered_model.graph.output[0].type.tensor_type.shape.dim):
        output_shape_list.append(output_shape.dim_value)
    node_name_dict.update({infered_model.graph.output[0].name:np.array(output_shape_list)})
    
    # get shape list for graph node 
    node_shape_list = infered_model.graph.value_info
    for node_obj in node_shape_list:
        dim = []
        for i in range(len(node_obj.type.tensor_type.shape.dim)):
            dim.append(node_obj.type.tensor_type.shape.dim[i].dim_value)

        node_tmp= {node_obj.name:np.array(dim)}
        #print(node_tmp)
        node_name_dict.update(node_tmp)

    def conv_macs(node, input_shape, output_shape, attrs):
        kernel_ops = np.prod(attrs['kernel_shape'])  # Kw x Kh
        bias_ops = len(node.input) == 3
        
        group = 1
        if 'group' in attrs:
            group = attrs['group']

        in_channels = input_shape[1]
        # save parameter to csv
        # input shape
        Input_feature_n = input_shape[0]
        Input_feature_c = input_shape[1]
        Input_feature_h = input_shape[2]
        Input_feature_w = input_shape[3]

        # kernel shape
        Kernel_h = attrs['kernel_shape'][0]
        Kernel_w = attrs['kernel_shape'][1]

        # output shape
        Output_feature_n = output_shape[0]
        Output_feature_c = output_shape[1]
        Output_feature_h = output_shape[2]
        Output_feature_w = output_shape[3]

        # calculate flops
        Conv_flops = np.prod(output_shape,dtype=np.int64) * (in_channels // group * kernel_ops + bias_ops)
        # bias
        '''
        if bias_ops:
            Conv_flop_A = np.prod(output_shape,dtype=np.int64)
        else:
            Conv_flop_A = 0
        '''
        # memory size
        Input_size = np.prod(input_shape)
        Kernel_size = kernel_ops * Input_feature_c * Output_feature_c  + Output_feature_c * bias_ops
        Output_size = np.prod(output_shape)
        Total_memory = Input_size + Kernel_size + Output_size
        return [Input_feature_c,Input_feature_h,Input_feature_w,Output_feature_c,Kernel_h,Kernel_w,group,Output_feature_h,Output_feature_w,Conv_flops,Input_size,Kernel_size,Output_size,Total_memory]

    def gemm_macs(node, input_shape, output_shape, attrs):
        # save parameter to csv
        # input shape
        Input_feature_n = input_shape[0]
        Input_feature_c = input_shape[1]
        Input_feature_h = input_shape[2]
        Input_feature_w = input_shape[3]

        # kernel shape
        Kernel_h = attrs['kernel_shape'][0]
        Kernel_w = attrs['kernel_shape'][1]

        # output shape
        Output_feature_n = output_shape[0]
        Output_feature_c = output_shape[1]
        Output_feature_h = output_shape[2]
        Output_feature_w = output_shape[3]

        Gemm_flops = np.prod(input_shape) * np.prod(output_shape)
        Input_size = np.prod(input_shape)
        Kernel_size = Kernel_h * Kernel_w
        Output_size = np.prod(output_shape)
        Total_memory = Input_size + Kernel_size + Output_size

        return [Input_feature_c,Input_feature_h,Input_feature_w,Output_feature_c,Kernel_h,Kernel_w,Input_feature_n,Output_feature_h,Output_feature_w,Gemm_flops,Input_size,Kernel_size,Output_size,Total_memory]
        #return np.prod(input_shape) * np.prod(output_shape)

    def bn_macs(node, input_shape, output_shape, attrs):
        batch_macs = np.prod(output_shape)
        if len(node.input) == 5:
            batch_macs *= 2
        
        # input shape
        Input_feature_n = input_shape[0]
        Input_feature_c = input_shape[1]
        Input_feature_h = input_shape[2]
        Input_feature_w = input_shape[3]

        # output shape
        Output_feature_n = output_shape[0]
        Output_feature_c = output_shape[1]
        Output_feature_h = output_shape[2]
        Output_feature_w = output_shape[3]

        Bn_flops = np.prod(input_shape) * 2 # BN parameter:s * (x - mean) / np.sqrt(var + epsilon) + bias 因此两次乘加 np.sqrt(var + epsilon)当做常数?
        Input_size = np.prod(input_shape)
        Kernel_size =  Input_feature_c * 4
        Output_size = np.prod(output_shape)
        Total_memory = Input_size + Kernel_size + Output_size
        return [Input_feature_c,Input_feature_h,Input_feature_w,Output_feature_c,'','',Input_feature_n,Output_feature_h,Output_feature_w,Bn_flops,Input_size,Kernel_size,Output_size,Total_memory]

    def upsample_macs(node, input_shape, output_shape, attrs):
        Upsample_flops = 0
        if 'mode' in attrs:
            if attrs['mode'].decode('utf-8') == 'nearest':
                Upsample_flops = 0
            if attrs['mode'].decode('utf-8') == 'linear':
                Upsample_flops = np.prod(output_shape) * 11

        # input shape
        Input_feature_n = input_shape[0]
        Input_feature_c = input_shape[1]
        Input_feature_h = input_shape[2]
        Input_feature_w = input_shape[3]
        # kernel shape
        Kernel_h = attrs['kernel_shape'][0]
        Kernel_w = attrs['kernel_shape'][1]
        # output shape
        Output_feature_n = output_shape[0]
        Output_feature_c = output_shape[1]
        Output_feature_h = output_shape[2]
        Output_feature_w = output_shape[3]

        Input_size = np.prod(input_shape)
        Kernel_size = np.prod(attrs['kernel_shape'])
        Output_size = np.prod(output_shape)
        Total_memory = Input_size + Kernel_size + Output_size
        return [Input_feature_c,Input_feature_h,Input_feature_w,Output_feature_c,Kernel_h,Kernel_w,Input_feature_n,Output_feature_h,Output_feature_w,Upsample_flops,Input_size,Kernel_size,Output_size,Total_memory]

    def relu_macs(node, input_shape, output_shape, attrs):
        # input shape
        Input_feature_n = input_shape[0]
        Input_feature_c = input_shape[1]
        Input_feature_h = input_shape[2]
        Input_feature_w = input_shape[3]
    
        # output shape
        Output_feature_n = output_shape[0]
        Output_feature_c = output_shape[1]
        Output_feature_h = output_shape[2]
        Output_feature_w = output_shape[3]
        
        Relu_flops = np.prod(input_shape)
        Input_size = np.prod(input_shape)
        Kernel_size = 0
        Output_size = np.prod(output_shape)
        Total_memory = Input_size + Kernel_size + Output_size
        return [Input_feature_c,Input_feature_h,Input_feature_w,Output_feature_c,'','',Input_feature_n,Output_feature_h,Output_feature_w,Relu_flops,Input_size,Kernel_size,Output_size,Total_memory]
    
    def add_macs(node, input_shape, output_shape, attrs):
        # input shape
        Input_feature_n = input_shape[0]
        Input_feature_c = input_shape[1]
        Input_feature_h = input_shape[2]
        Input_feature_w = input_shape[3]
    
        # output shape
        Output_feature_n = output_shape[0]
        Output_feature_c = output_shape[1]
        Output_feature_h = output_shape[2]
        Output_feature_w = output_shape[3]
        
        add_flops = np.prod(input_shape)
        Input_size = np.prod(input_shape) * 2 # the same input shape
        Kernel_size = 0
        Output_size = np.prod(output_shape)
        Total_memory = Input_size + Kernel_size + Output_size
        return [Input_feature_c,Input_feature_h,Input_feature_w,Output_feature_c,'','',Input_feature_n,Output_feature_h,Output_feature_w,add_flops,Input_size,Kernel_size,Output_size,Total_memory]


    def no_macs(node, input_shape, output_shape, attrs):
        Input_feature_n = input_shape[0]
        Input_feature_c = input_shape[1]
        Input_feature_h = input_shape[2]
        Input_feature_w = input_shape[3]

        # output shape
        Output_feature_n = output_shape[0]
        Output_feature_c = output_shape[1]
        Output_feature_h = output_shape[2]
        Output_feature_w = output_shape[3]
        
        No_flops = 0
        Input_size = np.prod(input_shape)
        Kernel_size = 0
        Output_size = np.prod(output_shape)
        Total_memory = Input_size + Kernel_size + Output_size
        return [Input_feature_c,Input_feature_h,Input_feature_w,Output_feature_c,'','',Input_feature_n,Output_feature_h,Output_feature_w,No_flops,Input_size,Kernel_size,Output_size,Total_memory]
    
    def pool_macs(node, input_shape, output_shape, attrs):
        Input_feature_n = input_shape[0]
        Input_feature_c = input_shape[1]
        Input_feature_h = input_shape[2]
        Input_feature_w = input_shape[3]

        # output shape
        Output_feature_n = output_shape[0]
        Output_feature_c = output_shape[1]
        Output_feature_h = output_shape[2]
        Output_feature_w = output_shape[3]
        
        Pool_flops = np.prod(input_shape) # maybe incorrectness.
        Input_size = np.prod(input_shape)
        Kernel_size = 0
        Output_size = np.prod(output_shape)
        Total_memory = Input_size + Kernel_size + Output_size
        return [Input_feature_c,Input_feature_h,Input_feature_w,Output_feature_c,'','',Input_feature_n,Output_feature_h,Output_feature_w,Pool_flops,Input_size,Kernel_size,Output_size,Total_memory]
      


    mac_calculators = {
        'Conv': conv_macs,
        'ConvTranspose':conv_macs,
        'Gemm': gemm_macs,
        'MatMul': gemm_macs,
        'BatchNormalization': bn_macs,
        'Relu': relu_macs,
        'Add': add_macs,
        'Reshape': no_macs,
        'Upsample': upsample_macs,
        'MaxPool':pool_macs,
        'AveragePool':pool_macs
    }

    csv_header = ['Net','Node Name','Node Attribute','Input Feature Map C','input height H','input Width W','Output Feature Maps k','kernel hight R',
    'kernal Width S','Group','Output feature Map height H','Output Feature Map Width Q','FLOPs','Input_size','Weight_size','Output_size','Memory Total']
    with open (csv_file_name,'w',newline='') as f:
        f_csv = csv.writer(f)
        f_csv.writerow(csv_header)
        for node in onnx_nodes:
            #print(node.output[0])
            node_output_shape = node_name_dict[node.output[0]]

            if len(node.input) > 0:
                node_input_shape = node_name_dict[node.input[0]]
             #print(node.name,node_output_shape,node_input_shape)
            if node.op_type in mac_calculators:
                node_list = mac_calculators[node.op_type](
                node, node_input_shape, node_output_shape, onnx_node_attributes_to_dict(node.attribute))
            else:
                node_list = []
            node_flops_list = [infered_model.graph.name,node.name,node.op_type] + node_list
            f_csv.writerow(node_flops_list)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='ONNX opcounter')
    parser.add_argument('model', type=str, help='Path to an ONNX model.')
    parser.add_argument('--calculate-macs', action='store_true', help='Calculate MACs.')
    
    args = parser.parse_args()

    model = onnx.load(args.model)
    csv_file_name = args.model.split('.')[0] + '.csv'
    print('Number of parameters in the model: {}'.format(calculate_params(model)))

    if args.calculate_macs:
       calculate_macs(model,csv_file_name)

