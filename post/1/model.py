import os
import numpy as np
import json
import cv2
import torch

from typing import List

from triton_python_backend_utils import Tensor, InferenceResponse, \
    get_input_tensor_by_name, InferenceRequest, get_input_config_by_name, \
    get_output_config_by_name, triton_string_to_numpy


def rle_encode(im_arr):
    height, width = im_arr.shape
    flat = im_arr.T.flatten()
    switches = np.nonzero(np.append(flat, 0) != np.append(0, flat))[0]
    rle_arr = (np.append(switches, switches[-1]) - np.append(0, switches))[0:-1]
    remaining = width * height - np.sum(rle_arr)
    if remaining > 0:
        rle_arr = np.append(rle_arr, remaining)
    return list(rle_arr)


def post_process(mask, scale, pad):
    print("--------<><<<<<< scale", scale)
    print("--------<><<<<<< pad", pad)
    print("-------->>>>>>> mask ", mask.shape)
    mask = np.array(mask[0])
    print("-------->>>>>>> mask11111 ", mask.shape)
    h, w = mask.shape[:2]
    print("-------->>>>>>> h, w ", h, w)
    print(" sasasassasass ", pad[1], h-pad[1], pad[0], w-pad[0])
    mask = mask[int(pad[1]):int(h-pad[1]), int(pad[0]):int(w-pad[0])]
    print("-------->>>>>>> mask pad off ",mask.shape)
    org_size = (int((w-2*pad[0])/scale[0]), int((h-2*pad[1])/scale[1]))
    print("------->>> org_size ", org_size)
    rles = []
    categories = []
    for i in range(30): # cityscape data has 30 classes
        img_out = np.zeros(mask.shape)
        img_out[mask == i] = 255
        
        if sum(sum(img_out)) == 0:
            continue
        cv2.imwrite(f"{i}.jpg", img_out)
        print("----->img_out ", i, img_out)
        img_out = np.array(cv2.resize(img_out, org_size, interpolation = cv2.INTER_AREA))
        img_out_bin = np.zeros(img_out.shape)
        img_out_bin[img_out>0] = 1
        cv2.imwrite(f"{i}_resize.jpg", img_out_bin*255)
        rle = rle_encode(img_out_bin)
        rle = [str(i) for i in rle]
        rle = ",".join(rle)
        rles.append(rle)
        categories.append(str(i+1))
    return rles, categories


class TritonPythonModel(object):
    def __init__(self):
        self.input_names = {
            'scale': 'scale',
            'pad': 'pad',
            'masks': 'masks',
        }
        self.output_names = {
            'rles': 'rles',
            'categories': 'categories',
        }

    def initialize(self, args):
        model_config = json.loads(args['model_config'])

        if 'input' not in model_config:
            raise ValueError('Input is not defined in the model config')

        input_configs = {k: get_input_config_by_name(
            model_config, name) for k, name in self.input_names.items()}
        for k, cfg in input_configs.items():
            if cfg is None:
                raise ValueError(
                    f'Input {self.input_names[k]} is not defined in the model config')
            if 'dims' not in cfg:
                raise ValueError(
                    f'Dims for input {self.input_names[k]} are not defined in the model config')
            if 'name' not in cfg:
                raise ValueError(
                    f'Name for input {self.input_names[k]} is not defined in the model config')

        if 'output' not in model_config:
            raise ValueError('Output is not defined in the model config')

        output_configs = {k: get_output_config_by_name(
            model_config, name) for k, name in self.output_names.items()}
        for k, cfg in output_configs.items():
            if cfg is None:
                raise ValueError(
                    f'Output {self.output_names[k]} is not defined in the model config')
            if 'dims' not in cfg:
                raise ValueError(
                    f'Dims for output {self.output_names[k]} are not defined in the model config')
            if 'name' not in cfg:
                raise ValueError(
                    f'Name for output {self.output_names[k]} is not defined in the model config')
            if 'data_type' not in cfg:
                raise ValueError(
                    f'Data type for output {self.output_names[k]} is not defined in the model config')

        self.output_dtypes = {k: triton_string_to_numpy(
            cfg['data_type']) for k, cfg in output_configs.items()}

    def execute(self, inference_requests: List[InferenceRequest]) -> List[InferenceResponse]:
        responses = []

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        for request in inference_requests:
            batch_in = {}
            for k, name in self.input_names.items():
                tensor = get_input_tensor_by_name(request, name)
                if tensor is None:
                    raise ValueError(f'Input tensor {name} not found ' f'in request {request.request_id()}')
                batch_in[k] = tensor.as_numpy()  # shape (batch_size, ...)

            batch_out = {k: [] for k, name in self.output_names.items(
            ) if name in request.requested_output_names()}

            in_scale = batch_in['scale']
            in_pad = batch_in['pad']
            in_masks = batch_in['masks'] # batching multiple images

            rs_rles = []
            rs_categories = []
            for masks, scale, pad in zip(in_masks, in_scale, in_pad): # single image
                image_rles, image_categories = post_process(masks, scale, pad)
                rs_rles.append(image_rles)
                rs_categories.append(image_categories)

            max_categories = max([len(i) for i in rs_categories])
            for ctg in rs_categories:
                for _ in range(max_categories - len(ctg)):
                    ctg.append("")
            max_rles = max([len(i) for i in rs_rles])
            for rles in rs_rles:
                for _ in range(max_rles - len(rles)):
                    rles.append("")

            batch_out['rles'] = rs_rles
            batch_out['categories'] = rs_categories

            # Format outputs to build an InferenceResponse
            output_tensors = [Tensor(self.output_names[k], np.asarray(
                out, dtype=self.output_dtypes[k])) for k, out in batch_out.items()]

            # TODO: should set error field from InferenceResponse constructor to handle errors
            # https://github.com/triton-inference-server/python_backend#execute
            # https://github.com/triton-inference-server/python_backend#error-handling
            response = InferenceResponse(output_tensors)
            responses.append(response)

        return responses