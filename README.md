# Custom Nodes for [ComfyUI](https://github.com/comfyanonymous/ComfyUI): CLIPSegPro
## This repository contains an enhanced custom node from [ComfyUI CLIPSeg repository](https://github.com/biegert/ComfyUI-CLIPSeg) to generate masks for image inpainting tasks based on text prompts.

It's named after the original node with "Pro" suffix featuring:

* Support GPU acceleration.
* Support batch images.
* Support specifying multiple text prompts separated by delimiter ";" in a single node to include multiple parts.


### CLIPSegPro
The CLIPSegPro node generates a binary mask for a given input image and text prompt.

**Inputs:**

- image: A torch.Tensor representing the input image.
- text: A string representing the text prompt.
- blur: A float value to control the amount of Gaussian blur applied to the mask.
- threshold: A float value to control the threshold for creating the binary mask.
- dilation_factor: A float value to control the dilation of the binary mask.

**Outputs:**

- tensor_bw: A torch.Tensor representing the binary mask.
- image_out_hm: A torch.Tensor representing the heatmap overlay on the input image.
- image_out_bw: A torch.Tensor representing the binary mask overlay on the input image.


## Installation

# ComfyUI Manager

Follwing this [guide](https://github.com/ltdrdata/ComfyUI-Manager#how-to-use) to install this extension


## Usage
Below is an example for the intended workflow. The [json file](https://github.com/hoveychen/ComfyUI-CLIPSegPro/blob/main/workflow/inpaint_CLIPSeg.json) for the example can be found inside the 'workflow' directory 
![](https://github.com/hoveychen/ComfyUI-CLIPSegPro/blob/main/workflow/workflow_0.png?raw=true)
![](https://github.com/hoveychen/ComfyUI-CLIPSegPro/blob/main/workflow/workflow_1.png?raw=true)
![](https://github.com/hoveychen/ComfyUI-CLIPSegPro/blob/main/workflow/workflow_2.png?raw=true)

## Requirements
- PyTorch
- CLIPSeg
- OpenCV
- numpy
- matplotlib

Make sure that you have the required libraries installed to the venv of ComfyUI.
