from clipseg import CLIPSegPro
"""
@author: hoveychen
@title: CLIPSegPro
@nickname: clipseg pro
@description: This repository contains an enhanced custom node to generate masks for image inpainting tasks based on text prompts by CLIP.
"""

# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "CLIPSegPro": CLIPSegPro,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CLIPSegPro": "Generate marks with CLIP"
}