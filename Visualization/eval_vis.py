
import ShowImageUtils as s_utils
import Backbones.ResNet as rn
import numpy as np
import DataLoaders.Unlabeled_dl as dl
import torch
import torchvision
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt





def visualize_feat_maps(model,img_file):
    
    # img_file is the name of the image (for example "img.png")
    
    # Storing the conv layers in a list
    conv_list = []
    conv_list.append(list(model.named_children())[0][1])
    module_list = list(model.named_modules())
    for name,module in module_list:
        if "conv" in name.lower():
            conv_list.append(module)
        #elif "downsample.0" in name.lower():
            #conv_list.append(module)
    
    # Getting the image and converting it to tensor
    img = Image.open(img_file)
    img = torchvision.transforms.functional.to_grayscale(img)
    print("Original image")
    plt.imshow(img, cmap='gray')
    plt.axis("off")
    plt.show()
    plt.close()
    convert_tensor = transforms.ToTensor()
    img_tensor = convert_tensor(img)
    img_tensor = torch.unsqueeze(img_tensor, 0)
     
    # Passing the image in the conv layers and storing the outputs
    outputs = [conv_list[0](img_tensor)]
    for i in range(1, len(conv_list)):
        outputs.append(conv_list[i](outputs[-1]))
        
    # Displaying the first 20 channels of the feature map for some of the conv layers  (Some code was taken from https://androidkt.com/how-to-visualize-feature-maps-in-convolutional-neural-networks-using-pytorch/) 
    for num_layer in range(len(outputs)):
        if num_layer % 10 == 0:
            plt.figure(figsize=(50, 10))
            layer_viz = outputs[num_layer][0, :, :, :]
            layer_viz = layer_viz.data
            print("Conv layer",num_layer+1)
            for i, featmap in enumerate(layer_viz):
                if i == 20: 
                    break
                plt.subplot(2, 10, i + 1)
                plt.imshow(featmap, cmap='gray')
                plt.axis("off")
            plt.show()
            plt.close()

"""            
# Testing the function 
model = rn.get_grayscale_rn50_backbone(pre_trained=True, with_pooling=False)
imgfile = "img.png"
visualize_feat_maps(model,imgfile)
"""
