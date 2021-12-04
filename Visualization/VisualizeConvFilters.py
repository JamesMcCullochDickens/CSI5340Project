import ShowImageUtils as s_utils
import Backbones.ResNet as rn

def visualize_conv_filters(model, save_fp=None):
    module_list = list(model.children())
    first_conv = module_list[0]
    conv_weights = first_conv.weight
    current_im = None
    im_list = []
    for i in range(64):
        image_i = conv_weights[i]
        if image_i.shape[0] != 3:
            image_i = conv_weights[i].repeat(3, 1, 1)
        image_i = image_i.clone().detach().numpy()
        image_i = s_utils.normalizeDepthImage(image_i) # same functionality really
        image_i = s_utils.channelsFirstToLast(image_i)
        image_i = s_utils.resizeImage(image_i, 64, 64)
        im_list.append(image_i)
        if i % 7 == 0 and i != 0 and current_im is None:
            current_im = s_utils.concatenateImagesHorizontally(im_list)
            im_list = []
            continue
        if (i+1) % 8 == 0 and i != 0:
            current_im = s_utils.concatenateImagesVertically([current_im, s_utils.concatenateImagesHorizontally(im_list)])
            im_list = []
    if save_fp is not None:
        s_utils.saveImage(save_fp, current_im)
    s_utils.showImage(current_im)


def visualize_conv_filters_backbone(model, save_fp=None):
    model.cpu()
    module_list = list(model.backbone.children())
    first_conv = module_list[0]
    conv_weights = first_conv.weight
    current_im = None
    im_list = []
    for i in range(64):
        image_i = conv_weights[i]
        if image_i.shape[0] != 3:
            image_i = conv_weights[i].repeat(3, 1, 1)
        image_i = image_i.clone().detach().numpy()
        image_i = s_utils.normalizeDepthImage(image_i) # same functionality really
        image_i = s_utils.channelsFirstToLast(image_i)
        image_i = s_utils.resizeImage(image_i, 64, 64)
        im_list.append(image_i)
        if i % 7 == 0 and i != 0 and current_im is None:
            current_im = s_utils.concatenateImagesHorizontally(im_list)
            im_list = []
            continue
        if (i+1) % 8 == 0 and i != 0:
            current_im = s_utils.concatenateImagesVertically([current_im, s_utils.concatenateImagesHorizontally(im_list)])
            im_list = []
    if save_fp is not None:
        s_utils.saveImage(save_fp, current_im)
    s_utils.showImage(current_im)



"""
# testing
#model = rn.resnet50(pretrained=True)
model = rn.get_grayscale_rn50_backbone(pre_trained=True, with_pooling=False)
save_fp = None
visualize_conv_filters(model)
"""



