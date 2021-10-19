import torch
import torchvision
import numpy as np
import cv2
from itertools import tee
import math
#from BoxUtils import BB_Utils
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage

# vals chosen from https://www.rapidtables.com/web/color/RGB_Color.html
color_dict = {
# for seg values
0: [153, 0, 0], # unknown: dark red
1: [255, 128, 0], # wall: medium orange
2: [51, 102, 0], # floor: dark green
3: [255, 255, 204], # cabinet: light yellow
4: [0, 0, 0],  # bed: darkest black
5: [255, 255, 255], # chair: whitest white
6: [0, 51, 102], # sofa: dark blue
7: [128, 128, 128], # table: medium gray
8: [255, 51, 153], # door: medium pink
9: [153, 204, 255], # window: light blue
10: [255, 51, 51], # bookshelf: medium red
11: [204, 255, 153], # picture: light green
12: [255, 204, 204], # counter: light red
13: [127, 0, 255], # blinds: dark purple
14: [160, 160, 160], # desk: third lightest gray
15: [204, 153, 255], # shelves: light purple
16: [102, 0, 51], # curtain: dark magenta
17: [0, 102, 102], # dresser: dark cyan
18: [0, 255, 0], # pillow: medium green
19: [255, 229, 204], # mirror: light orange
20: [255, 204, 229], # floor mat: light pink
21: [255, 0, 127], # clothes: medium magenta
22: [0, 128, 255], # ceiling: medium blue
23: [224, 224, 224], # books: light gray
24: [0, 255, 255], # fridge: medium cyan
25: [255, 255, 0], # tv: medium yellow
26: [255, 0, 255], # paper: medium purple
27: [255, 204, 229], # towel: light magenta
28: [153, 76, 0], # shower curtain: dark orange
29: [204, 255, 204], # box: light simple green
30: [153, 153, 255], # whiteboard: light blue violet
31: [153, 255, 255], # person: light cyan
32: [0, 255, 128], # night stand: medium simple green
33: [255, 255, 204], # toilet: light yellow
34: [64, 64, 64], # sink: dark gray
35: [127, 0, 255], # lamp: medium blue violet
36: [0, 51, 25], # bathtub: dark simple green
37: [51, 0, 102], # bag: dark blue violet
38: [25, 78, 90],   # other-struct"
39: [230, 34, 78], # other-furniture
40: [200, 200, 1],  # other-prop
}

# save an image from a numpy array
def saveImage(fp, im):
    im_arr = Image.fromarray(im)
    im_arr.save(fp+".jpg")



def showImageWithBoundingBoxes(im, bounding_boxes, labels=None, confidence_scores=None):
    fnt = ImageFont.truetype("arial.ttf", 25)
    fill_ = (0, 255, 0, 255)
    im = np.uint8(im)
    image_height = im.shape[0]
    image_width = im.shape[1]
    im = Image.fromarray(im)
    img1 = ImageDraw.Draw(im)
    for index, box in enumerate(bounding_boxes):
        x_min = int(box[0])
        y_min = int(box[1])
        x_max = int(box[2])
        y_max = int(box[3])
        shape = [(x_min, y_min), (x_max, y_max)]
        img1.rectangle(shape, outline ="red", width=3)
        if labels is not None:
            label = str(labels[index])
            font_x_min = max(0, x_min+6)
            if font_x_min >= image_width:
                font_x_min = image_width - 10
            font_y_min = max(0, y_min-26)
            if confidence_scores is not None:
                label += " " + str(confidence_scores[index])
            img1.text((font_x_min, font_y_min), label, font=fnt, fill=fill_)
    im.show()


"""
random_image = np.random.rand(600, 600, 3)
random_image = random_image*255
random_image = random_image.astype(np.uint8)
bounding_boxes = [[200, 40, 500, 200], [20, 50, 100, 200]]
labels = ["cat", "dog"]
confidence_scores = [0.8, 0.9]
showImageWithBoundingBoxes(random_image, bounding_boxes, labels, confidence_scores)
debug = "debug"
"""

def showImage(image, with_resize=False, image_width=600, image_height=600):
    image = image.astype(np.uint8)
    image = Image.fromarray(image)
    image.show()
    if with_resize:
        image = image.resize((image_width, image_height))
        image.show()

def showImageWithLabel(image, label):
    fnt = ImageFont.truetype("arial.ttf", 25)
    fill_ = (0, 255, 0, 255)
    image = image.astype(np.uint8)
    image_width = image.shape[1]
    font_x_min = int(image_width/2) - 20
    font_y_min = 0
    image = Image.fromarray(image)
    image_draw = ImageDraw.Draw(image)
    image_draw.text((font_x_min, font_y_min), label, font=fnt, fill=fill_)
    image.show()



"""
test_image = np.zeros([300, 300, 3], dtype=np.uint8)
for i in range(200):
    test_image[200][i] = [100, 0, 0]
showImage(test_image, with_resize=True)
debug = "debug"
"""




def getRandomRGBVal():
    return np.random.randint(255, size=3)

def showSegmentationImage(seg_array, original_image = [], with_conversion=False):
    image_height = seg_array.shape[0]
    image_width = seg_array.shape[1]
    seg_image = np.empty([image_height, image_width, 3], dtype=np.uint8)
    random_color_dict = {}
    for i in range(image_height):
        for j in range(image_width):
                if seg_array[i][j] in color_dict:
                    seg_image[i][j] = color_dict[seg_array[i][j]]
                else:
                    if seg_array[i][j] not in random_color_dict:
                        random_color = getRandomRGBVal()
                        random_color_dict[seg_array[i][j]] = random_color
                        seg_image[i][j] = random_color
                    else:
                        seg_image[i][j] = random_color_dict[seg_array[i][j]]
    showImage(seg_image)
    if original_image != []:
        if with_conversion:
            original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        overlayed_segmentation = cv2.addWeighted(original_image, 0.60, seg_image, 0.40, 0)
        showImage(overlayed_segmentation)

def getSegmentationImage(seg_array, original_image = []):
    image_height = seg_array.shape[0]
    image_width = seg_array.shape[1]
    seg_image = np.zeros([image_height, image_width, 3], dtype=np.uint8)
    unique_vals = np.unique(seg_array)
    seg_array = np.expand_dims(seg_array, axis=2)
    seg_array = np.concatenate((seg_array, seg_array, seg_array), axis=2)
    for key in unique_vals:
        seg_image = np.where(seg_array == [key, key, key], color_dict[key], seg_image)
    seg_image = seg_image.astype(np.uint8)
    if original_image != []:
        overlayed_segmentation = cv2.addWeighted(original_image, 0.60, seg_image, 0.40, 0)
    return seg_image, overlayed_segmentation



def showBinaryInstanceMask(image, instance_mask, bb=None, labels=None, confidence_score=None, needs_conversion=False):
    if needs_conversion:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.astype(np.uint8)
    #showImage(image)
    instance_mask = instance_mask.astype(np.uint8)
    instance_mask = np.where(instance_mask == 1, 255, 0)
    mask_height = instance_mask.shape[0]
    mask_width = instance_mask.shape[1]
    showImage(instance_mask)
    instance_mask = instance_mask.reshape(mask_height, mask_width, 1)
    instance_mask = np.repeat(instance_mask, 3, axis=2)
    seg_image = np.where(instance_mask == [255,255,255], [255, 255, 255], [0, 0, 0]).astype(np.uint8)  # white
    overlayed_segmentation_image = cv2.addWeighted(image, 0.7, seg_image, 0.3, 0)
    if bb is None:
        showImage(overlayed_segmentation_image)
    else:
        showImageWithBoundingBoxes(overlayed_segmentation_image, bb, labels, confidence_score)

def showBinaryInstanceMasks(image, instance_masks, bbs=None, labels=None, confidence_scores=None):
    image = image.astype(np.uint8)
    seg_image = np.zeros_like(image)
    #showImage(image)
    max_val = 255
    for index, instance_mask in enumerate(instance_masks):
        current_val = max_val
        instance_mask = instance_mask.astype(np.uint8)
        instance_mask = np.where(instance_mask == 1, current_val, 0)
        mask_height = instance_mask.shape[0]
        mask_width = instance_mask.shape[1]
        #showImage(instance_mask)
        instance_mask = instance_mask.reshape(mask_height, mask_width, 1)
        instance_mask = np.repeat(instance_mask, 3, axis=2)
        new_color = np.random.randint(low=0, high=255, size=3)
        seg_image = np.where(instance_mask == [max_val, max_val, max_val], new_color, seg_image).astype(np.uint8)  # white
    overlayed_segmentation_image = cv2.addWeighted(image, 0.6, seg_image, 0.4, 0)
    showImage(seg_image)
    if bbs is None:
        showImage(overlayed_segmentation_image)
    else:
        showImageWithBoundingBoxes(overlayed_segmentation_image, bbs, labels, confidence_scores)

"""
random_image = np.random.randint(255, size=(600, 600, 3))
random_instance = np.zeros((600, 600))
for i in range(100, 200):
    for j in range(100, 200):
        random_instance[i][j] = 1
showBinaryInstanceMask(random_image, random_instance)
"""

def readImage(im_path):
    img = cv2.imread(im_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

"""
def visualizeIOU(bb1, bb2, image_width=600, image_height=600, image=None, onlyBoundary=False):
    red = np.array([250, 0, 0], dtype=np.uint8) # bb1 values not in bb1
    green = np.array([0, 250, 0], dtype=np.uint8) # bb2 values not in bb2
    blue = np.array([0, 0, 250], dtype=np.uint8) # intersection values
    bb_image = np.zeros((image_height, image_width, 3), dtype=np.uint8)
    for i in range(image_height):
        for j in range(image_width):
            coord = [j,i]
            isInBB1 = BB_Utils.isInBB(coord, bb1)
            isInBB2 = BB_Utils.isInBB(coord, bb2)
            isOnBoundary = BB_Utils.isOnBoundary(coord, bb1) or BB_Utils.isOnBoundary(coord, bb2)
            if isInBB1 and isInBB2:
                if not onlyBoundary:
                    bb_image[i][j] = blue
                else:
                    if isOnBoundary:
                        bb_image[i][j] = blue
            elif isInBB1:
                if not onlyBoundary:
                    bb_image[i][j] = red
                else:
                    if isOnBoundary:
                        bb_image[i][j] = red
            elif isInBB2:
                if not onlyBoundary:
                    bb_image[i][j] = green
                else:
                    if isOnBoundary:
                        bb_image[i][j] = green
    if image is not None:
        bb_image = cv2.addWeighted(bb_image, 0.7, image, 0.3, 0)
    showImage(bb_image)
"""
"""
# testing visualize IOU
bb1 = [100, 200, 300, 400]
bb2 = [90, 80, 250, 500]
visualizeIOU(bb1, bb2, onlyBoundary=False)
visualizeIOU(bb1, bb2, onlyBoundary=True)
"""

class NormalizeInverse(torchvision.transforms.Normalize):
    """
    Undoes the normalization and returns the reconstructed images in the input domain.
    """

    def __init__(self, mean, std):
        mean = torch.as_tensor(mean)
        std = torch.as_tensor(std)
        std_inv = 1 / (std + 1e-7)
        mean_inv = -mean * std_inv
        super().__init__(mean=mean_inv, std=std_inv)

    def __call__(self, tensor):
        return super().__call__(tensor.clone())

def unNormalizeImage(image):
    # these are actually 2012 values...
    image = image.cpu()
    inv_normalize = NormalizeInverse(mean=[0.485, 0.456, 0.406],  # image net weights
                                               std=[0.229, 0.224, 0.225])
    un_normalized_image = inv_normalize(image)
    un_normalized_image = un_normalized_image*255
    un_normalized_image = un_normalized_image.numpy().astype(np.uint8)
    un_normalized_image = np.transpose(un_normalized_image, (1, 2, 0))
    return un_normalized_image

def cleanBoundingBoxes(bounding_boxes):
    bounding_boxes_np = bounding_boxes[0]
    int_bounding_boxes = []
    for box in bounding_boxes_np:
        x_min = int(np.floor(box[0]))
        y_min = int(np.floor(box[1]))
        x_max = min(int(np.ceil(box[2])), 600)
        y_max = min(int(np.ceil(box[3])), 600)
        int_bounding_boxes.append([x_min, y_min, x_max, y_max])
    return int_bounding_boxes

# Scale a tensor [A, B] to [0, 1]
def linearScaling(tensor):
    A = torch.min(tensor)
    B = torch.max(tensor)
    range = (B-A).item()
    tensor = torch.divide((tensor-A), range)
    return tensor

"""
# testing linear scaling
t = torch.tensor([[20, 3, 4, 5], [20, 30, 1, 3]])
t1 = linearScaling(t)
debug = "debug"
"""

# input a tensor feature map (C * H * W)
def visualizeFeatureMap(feature_map, channel=1, with_resize=False, image_width=600, image_height=600):
    map = feature_map[channel, :, :]
    scaled_map = linearScaling(map)
    scaled_map = scaled_map.detach().cpu().numpy()
    scaled_map *= 255
    scaled_map = scaled_map.astype(np.uint8)
    if with_resize:
        showImage(scaled_map, True, image_width, image_height)
    else:
        showImage(scaled_map)

"""
# testing visualize feature map
map = torch.randint(1, 100000, (64, 30, 30))
visualizeFeatureMap(map, 1)
"""

"""
def showImageWithClosestBoundingBox(image, rps, bbs, label_names=None): # rps = region proposals, bbs = bounding boxes
    for index, bb in enumerate(bbs):
        if label_names is not None:
            print(label_names[index])
        rp = BB_Utils.getMaxIOU(bb, rps)[0]
        showImageWithBoundingBoxes(image, [rp, bb])
"""

def decodeImageToArray(im_path):
    image = Image.open(im_path)
    image = np.asarray(image).astype(np.uint8)
    return image

def isValidLocation(i, j, image_height, image_width):
    if i<0:
        return False
    if i>image_height-1:
        return False
    if j<0:
        return False
    if j>image_width-1:
        return False
    return True

def get8Neighbourhood(i, j, image_height, image_width):
    nbd = []
    for height_offset in [-1, 0, 1]:
        for width_offset in [-1, 0, 1]:
            if isValidLocation(i+height_offset, j+width_offset, image_height, image_width):
                nbd.append((i+height_offset, j+width_offset))
    return nbd

filters = []
for i in [0, 1, 2]:
    for j in [0, 1, 2]:
        filter = np.zeros([3,3], dtype=np.int)
        if i ==1 and j==1:
            pass
        else:
            filter[i][j] = -1
            filter[1][1] = 1
            filters.append(filter)

def getCountourImage(seg_image):
    convolved_images = []
    for filter in filters:
        convoled_image = ndimage.correlate(seg_image, filter, mode='reflect')
        convolved_images.append(convoled_image)
    convoled_images = np.add.reduce(convolved_images)
    seg_image = np.where(convoled_images != 0, 1, 0) # this has to be converted from 1 to 255 to notice a change
    return seg_image

# show with the original image
def showContourImage(original_image, contour_image):
    contoured_image = np.where(contour_image == [1, 1, 1], [255, 255, 255], original_image)
    showImage(contoured_image)

# the binary contour image
def displayOnlyContourImage(contour_image):
    contour_image = np.where(contour_image == 1, 255, 0)
    showImage(contour_image)

def getCannyEdgeDetectedImage(image):
    return cv2.Canny(image, threshold1=100, threshold2=200, edges=None, apertureSize=3)

def resizeImage(image, resize_width, resize_height):
    return cv2.resize(image, (resize_width, resize_height))


"""
test_image_path = "C:/Users/James/PycharmProjects/Masters/SUNRGBDdataset/SUNRGBD/kv2/kinect2data/000002_2014-05-26_14-23-37_260595134347_rgbf000103-resize/image/0000103.jpg"
im = readImage(test_image_path)
canny_im = getCannyEdgeDetectedImage(im)
showImage(canny_im)
"""

# assuming depth image is of the shape (height, width, 1)
def convertDepthToJetMap(depth_image):
    height = depth_image.shape[0]
    width = depth_image.shape[1]
    vmax = np.max(depth_image)
    vmin = np.min(depth_image)
    dv = vmax-vmin
    rgb_image = np.zeros((height, width, 3))

    # first case
    red = np.zeros((height, width, 1))
    green = 4 * (depth_image-vmin) / dv
    blue = np.ones((height, width, 1))
    replacement_array = np.concatenate((red, green, blue), axis=2)
    rgb_image = np.where(depth_image < (vmin + 0.25 * dv), replacement_array, rgb_image)

    # second case
    red = np.zeros((height, width, 1))
    green = np.ones((height, width, 1))
    blue = 1 + 4 * (vmin + 0.25 * dv - depth_image) / dv
    replacement_array = np.concatenate((red, green, blue), axis=2)
    rgb_image = np.where(np.logical_and(depth_image >= (vmin + 0.25 * dv), depth_image < (vmin + 0.5 * dv)), replacement_array, rgb_image)

    # third case
    red = 4 * (depth_image - vmin - 0.5 * dv) / dv
    green = np.ones((height, width, 1))
    blue = np.zeros((height, width, 1))
    replacement_array = np.concatenate((red, green, blue), axis=2)
    rgb_image = np.where(np.logical_and(depth_image >= (vmin + 0.5 * dv),  depth_image < (vmin + 0.75 * dv)), replacement_array, rgb_image)

    # fourth case
    red = np.ones((height, width, 1))
    green = 1 + 4 * (vmin + 0.75 * dv - depth_image) / dv
    blue = np.zeros((height, width, 1))
    replacement_array = np.concatenate((red, green, blue), axis=2)
    rgb_image = np.where(depth_image >= (vmin + 0.75 * dv), replacement_array, rgb_image)

    return (rgb_image*255).astype(np.uint8)

"""
# testing depth to jet map
depth_im_path = "C:/Users/James/PycharmProjects/Masters/SUNRGBDdataset/SUNRGBD/kv1/b3dodata/img_0063/depth_bfx/img_0063_abs.png"
depth_im = np.asarray(Image.open(depth_im_path))
depth_image_modified = (depth_im * (255/65535)).astype(np.uint8)
showImage(depth_image_modified)
depth_im = depth_im.reshape((depth_im.shape[0], depth_im.shape[1], 1))
jet_map = convertDepthToJetMap(depth_im)
showImage(jet_map)
"""

def channelsFirstToChannelsLast(im):
    return np.rollaxis(im, 0, 3)


# for an anchor box of area x, aspect ratio w/h
"""
def showAnchorBoxes(im, stride, anchor_size, aspect_ratio):
    ref_box = BB_Utils.getReferenceAnchorBox(anchor_size, aspect_ratio)
    im_height = im.shape[0]
    im_width = im.shape[1]
    num_h = math.floor(im_height/stride)
    num_w = math.floor(im_width/stride)
    stride_h = math.floor(im_height/num_h)
    stride_w = math.floor(im_width/num_w)
    im = Image.fromarray(im)
    img1 = ImageDraw.Draw(im)
    for h in range(num_h):
        for w in range(num_w):
            xmin = ref_box[0]+(stride_w*w)
            ymin = ref_box[1]+(stride_h*h)
            xmax = ref_box[2]+(stride_w*w)
            ymax = ref_box[3]+(stride_h*h)
            if xmin < 0 or ymin < 0 or xmax > im_width-1 or ymax > im_height-1:
                continue
            shape = [(xmin, ymin), (xmax, ymax)]
            img1.rectangle(shape, outline="red", width=3)
            im.show()
    im.show()
"""

def showAnchorBoxesCenters(im, stride, anchor_size, aspect_ratio):
    ref_box = BB_Utils.getReferenceAnchorBox(anchor_size, aspect_ratio)
    im_height = im.shape[0]
    im_width = im.shape[1]
    num_h = math.floor(im_height / stride)
    num_w = math.floor(im_width / stride)
    stride_h = math.floor(im_height / num_h)
    stride_w = math.floor(im_width / num_w)
    im = Image.fromarray(im)
    img1 = ImageDraw.Draw(im)
    for h in range(num_h):
        for w in range(num_w):
            xmin = ref_box[0] + (stride_w * w)
            ymin = ref_box[1] + (stride_h * h)
            xmax = ref_box[2] + (stride_w * w)
            ymax = ref_box[3] + (stride_h * h)
            xmin = int(xmin+xmax)/2
            xmax = xmin+5
            ymin = int(ymin+ymax)/2
            ymax = ymin+5
            shape = [(xmin, ymin), (xmax, ymax)]
            img1.rectangle(shape, outline="red", width=3)
    im.show()

def showSampleAnchorBoxes(im, stride, anchor_sizes, aspect_ratios):
    im_height = im.shape[0]
    im_width = im.shape[1]
    num_h = math.floor(im_height / stride)
    num_w = math.floor(im_width / stride)
    stride_h = math.floor(im_height / num_h)
    stride_w = math.floor(im_width / num_w)
    im = Image.fromarray(im)
    img1 = ImageDraw.Draw(im)
    for anchor_size in anchor_sizes:
        for aspect_ratio in aspect_ratios:
            ref_box = BB_Utils.getReferenceAnchorBox(anchor_size, aspect_ratio)
            xmin = ref_box[0] + (stride_w * 2)
            ymin = ref_box[1] + (stride_h * 2)
            xmax = ref_box[2] + (stride_w * 2)
            ymax = ref_box[3] + (stride_h * 2)
            if xmin < 0 or ymin < 0 or xmax > im_width - 1 or ymax > im_height - 1:
                continue
            shape = [(xmin, ymin), (xmax, ymax)]
            img1.rectangle(shape, outline="red", width=3)
    im.show()


"""
# testing showAnchorBoxes
im_path = "C:/Users/James/PycharmProjects/Masters/MS_COCO/train2014/COCO_train2014_000000000009.jpg"
im = np.asarray(Image.open(im_path))
showAnchorBoxesCenters(im, 32, 256, 1.5)
#showSampleAnchorBoxes(im, 32, [32, 64, 256, 512], [0.5, 1.0, 1.5])
"""

def channelsFirstToLast(arr):
    return np.moveaxis(arr, 0, -1)

def channelsLastToFirst(arr):
    return np.moveaxis(arr, -1, 0)

# assumes an input of numpy array of type np.uint8 of shape (h,w,3)
def convertToGrayScaleMaxMethod(im):
    return im.max(axis=2)

def convertToGrayScaleWeightedMethod(im):
    r_weight = 0.3
    g_weight = 0.59
    b_weight = 0.11
    r_vals = im[:, :, 0]
    g_vals = im[:, :, 1]
    b_vals = im[:, :, 2]
    grey_scale_im = r_weight*r_vals+g_weight*g_vals+b_weight*b_vals
    return grey_scale_im.astype(np.uint8)


def normalizeDepthImage(depth_im):
    max = np.amax(depth_im)
    min = np.amin(depth_im)
    depth_im_c = (depth_im-min)*(1/(max-min))
    depth_im_c *= 255
    return depth_im_c.astype(np.uint8)

"""
# testing the appearance of normalizeDepthImage
rgb_im1 = np.asarray(Image.open("F:/Datasets/NYUDv2/eccv14-data/data/images/img_5001.png"))
showImage(rgb_im1)
depth_im1 = np.asarray(Image.open("F:/Datasets/NYUDv2/eccv14-data/data/depth/img_5001.png"))
depth_im1 = normalizeDepthImage(depth_im1)
showImage(depth_im1)
"""


def getPSMask(seg_mask, instance_masks, seg_mask_max_label=40):
    label_num = seg_mask_max_label
    for instance_mask in instance_masks:
        label_num += 1
        seg_mask = np.where(instance_mask == 1, label_num, seg_mask)
    return seg_mask

def showPSMask(ps_mask, original_image=[], with_conversion=False):
    image_height = ps_mask.shape[0]
    image_width = ps_mask.shape[1]
    seg_image = np.empty([image_height, image_width, 3], dtype=np.uint8)
    new_color_dict = {}
    for i in range(image_height):
        for j in range(image_width):
            if ps_mask[i][j] <= 37:
                seg_image[i][j] = color_dict[ps_mask[i][j]]
            else:
                if ps_mask[i][j] in new_color_dict.keys():
                    seg_image[i][j] = new_color_dict[ps_mask[i][j]]
                else:
                    new_color_dict[ps_mask[i][j]] = np.random.randint(low=0, high=255, size=3)
                    seg_image[i][j] = new_color_dict[ps_mask[i][j]]

    showImage(seg_image)
    if original_image != []:
        if with_conversion:
            original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        overlayed_segmentation = cv2.addWeighted(original_image, 0.60, seg_image, 0.40, 0)
        showImage(overlayed_segmentation)



# np.random.randint(low=0, high=255, size=3)
# p_ps_mask for predicted panoptic segmentation mask
def getPsMask(p_ps_mask, im, num_classes):
    image_height = p_ps_mask.shape[0]
    image_width = p_ps_mask.shape[1]
    ps_mask = np.empty([image_height, image_width, 3], dtype=np.uint8)
    p_ps_mask = np.expand_dims(p_ps_mask, axis=2)
    p_ps_mask = np.concatenate((p_ps_mask, p_ps_mask, p_ps_mask), axis=2)
    unique_vals = np.unique(p_ps_mask)
    for val in unique_vals:
        if val <= num_classes:
            ps_mask = np.where(p_ps_mask == [val, val, val], color_dict[val], ps_mask)
        else:
            new_color = np.random.randint(low=0, high=255, size=3)
            ps_mask = np.where(p_ps_mask == [val, val, val], new_color, ps_mask)
    ps_mask = ps_mask.astype(np.uint8)
    overlayed_segmentation = cv2.addWeighted(im, 0.60, ps_mask, 0.40, 0)
    return ps_mask, overlayed_segmentation


"""
im_path = "C:/Users/James/PycharmProjects/Masters/MS_COCO/train2014/COCO_train2014_000000000009.jpg"
im = np.asarray(Image.open(im_path))
greyscale_im1 = convertToGrayScaleWeightedMethod(im)
showImage(greyscale_im1)
greyscale_im2 = convertToGrayScaleMaxMethod(im)
showImage(greyscale_im2)
"""

# assumes images are in channels last format and of the same size
# this is done horizontally
def concatenateImagesHorizontally(ims):
    return np.concatenate([im for im in ims], axis=1)

def concatenateImagesVertically(ims):
    return np.concatenate([im for im in ims], axis=0)

"""
# testing concatenate images
im_path = "C:/Users/James/PycharmProjects/Masters/MS_COCO/train2014/COCO_train2014_000000000009.jpg"
im1 = np.asarray(Image.open(im_path))
ims = [im1, im1]
concatenated_ims = concatenateImagesHorizontally(ims)
showImage(concatenated_ims)
"""