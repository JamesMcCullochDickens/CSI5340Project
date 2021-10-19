import numpy as np
import ShowImageUtils as s_utils
import scipy
#from scipy.sparse.linalg import spsolve
from PIL import Image
from pypardiso import spsolve
from numba import double, jit
import numba
from numba import cuda

# fill_depth_colorization.m
# Preprocesses the kinect depth image using a gray scale version of the
# RGB image as a weighting for the smoothing. This code is a slight
# adaptation of Anat Levin's colorization code:
#
# See: www.cs.huji.ac.il/~yweiss/Colorization/

def fill_depth_colorization(imgRgb=None, imgDepthInput=None, alpha=1.):
    '''
    :param imgRgb: - HxWx3 matrix, the rgb image for the current frame. This must be between 0 and 1.
    :param imgDepthInput:  HxW matrix, the depth image for the current frame in absolute (meters) space.
    :param alpha: a penalty value between 0 and 1 for the current depth values.
    :return: Filled depth
    '''

    # the values of the depth image are  0 vals
    imgIsNoise = imgDepthInput == 0


    # normalize the depth image to be in between 0 and 1
    maxImgAbsDepth = np.max(imgDepthInput)
    imgDepth = imgDepthInput / maxImgAbsDepth
    imgDepth[imgDepth > 1] = 1


    # get the image shape
    (H, W) = imgDepth.shape

    # get the total number of pixels
    numPix = H * W


    # columns count up to the total number of pixels
    indsM = np.arange(numPix).reshape((W, H)).transpose()

    # set the values where there is noise to be 0, and 1 otherwise in a mask
    knownValMask = (imgIsNoise == False).astype(int) # valid values regions


    # convert the RGB image to gray scale
    grayImg = s_utils.convertToGrayScaleWeightedMethod(imgRgb)


    # the radius of the window
    winRad = 1


    len_ = 0
    absImgNdx = 0
    len_window = (2 * winRad + 1) ** 2
    len_zeros = numPix * len_window

    # these are row vectors initialized to -1
    cols = np.zeros(len_zeros) - 1
    rows = np.zeros(len_zeros) - 1
    vals = np.zeros(len_zeros) - 1
    gvals = np.zeros(len_window) - 1

    for j in range(W):
        for i in range(H):
            nWin = 0
            for ii in range(max(0, i - winRad), min(i + winRad + 1, H)):
                for jj in range(max(0, j - winRad), min(j + winRad + 1, W)):
                    if ii == i and jj == j: # ignore diagonal values of each window
                        continue

                    rows[len_] = absImgNdx # absolute image index
                    cols[len_] = indsM[ii, jj] # get the column of the current window pixel
                    gvals[nWin] = grayImg[ii, jj] # get the grey value of the current window

                    len_ = len_ + 1 # a total counter for pixels in all windows
                    nWin = nWin + 1 # a counter just for pixels in the window

            # the current gray value at position pixel i,j  (height, width)
            curVal = grayImg[i, j]


            # The last value in the gvals array is the curval
            gvals[nWin] = curVal


            # standard deviation (??)
            c_var = np.mean((gvals[:nWin + 1] - np.mean(gvals[:nWin + 1])) ** 2)
            csig = c_var * 0.6

            # min grey value
            mgv = np.min((gvals[:nWin] - curVal) ** 2)


            # making sure csig not too small?
            if csig < -mgv / np.log(0.01):
                csig = -mgv / np.log(0.01)

            if csig < 2e-06:
                csig = 2e-06



            gvals[:nWin] = np.exp(-(gvals[:nWin] - curVal) ** 2 / csig) # compute the distance from the window pixel to the
            gvals[:nWin] = gvals[:nWin] / sum(gvals[:nWin]) # normalize the values
            vals[len_ - nWin:len_] = -gvals[:nWin]

            # Now the self-reference (along the diagonal).
            rows[len_] = absImgNdx
            cols[len_] = absImgNdx
            vals[len_] = 1  # sum(gvals(1:nWin))

            len_ = len_ + 1
            absImgNdx = absImgNdx + 1

    vals = vals[:len_]
    cols = cols[:len_]
    rows = rows[:len_]
    A = scipy.sparse.csr_matrix((vals, (rows, cols)), (numPix, numPix))

    rows = np.arange(0, numPix)
    cols = np.arange(0, numPix)
    vals = (knownValMask * alpha).transpose().reshape(numPix)
    G = scipy.sparse.csr_matrix((vals, (rows, cols)), (numPix, numPix))

    A = A + G

    # this is an element-wise multiplication
    # flatten('F') is the column wise flatten operation
    b = np.multiply(vals.reshape(numPix), imgDepth.flatten('F'))

    # print ('Solving system..')

    new_vals = spsolve(A, b)
    new_vals = np.reshape(new_vals, (H, W), 'F')

    # print ('Done.')

    denoisedDepthImg = new_vals * maxImgAbsDepth

    output = denoisedDepthImg.reshape((H, W)).astype('float32')

    output = np.multiply(output, (1 - knownValMask)) + imgDepthInput

    return output

"""
# testing denoiseDepthImage
rgb_im = np.asarray(Image.open("F:/Datasets/NYUDv2/eccv14-data/data/images/img_5001.png"))/255

depth_im = np.asarray(Image.open("F:/Datasets/NYUDv2/eccv14-data/data/rawdepth/img_5001.png"))

denoised_depth_im = fill_depth_colorization(rgb_im, depth_im, 1.)

denoised_depth_im = s_utils.showImage(denoised_depth_im)
"""