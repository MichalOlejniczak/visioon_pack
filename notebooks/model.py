
# coding: utf-8

# In[1]:

from keras.models import Model
from keras.layers import Input, Activation, concatenate
from keras.layers import Convolution2D, MaxPooling2D, UpSampling2D
from keras.layers import Deconvolution2D
from keras.layers import Conv2D, Conv2DTranspose, Concatenate
from keras.layers import Flatten
from keras.layers import Lambda, Reshape

import resize
import numpy as np
import matplotlib.colors as cl
import matplotlib.pyplot as plt
# In[2]:
#https://github.com/liruoteng/OpticalFlowToolkit/blob/master/lib/flowlib.py
UNKNOWN_FLOW_THRESH = 1e7

def read_flow(filename):
    """
    read optical flow from Middlebury .flo file
    :param filename: name of the flow file
    :return: optical flow data in matrix
    """
    f = open(filename, 'rb')
    magic = np.fromfile(f, np.float32, count=1)
    data2d = None

    if 202021.25 != magic:
        print ('Magic number incorrect. Invalid .flo file')
    else:
        w = np.fromfile(f, np.int32, count=1)
        h = np.fromfile(f, np.int32, count=1)
        print ("Reading %d x %d flo file") #% (h, w)
        data2d = np.fromfile(f, np.float32, count=2 * w * h)
        # reshape data into 3D array (columns, rows, channels)
        data2d = np.resize(data2d, (h, w, 2))
    f.close()
    return data2d

def visualize_flow(flow, mode='Y'):
    """
    this function visualize the input flow
    :param flow: input flow in array
    :param mode: choose which color mode to visualize the flow (Y: Ccbcr, RGB: RGB color)
    :return: None
    """
    if mode == 'Y':
        # Ccbcr color wheel
        img = flow_to_image(flow)
        plt.imshow(img)
        plt.show()
    elif mode == 'RGB':
        (h, w) = flow.shape[0:2]
        du = flow[:, :, 0]
        dv = flow[:, :, 1]
        valid = flow[:, :, 2]
        max_flow = max(np.max(du), np.max(dv))
        img = np.zeros((h, w, 3), dtype=np.float64)
        # angle layer
        img[:, :, 0] = np.arctan2(dv, du) / (2 * np.pi)
        # magnitude layer, normalized to 1
        img[:, :, 1] = np.sqrt(du * du + dv * dv) * 8 / max_flow
        # phase layer
        img[:, :, 2] = 8 - img[:, :, 1]
        # clip to [0,1]
        small_idx = img[:, :, 0:3] < 0
        large_idx = img[:, :, 0:3] > 1
        img[small_idx] = 0
        img[large_idx] = 1
        # convert to rgb
        img = cl.hsv_to_rgb(img)
        # remove invalid point
        img[:, :, 0] = img[:, :, 0] * valid
        img[:, :, 1] = img[:, :, 1] * valid
        img[:, :, 2] = img[:, :, 2] * valid
        # show
        plt.imshow(img)
        plt.show()

    return None

def flow_to_image(flow):
    """
    Convert flow into middlebury color code image
    :param flow: optical flow map
    :return: optical flow image in middlebury color
    """
    u = flow[:, :, 0]
    v = flow[:, :, 1]

    maxu = -999.
    maxv = -999.
    minu = 999.
    minv = 999.

    idxUnknow = (abs(u) > UNKNOWN_FLOW_THRESH) | (abs(v) > UNKNOWN_FLOW_THRESH)
    u[idxUnknow] = 0
    v[idxUnknow] = 0

    maxu = max(maxu, np.max(u))
    minu = min(minu, np.min(u))

    maxv = max(maxv, np.max(v))
    minv = min(minv, np.min(v))

    rad = np.sqrt(u ** 2 + v ** 2)
    maxrad = max(-1, np.max(rad))

    print ("max flow: %.4f\nflow range:\nu = %.3f .. %.3f\nv = %.3f .. %.3f" % (maxrad, minu,maxu, minv, maxv))

    u = u/(maxrad + np.finfo(float).eps)
    v = v/(maxrad + np.finfo(float).eps)

    img = compute_color(u, v)

    idx = np.repeat(idxUnknow[:, :, np.newaxis], 3, axis=2)
    img[idx] = 0

    return np.uint8(img)
def compute_color(u, v):
    """
    compute optical flow color map
    :param u: optical flow horizontal map
    :param v: optical flow vertical map
    :return: optical flow in color code
    """
    [h, w] = u.shape
    img = np.zeros([h, w, 3])
    nanIdx = np.isnan(u) | np.isnan(v)
    u[nanIdx] = 0
    v[nanIdx] = 0

    colorwheel = make_color_wheel()
    ncols = np.size(colorwheel, 0)

    rad = np.sqrt(u**2+v**2)

    a = np.arctan2(-v, -u) / np.pi

    fk = (a+1) / 2 * (ncols - 1) + 1

    k0 = np.floor(fk).astype(int)

    k1 = k0 + 1
    k1[k1 == ncols+1] = 1
    f = fk - k0

    for i in range(0, np.size(colorwheel,1)):
        tmp = colorwheel[:, i]
        col0 = tmp[k0-1] / 255
        col1 = tmp[k1-1] / 255
        col = (1-f) * col0 + f * col1

        idx = rad <= 1
        col[idx] = 1-rad[idx]*(1-col[idx])
        notidx = np.logical_not(idx)

        col[notidx] *= 0.75
        img[:, :, i] = np.uint8(np.floor(255 * col*(1-nanIdx)))

    return img

def make_color_wheel():
    """
    Generate color wheel according Middlebury color code
    :return: Color wheel
    """
    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6

    ncols = RY + YG + GC + CB + BM + MR

    colorwheel = np.zeros([ncols, 3])

    col = 0

    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.transpose(np.floor(255*np.arange(0, RY) / RY))
    col += RY

    # YG
    colorwheel[col:col+YG, 0] = 255 - np.transpose(np.floor(255*np.arange(0, YG) / YG))
    colorwheel[col:col+YG, 1] = 255
    col += YG

    # GC
    colorwheel[col:col+GC, 1] = 255
    colorwheel[col:col+GC, 2] = np.transpose(np.floor(255*np.arange(0, GC) / GC))
    col += GC

    # CB
    colorwheel[col:col+CB, 1] = 255 - np.transpose(np.floor(255*np.arange(0, CB) / CB))
    colorwheel[col:col+CB, 2] = 255
    col += CB

    # BM
    colorwheel[col:col+BM, 2] = 255
    colorwheel[col:col+BM, 0] = np.transpose(np.floor(255*np.arange(0, BM) / BM))
    col += + BM

    # MR
    colorwheel[col:col+MR, 2] = 255 - np.transpose(np.floor(255 * np.arange(0, MR) / MR))
    colorwheel[col:col+MR, 0] = 255

    return colorwheel

def FlowNet():
    
    input_img1 = Input(shape=(384,512,3))
    input_img2 = Input(shape=(384,512,3))
    c = concatenate([input_img1, input_img2], axis=3)

    conv1 = Conv2D(64, (7, 7), padding="same", name="conv1", activation="relu", strides=(2, 2))(c)
    conv2 = Conv2D(128, (5, 5), padding="same", name="conv2", activation="relu", strides=(2, 2))(conv1)
    conv3 = Conv2D(256, (3, 3), padding="same", name="conv3", activation="relu", strides=(2, 2))(conv2)
    conv3_1 = Conv2D(256, (3, 3), padding="same", activation="relu", name="conv3_1")(conv3)
    conv4 = Conv2D(512, (3, 3), padding="same", name="conv4", activation="relu", strides=(2, 2))(conv3_1)
    conv4_1 = Conv2D(512, (3, 3), padding="same", name="conv4_1", activation="relu")(conv4)
    conv5 = Conv2D(512, (3, 3), padding="same", name="conv5", activation="relu", strides=(2, 2))(conv4_1)
    conv5_1 = Conv2D(512, (3, 3), padding="same", name="conv5_1", activation="relu")(conv5)
    conv6 = Conv2D(1024, (3, 3), padding="same", name="conv6", activation="relu", strides=(2, 2))(conv5_1)
    conv6_1 = Conv2D(1024, (3, 3), padding="same", name="conv6_1", activation="relu", strides=(1, 1))(conv6)
    
    deconv5 = Conv2DTranspose(512, (4, 4), padding="same", name="deconv5", strides=(2, 2), activation="relu")(conv6_1)
    
    predict_flow6 = Conv2D(2, (3, 3), name="predict_flow6", padding="same", strides=(1, 1), activation="relu")(conv6_1)
    upsample_flow6to5 = Conv2DTranspose(2, (4, 4), padding="same", name="upsample_flow6to5", strides=(2, 2), activation="relu")(predict_flow6)

    concat5 = Concatenate(axis=3, name='concat5')([conv5_1, deconv5, upsample_flow6to5])
    
    deconv4 =Conv2DTranspose(256, (4, 4), activation="relu", name="deconv4", strides=(2, 2), padding="same")(concat5)
    
    predict_flow5=Conv2D(2, (3, 3), activation="relu", name="predict_flow5", strides=(1, 1), padding="same")(concat5)
    upsample_flow5to4=Conv2DTranspose(2, (4, 4), activation="relu", name="upsample_flow5to4", strides=(2, 2), padding="same")(predict_flow5)
    
    concat4 = concatenate([conv4_1,deconv4, upsample_flow5to4], axis = 3, name='concat4')
    
    deconv3 =Conv2DTranspose(128, (4, 4), activation="relu", name="deconv3", strides=(2, 2), padding="same")(concat4)
    
    predict_flow4=Conv2D(2, (3, 3), activation="relu", name="predict_flow4", strides=(1, 1), padding="same")(concat4)
    upsample_flow4to3=Conv2DTranspose(2, (4, 4), activation="relu", name="upsample_flow4to3", strides=(2, 2), padding="same")(predict_flow4)
    
    concat3 = concatenate([conv3_1,deconv3, upsample_flow4to3], axis = 3, name = 'concat3')

    deconv2 = Conv2DTranspose(64, (4, 4), name="deconv2", padding="same", activation="relu", strides=(2, 2))(concat3)
    
    predict_flow3=Conv2D(2, (3, 3), activation="relu", name="predict_flow3", strides=(1, 1), padding="same")(concat3)
    upsample_flow3to2=Conv2DTranspose(2, (4, 4), activation="relu", name="upsample_flow3to2", strides=(2, 2), padding="same")(predict_flow3)
    
    concat2 = concatenate([conv2,deconv2, upsample_flow3to2], axis = 3, name = 'concat2')
    
    predict_flow2 = Conv2D(2, (3, 3), activation="relu", name="predict_flow2", strides=(1, 1), padding="same")(concat2)
    
    eltwise = Lambda(lambda x: x + 20.0, name='eltwise')(predict_flow2)
    
    predict_flow_resize = UpSampling2D(size = (4, 4), name='reshape')(eltwise)
    predict_flow_final = Conv2D(2, (1, 1), name="predict_flow_final", padding="valid", strides=(1, 1))(predict_flow_resize)
    
    return Model(inputs = [input_img1, input_img2], outputs=[predict_flow_final])



