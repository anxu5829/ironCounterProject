import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
# import sys
# sys.path.append()

from stableMethod.utils import *


def getIntervalOfIron(cut,img_diff_abs,maxInterval = 100,bandWidth = 10):
    getCoords = np.argwhere(img_diff_abs > cut).ravel()
    start = 0
    end = 0
    startHis = 0
    endHis = 0
    for coords in range(getCoords.shape[0]-1):
        if getCoords[coords+1] - getCoords[coords] > maxInterval:
            # print(getCoords[coords+1],getCoords[coords])
            start = coords
            end = coords
        else:
            end = coords+1
        if getCoords[end] - getCoords[start] >= getCoords[endHis] - getCoords[startHis]:
            startHis = start +1
            endHis = end
    if img_diff_abs[(getCoords[startHis] - bandWidth): getCoords[startHis]].shape[0] != 0 and  img_diff_abs[getCoords[endHis] : (getCoords[endHis] + bandWidth ) ].shape[0] != 0 :
        s = getCoords[startHis] + np.argmin(img_diff_abs[(getCoords[startHis] - bandWidth) : getCoords[startHis]]) - bandWidth
        e = getCoords[endHis] + np.argmin(img_diff_abs[getCoords[endHis] : (getCoords[endHis] + bandWidth ) ])
        return (s,e)
    else:
        return (getCoords[startHis],getCoords[endHis])



def getSplitPoint(img_diff):
    img_diff_abs = np.abs(img_diff)

    above_half_range = img_diff_abs[img_diff_abs > 0.5 * img_diff_abs.max()]
    height = above_half_range[np.logical_and(above_half_range < np.percentile(above_half_range, 80),
                                             above_half_range > np.percentile(above_half_range, 20))].mean()

    ironRegion = []
    for alpha in np.logspace(1, 3, 10) / 1000:
        a, b = getIntervalOfIron(alpha * height, img_diff_abs)
        ironRegion.append(b - a)

    ironRegion = np.array(ironRegion)
    alpha = (np.logspace(1, 3, 10) / 1000)[np.argmin(np.diff(ironRegion)) + 1]
    a, b = getIntervalOfIron(alpha * height, img_diff_abs)

    return  a,b

def getImageSplitPoint(img,precision = 5):
    row = jpg.shape[0]
    cutPoint =(np.linspace(0,1,precision+2)*row).astype(int)[1:-1]
    start = []
    end = []
    for cut in cutPoint:
        img_diff = (np.diff(img_corrected[cut:(cut + 80), :].mean(0)))
        a, b = getSplitPoint(img_diff)
        start.append(a)
        end.append(b)
    return np.min(a),np.max(b)


if __name__ == "__main__":
    os.chdir(r"C:\Users\an\Documents\work\铜片\opencv快速入门\chapter01 图像调整\sample")
    jpg = cv2.imread("sample4.jpg",0)

    # gamma 变换调整图像

    img_corrected = gamma_trans(jpg, 0.5)


    start , end = getImageSplitPoint(img_corrected)

