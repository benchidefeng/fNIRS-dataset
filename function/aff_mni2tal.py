# -*- coding: utf-8 -*-
"""
Created on Mon Apr 17 19:49:46 2023
@function: Transform MNI coordinate to Talairach coordinate
#Input:    MNI coordinate
#Output:   Talairach coordinate
@author: Xiaofei Zhang
@email:csezxf@163.com

# """
import numpy as np

def aff_mni2tal(inpoint):
    Tfrm = np.array([[0.88, 0, 0, -0.8],
                     [0, 0.97, 0, -3.32],
                     [0, 0.05, 0.88, -0.44],
                     [0, 0, 0, 1]])
    tmp = np.dot(Tfrm, np.append(inpoint, 1))
    outpoint = tmp[:3]
    return outpoint
if __name__ == '__main__':
    MNI_coordinate=[38.47,70.37,19.56]
    Talairachcoordinate=aff_mni2tal(MNI_coordinate)
