# -*- coding: utf-8 -*-
"""
Created on Tue Oct 23 18:56:00 2018

@author: chenc
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 15 13:32:36 2018

@author: chenc
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Oct 13 00:24:24 2018

@author: chenc
improves the accuracy from 0.8395 to 0.95
"""

# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
from sklearn.preprocessing import StandardScaler

'''
class MidpointNormalize(Normalize):
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))
'''    



dataset = np.load('../input_data/mnist_dataset.npz')
size_image=28
dim_image=1 

#size_image=28
#dim_image=1

Xtr = dataset ['Xtr']
Str = dataset ['Str'].ravel()

scaler = StandardScaler()
Xtr = scaler.fit_transform(Xtr.T).T


from densratio import densratio
PY1=sum(Str)/Str.shape
PY0=1-PY1
XY1=Xtr[Str==1,:]
XY0=Xtr[Str==0,:]
XY1oX=densratio(XY1,Xtr)
XY10XV=min(XY1oX.compute_density_ratio(Xtr))
Y1X=XY10XV*PY1 #array([0.20543082])

XY0oX=densratio(XY0,Xtr)
XY00XV=min(XY0oX.compute_density_ratio(Xtr))
Y0X=XY00XV*PY0 #array([0.28450032])