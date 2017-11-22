# -*- coding: utf-8 -*-
# qui nov 16 13:34:31 -02 2017
# qua nov 22 10:08:34 -02 2017
import numpy as np
import matplotlib.pyplot as plt
import cv2
import helper
import os
plt.rc('text', usetex=True)


folder, num_bins = helper.arguments()
images = helper.scan_folder(folder)
for raw_image,image_name in images:
    image      = cv2.bitwise_not(raw_image) / np.max(raw_image) # 0 < image < 1
    image_fft  = np.fft.fft2(image)
    power      = image_fft*np.conj(image_fft)
    correlation= np.real(np.fft.ifft2(power))/image.size
    correlation= (correlation - np.mean(image)**2.0) # /(np.var(image)+epsilon)
    # fking shit normalization

    # using log to highlight variations
    correlation= np.log(np.abs(correlation))

    plt.figure()
    plt.subplot(1,2,1)
    plt.title('$I(x,y)$')
    plt.imshow(image,cmap='gray')
    plt.subplot(1,2,2)
    plt.title('$\\log C(x,y)$')
    plt.imshow(correlation,cmap='gray') ;
    filename,ext = os.path.splitext(image_name)
    filename = 'correlation_'+filename + '.pdf'
    plt.savefig(filename,dpi=600)



    
#plt.show()
