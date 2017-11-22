#!/usr/bin/python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import os


c_imag = 1.0j

def scan_folder(folder):
    """
    collect files inside a folder and load them into memory
    """
    path = list(os.walk(folder))
    if len(path) < 1:
        return []
    image = []
    for filename in path[0][2]:
        _,extension = os.path.splitext(filename)
        if extension in ['.jpg','.jpeg','.png']:
            # open grayscale image
            image.append( (cv2.imread(os.path.join(folder,filename),0),filename) )
    return image
def fourier(image):
    """
    simple handler to compute FFT from a single image and both wave vectors
    """
    return np.fft.fft2(image),np.fft.fftfreq(image.shape[0]),np.fft.fftfreq(image.shape[1])
def density(f):
    """
    returns the density probability function from the power distr. f
    """
    tmp = np.real(f*np.conj(f))
    return tmp/np.sum(tmp)
def density_radius(pdf,x,y,num_bins=50):
    """
    returns the pdf in polar coordinates.
    x          : input float. x-direction coords
    y          : input float. y-direction coords.
    num_bins   : input. optional.
    radius     : output float. various values of radius
    pdf_radius : output float. pdf in polar coordinates
    """
    r_max = np.sqrt(np.max(x*x) + np.max(y*y))
    dr    = r_max /num_bins
    radius = np.arange(0,r_max+dr,dr)
    pdf_radius = np.zeros(radius.size)
    for i,X in enumerate(x):
        for j,Y in enumerate(y):
            index = np.int( np.sqrt(X*X+Y*Y) / dr)
            pdf_radius[index] += pdf[i,j]
    return radius,pdf_radius
def density_radius_from_image(image,num_bins=50):
    image_fourier,kx,ky = fourier(image)
    pdf_fourier = density(image_fourier)
    return density_radius(pdf_fourier,kx,ky,num_bins)
def moments(pdf,x,m=10):
    """
    returns the first m statistical moments of random variable x with pdf
    """
    return [ np.sum( pdf*(x**k) ) for k in range(m) ]
def generate_graphs(image,image_name,plotter):
    k_,pdf_ = density_radius_from_image(image,num_bins=np.max(image.shape) // 5 )
    plotter.figure()
    plotter.clf()
    #plotter.subplots(2,2)
    #plotter.tight_layout()
    plotter.subplots_adjust(wspace=0.3)
    plotter.subplot(2,2,1)
    plotter.axis('off')
    plotter.imshow(image,cmap='gray')
    plotter.subplot(2,2,3)
    axis = plotter.gca()
    axis.set_ylim([0.0001,1])
    plotter.xscale('log')
    plotter.yscale('log')
    plotter.xlabel('$k$')
    plotter.ylabel('$\\rho(k)$')
    plotter.plot(k_,pdf_,'.-')
    plotter.subplot(2,2,4)
    plotter.xlabel('$\\ell$')
    plotter.ylabel('$\\mu_{\\ell}$')
    plotter.plot(moments(pdf_,k_)[1:],'x-')
    #plotter.subplot(2,2,4)
    filename,ext = os.path.splitext(image_name)
    filename = filename + '_processed.pdf'
    plotter.savefig(filename,dpi=600)
    return None
def arguments():
    import argparse
    cwd = os.getcwd()
    parser = argparse.ArgumentParser(' ')
    parser.add_argument('-f','--folder',default=str(cwd),help='path to image folder')
    parser.add_argument('-n','--number-bins',default=50,help='number of bins used to parse k-pdf')
    args=vars(parser.parse_args())
    return args['folder'],int(args['number_bins'])



















if __name__ == '__main__':
    # test ::1 scan_folder
    images = scan_folder('/home/gilberto/Documentos/posdoc/drylands/numerical/imagens_binarizadas/')
    print('test #1 :: expected number of images = 30 | result = %d' % len(images) )
    #test ::2 fourier
    n=256
    test = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            test[i,j] = np.sin(2*np.pi*i/n)*np.sin(200 *2*np.pi*j/n)
    delta_freq = [np.fft.fftfreq(test.shape[0])[:2],np.fft.fftfreq(test.shape[1])[:2]]
    delta_freq = [delta_freq[0][1] - delta_freq[0][0],delta_freq[1][1] - delta_freq[1][0],]
    f_test,kx,ky    = fourier(test)
    power1 = np.sum(test*test)
    power2 = np.sum(np.real(f_test * np.conj(f_test)))*np.prod(delta_freq)
    print('test #2 :: fourier (parseval) expected %f | result %f' % (power1,power2))
    #test ::3 density
    pdf = density(f_test)
    print('test #3 :: pdf        : integral over total domain = %f' % np.sum(pdf))
    #test ::4 density_radius
    #k,pdf_radius = density_radius(pdf,kx,ky)
    k_,pdf_radius = density_radius(pdf,kx,ky)
    print('test #4 :: pdf_radius : integral over total domain = %f' % np.sum(pdf_radius))
    #test ::5 moments
    moments_ = moments(pdf_radius,k_)
    variance = moments_[2] - moments_[1]**2.0
    print('test #5 :: moments    : expected variance = 0.00000 | result =%f' % (variance))
