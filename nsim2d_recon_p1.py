# -*- coding: utf-8 -*-
"""
Created on Tue Jun 13 15:59:11 2023

@author: sl52873
"""

import os
from pylab import imshow, subplot, subplots, figure, colorbar
import psfsim36 as psfsim
import numpy as np
import tifffile as tf
from scipy import signal
# from cupyfft import fft2, ifft2
# import cv2 
from skimage.filters import window

fft2 = np.fft.fft2
ifft2 = np.fft.ifft2
fftshift = np.fft.fftshift


class si2D(object):

    def __init__(self, fns, farme_order, nangles, nphs, norders, wavelength, na, dx):
        self.path = os.path.dirname(os.path.abspath(fns))
        self.farme_order = farme_order
        self.na = na
        self.nph = nphs
        self.nang = nangles
        self.wl = wavelength
        self.norders = norders
        self.img = tf.imread(fns)
        nz, nx, ny = self.img.shape

        self.img = self.img.reshape(self.nang, self.nph, nx, ny)
        
        # self.hanning_window()
         
        self.dx = dx/3
        self.nx = nx*3
        self.ny = ny*3
        self.psf = self.getpsf()
        self.meshgrid()
        self.sepmat = self.sepmatrix()
        self.mu = 0.02
        self.cutoff = 0.004
        self.strength = 1.
        self.sigma = 8.
        self.eh = []
        self.eta = 0.08
        self.winf = self.winfwindow(self.eta)


    def getpsf(self):
        ps = psfsim.psf()
        ps.setParams(wl=self.wl, na=self.na, dx=self.dx, nx=self.nx)
        self.radius = ps.radius
        ps.getFlatWF()
        psf1 = np.abs((fft2(fftshift(ps.bpp))))**2
        psf1 = psf1 / psf1.sum()
        return psf1
    
    def hanning_window(self):
        angle, nph, Nw, Nw = self.img.shape
        for i in range(angle):
            for m in range(nph):
                img = self.img[i,m,:,:] 
                filtered_img = img * window('hamming', img.shape)
                self.img[i,m,:,:] = filtered_img
        # tf.imsave(self.path + '/hanning_image.tif', self.img, photometric='minisblack')

    def meshgrid(self):
        x = np.arange(-self.nx/2, self.nx/2)
        y = np.arange(-self.ny/2, self.ny/2)
        xv, yv = np.meshgrid(x, y, indexing='ij', sparse=True)
        self.xv = np.roll(xv, int(self.nx/2))
        self.yv = np.roll(yv, int(self.ny/2))

    def sepmatrix(self):
        nphases = self.nph
        norders = int((self.norders + 1) / 2)
        sepmat = np.zeros((self.norders, nphases), dtype=np.float32)
        for j in range(nphases):
            sepmat[0, j] = 1.0
            for order in range(1, norders):
                sepmat[2 * order - 1, j] = 2 * np.cos( 2 * np.pi * (j * order) / nphases ) / nphases
                sepmat[2 * order, j] = 2 * np.sin(2 * np.pi * (j * order) / nphases ) / nphases
        # return np.linalg.inv(np.transpose(sepmat))
        # return sepmat/nphases
        return np.linalg.pinv(np.transpose(sepmat))
        
    # def hamming_window(self):
    #     angle, nph, Nw, Nw = self.img.shape
    #     ham = np.hamming(Nw)[:,None]
    #     ham2d = np.sqrt(np.dot(ham, ham.T))
    #     for i in range(angle):
    #         for m in range(nph):
    #             img = self.img[i,m,:,:] 
    #             r = Nw-10 # how narrower the window is
    #             ham = np.hamming(Nw)[:,None] # 1D hamming
    #             ham2d = np.sqrt(np.dot(ham, ham.T)) ** r # expand to 2D hamming
    #             f = cv2.dft(img.astype(np.float32), flags=cv2.DFT_COMPLEX_OUTPUT)
    #             f_shifted = np.fft.fftshift(f)
    #             f_complex = f_shifted[:,:,0]*1j + f_shifted[:,:,1]
    #             f_filtered = ham2d * f_complex
    #             f_filtered_shifted = np.fft.fftshift(f_filtered)
    #             inv_img = np.fft.ifft2(f_filtered_shifted) # inverse F.T.
    #             filtered_img = np.abs(inv_img)
    #             filtered_img -= filtered_img.min()
    #             filtered_img = filtered_img *255 / filtered_img.max()
    #             self.img[i,m,:,:] = filtered_img.astype(np.uint8)
    #     tf.imsave(self.path + '/hamming_image.tif', self.img[0,0,:,:].real.astype(np.float32), photometric='minisblack')
    
    
          
    def separate(self, nangle):
        angle, nph, Nw, Nw = self.img.shape
        outr = np.dot(self.sepmat, self.img[nangle].reshape(nph, Nw ** 2))
        self.separr = np.zeros((self.norders, 3*Nw, 3*Nw), dtype=np.complex64)
        self.separr[0] = np.fft.fftshift(self.interp(outr[0].reshape(Nw, Nw)) * self.winf)
        self.separr[1] = np.fft.fftshift(self.interp(((outr[1] + 1j * outr[2])/2).reshape(Nw, Nw)) * self.winf)
        self.separr[2] = np.fft.fftshift(self.interp(((outr[1] - 1j * outr[2])/2).reshape(Nw, Nw)) * self.winf)

            
    
        
    # def separate(self, nangle):
    #     angle, nph, Nw, Nw = self.img.shape
    #     outr = np.dot(self.sepmat, self.img[nangle].reshape(nph, Nw ** 2))
    #     self.separr = np.zeros((self.norders, 3*Nw, 3*Nw), dtype=np.complex64)
    #     self.separr[0] = np.fft.fftshift(self.interp(outr[0].reshape(Nw, Nw)) * self.winf)
    #     self.separr[1] = np.fft.fftshift(self.interp(((outr[1] + 1j * outr[2])/2).reshape(Nw, Nw)) * self.winf)
    #     self.separr[2] = np.fft.fftshift(self.interp(((outr[1] - 1j * outr[2])/2).reshape(Nw, Nw)) * self.winf)
    #     self.separr[3] = np.fft.fftshift(self.interp(((outr[3] + 1j * outr[4])/2).reshape(Nw, Nw)) * self.winf)
    #     self.separr[4] = np.fft.fftshift(self.interp(((outr[3] - 1j * outr[4])/2).reshape(Nw, Nw)) * self.winf)
        
        
    def shift0(self):
        self.otf0 = fft2(self.psf)
        # zsp = self.zerosuppression(0, 0)
        self.otf0 = self.otf0
        self.imgf0 = fft2(self.separr[0])
    

        
    # def shift0(self):
    #     self.otf0 = fft2(self.psf)
    #     zsp = self.zerosuppression(0, 0)
    #     self.otf0 = self.otf0 * zsp
    #     self.imgf0 = fft2(self.separr[0])


    def shift1(self, angle, spacing):
        ''' shift data in freq space by multiplication in real space '''
        Nw = self.nx
        dx = self.dx
        kx = dx * np.cos(angle) / spacing
        ky = dx * np.sin(angle) / spacing
        # shift matrix
        ysh = np.zeros((2, Nw, Nw), dtype=np.complex64)
        ysh[0, :, :] = np.exp(2j * np.pi * (kx * self.xv + ky * self.yv))
        ysh[1, :, :] = np.exp(2j * np.pi * (-kx * self.xv - ky * self.yv))
        # 1st order positive
        self.otf_1_0 = fft2(self.psf * ysh[0])
        yshf = np.abs(fft2(ysh[0]))
        sx, sy = np.unravel_index(yshf.argmax(), yshf.shape)
        if (sx < Nw/2):
            sx = sx
        else:
            sx = sx - Nw
        if (sy < Nw/2):
            sy = sy
        else:
            sy = sy - Nw
        temp = np.sqrt(sx ** 2 + sy ** 2) / (2 * self.radius)
        self.eh = np.append(self.eh, temp)
        zsp = self.zerosuppression(sx, sy)
        self.otf_1_0 = self.otf_1_0 * zsp
        self.imgf_1_0 = fft2(self.separr[1] * ysh[0])
        # 1st order negative
        self.otf_1_1 = fft2(self.psf * ysh[1])
        yshf = np.abs(fft2(ysh[1]))
        sx, sy = np.unravel_index(yshf.argmax(), yshf.shape)
        if (sx < Nw/2):
            sx = sx
        else:
            sx = sx - Nw
        if (sy < Nw/2):
            sy = sy
        else:
            sy = sy - Nw
        zsp = self.zerosuppression(sx, sy)
        temp = np.sqrt(sx ** 2 + sy ** 2) / (2 * self.radius)
        self.eh = np.append(self.eh, temp)
        self.otf_1_1 = self.otf_1_1 * zsp
        self.imgf_1_1 = fft2(self.separr[2] * ysh[1])

    def shift2(self, angle, spacing):
        ''' shift data in freq space by multiplication in real space '''
        Nw = self.nx
        dx = self.dx
        kx = dx * np.cos(angle) / spacing
        ky = dx * np.sin(angle) / spacing
        # shift matrix
        ysh = np.zeros((2, Nw, Nw), dtype=np.complex64)
        ysh[0, :, :] = np.exp(2j * np.pi * (kx * self.xv + ky * self.yv))
        ysh[1, :, :] = np.exp(2j * np.pi * (-kx * self.xv - ky * self.yv))
        # 1st order positive
        self.otf_2_0 = fft2(self.psf * ysh[0])
        yshf = np.abs(fft2(ysh[0]))
        sx, sy = np.unravel_index(yshf.argmax(), yshf.shape)
        if (sx < Nw/2):
            sx = sx
        else:
            sx = sx - Nw
        if (sy < Nw/2):
            sy = sy
        else:
            sy = sy - Nw
        temp = np.sqrt(sx ** 2 + sy ** 2) / (2 * self.radius)
        self.eh = np.append(self.eh, temp)
        zsp = self.zerosuppression(sx, sy)
        self.otf_2_0 = self.otf_2_0 * zsp
        self.imgf_2_0 = fft2(self.separr[3] * ysh[0])
        # 1st order negative
        self.otf_2_1 = fft2(self.psf * ysh[1])
        yshf = np.abs(fft2(ysh[1]))
        sx, sy = np.unravel_index(yshf.argmax(), yshf.shape)
        if (sx < Nw/2):
            sx = sx
        else:
            sx = sx - Nw
        if (sy < Nw/2):
            sy = sy
        else:
            sy = sy - Nw
        zsp = self.zerosuppression(sx, sy)
        temp = np.sqrt(sx ** 2 + sy ** 2) / (2 * self.radius)
        self.eh = np.append(self.eh, temp)
        self.otf_2_1 = self.otf_2_1 * zsp
        self.imgf_2_1 = fft2(self.separr[4] * ysh[1])

    def getoverlap1(self, angle, spacing, order, verbose=False):
        ''' shift 2nd order data '''
        dx = self.dx
        Nw = self.nx
        kx = dx * np.cos(angle) / spacing
        ky = dx * np.sin(angle) / spacing

        ysh = np.exp(2j * np.pi * (kx * self.xv + ky * self.yv))
        otf = fft2(self.psf * ysh)
        yshf = np.abs(fft2(ysh))
        sx, sy = np.unravel_index(yshf.argmax(), yshf.shape)
        if (sx < Nw/2):
            sx = sx
        else:
            sx = sx - Nw
        if (sy < Nw/2):
            sy = sy
        else:
            sy = sy - Nw
        zsp = self.zerosuppression(sx, sy)
        otf = otf * zsp
        imgf = fft2(self.separr[order] * ysh)
        cutoff = self.cutoff
        imgf0 = self.imgf0
        otf0 = self.otf0
        wimgf0 = otf * imgf0
        wimgf1 = otf0 * imgf
        msk = abs(otf0 * otf) > cutoff
        a = np.sum(msk * wimgf1 * wimgf0.conj()) / np.sum(msk * wimgf0 * wimgf0.conj())
        mag = np.abs(a)
        phase = np.angle(a)
        if verbose:
            t = (msk * wimgf1 * wimgf0.conj()) / (msk * wimgf0 * wimgf0.conj())
            t[np.isnan(t)] = 0.0
            figure()
            imshow(np.abs(np.fft.fftshift(t)), interpolation='nearest', vmin=0.0, vmax=2.0)
            colorbar()
            figure()
            imshow(np.angle(np.fft.fftshift(t)), interpolation='nearest')
            colorbar()
        return mag, phase

    def getoverlap2(self, angle, spacing, order, verbose=False):
        ''' shift 2nd order data '''
        dx = self.dx
        Nw = self.nx
        kx = dx * np.cos(angle) / spacing
        ky = dx * np.sin(angle) / spacing

        ysh = np.exp(2j * np.pi * (kx * self.xv + ky * self.yv))
        otf = fft2(self.psf * ysh)
        yshf = np.abs(fft2(ysh))
        sx, sy = np.unravel_index(yshf.argmax(), yshf.shape)
        if (sx < Nw/2):
            sx = sx
        else:
            sx = sx - Nw
        if (sy < Nw/2):
            sy = sy
        else:
            sy = sy - Nw
        zsp = self.zerosuppression(sx, sy)
        otf = otf * zsp
        imgf = fft2(self.separr[order] * ysh)
        cutoff = self.cutoff
        imgf0 = self.imgf_1_0
        otf0 = self.otf_1_0
        wimgf0 = otf * imgf0
        wimgf1 = otf0 * imgf
        msk = abs(otf0 * otf) > cutoff
        a = np.sum(msk * wimgf1 * wimgf0.conj()) / np.sum(msk * wimgf0 * wimgf0.conj())
        mag = np.abs(a)
        phase = np.angle(a)
        if verbose:
            t = (msk * wimgf1 * wimgf0.conj()) / (msk * wimgf0 * wimgf0.conj())
            t[np.isnan(t)] = 0.0
            figure()
            imshow(np.abs(np.fft.fftshift(t)), interpolation='nearest', vmin=0.0, vmax=2.0)
            colorbar()
            figure()
            imshow(np.angle(np.fft.fftshift(t)), interpolation='nearest')
            colorbar()
        return mag, phase
    
    def getoverlap_2(self, angle, spacing, order, verbose=False):
        ''' shift 2nd order data '''
        dx = self.dx
        Nw = self.nx
        kx = dx * np.cos(angle) / spacing
        ky = dx * np.sin(angle) / spacing

        ysh = np.exp(2j * np.pi * (kx * self.xv + ky * self.yv))
        otf = fft2(self.psf * ysh)
        yshf = np.abs(fft2(ysh))
        sx, sy = np.unravel_index(yshf.argmax(), yshf.shape)
        if (sx < Nw/2):
            sx = sx
        else:
            sx = sx - Nw
        if (sy < Nw/2):
            sy = sy
        else:
            sy = sy - Nw
        zsp = self.zerosuppression(sx, sy)
        otf = otf * zsp
        imgf = fft2(self.separr[order] * ysh)
        cutoff = self.cutoff
        imgf0 = self.imgf0
        otf0 = self.otf0
        wimgf0 = otf * imgf0
        wimgf1 = otf0 * imgf
        msk = abs(otf0 * otf) > cutoff
        a = np.sum(msk * wimgf1 * wimgf0.conj()) / np.sum(msk * wimgf0 * wimgf0.conj())
        mag = np.abs(a)
        phase = np.angle(a)
        if verbose:
            t = (msk * wimgf1 * wimgf0.conj()) / (msk * wimgf0 * wimgf0.conj())
            t[np.isnan(t)] = 0.0
            figure()
            imshow(np.abs(np.fft.fftshift(t)), interpolation='nearest', vmin=0.0, vmax=2.0)
            colorbar()
            figure()
            imshow(np.angle(np.fft.fftshift(t)), interpolation='nearest')
            colorbar()
        return mag, phase

    def mapoverlap1(self, angle, spacing, order, nps=10, r_ang=0.02, r_sp=0.008):
        d_ang = 2 * r_ang / nps
        d_sp = 2 * r_sp / nps
        ang_iter = np.arange(-r_ang, r_ang + d_ang / 2, d_ang) + angle
        sp_iter = np.arange(-r_sp, r_sp + d_sp / 2, d_sp) + spacing
        magarr = np.zeros((nps + 1, nps + 1))
        pharr = np.zeros((nps + 1, nps + 1))
        for m, ang in enumerate(ang_iter):
            for n, sp in enumerate(sp_iter):
                print(m, n)
                mag, phase = self.getoverlap1(ang, sp, order)
                if np.isnan(mag):
                    magarr[m, n] = 0.0
                else:
                    magarr[m, n] = mag
                    pharr[m, n] = phase
        # figure()
        # subplot(211)
        # imshow(magarr, vmin = magarr.min(), vmax = magarr.max(), extent = [sp_iter.min(), sp_iter.max(), ang_iter.max(), ang_iter.min()], interpolation=None)
        # subplot(212)
        # imshow(pharr, interpolation='nearest')
        # get maximum
        k, l = np.where(magarr == magarr.max())
        angmax = k[0] * d_ang - r_ang + angle
        spmax = l[0] * d_sp - r_sp + spacing
        return (angmax, spmax, magarr.max())

    def mapoverlap2(self, angle, spacing, order, nps=10, r_ang=0.02, r_sp=0.008):
        d_ang = 2 * r_ang / nps
        d_sp = 2 * r_sp / nps
        ang_iter = np.arange(-r_ang, r_ang + d_ang / 2, d_ang) + angle
        sp_iter = np.arange(-r_sp, r_sp + d_sp / 2, d_sp) + spacing
        magarr = np.zeros((nps + 1, nps + 1))
        pharr = np.zeros((nps + 1, nps + 1))
        for m, ang in enumerate(ang_iter):
            for n, sp in enumerate(sp_iter):
                print(m, n)
                mag, phase = self.getoverlap2(ang, sp, order)
                if np.isnan(mag):
                    magarr[m, n] = 0.0
                else:
                    magarr[m, n] = mag
                    pharr[m, n] = phase
        # figure()
        # subplot(211)
        # imshow(magarr, vmin = magarr.min(), vmax = magarr.max(), extent = [sp_iter.min(), sp_iter.max(), ang_iter.max(), ang_iter.min()], interpolation=None)
        # subplot(212)
        # imshow(pharr, interpolation='nearest')
        # get maximum
        k, l = np.where(magarr == magarr.max())
        angmax = k[0] * d_ang - r_ang + angle
        spmax = l[0] * d_sp - r_sp + spacing
        return (angmax, spmax, magarr.max())
    
    def mapoverlap_2(self, angle, spacing, order, nps=10, r_ang=0.02, r_sp=0.008):
        d_ang = 2 * r_ang / nps
        d_sp = 2 * r_sp / nps
        ang_iter = np.arange(-r_ang, r_ang + d_ang / 2, d_ang) + angle
        sp_iter = np.arange(-r_sp, r_sp + d_sp / 2, d_sp) + spacing
        magarr = np.zeros((nps + 1, nps + 1))
        pharr = np.zeros((nps + 1, nps + 1))
        for m, ang in enumerate(ang_iter):
            for n, sp in enumerate(sp_iter):
                print(m, n)
                mag, phase = self.getoverlap_2(ang, sp, order)
                if np.isnan(mag):
                    magarr[m, n] = 0.0
                else:
                    magarr[m, n] = mag
                    pharr[m, n] = phase
        # figure()
        # subplot(211)
        # imshow(magarr, vmin = magarr.min(), vmax = magarr.max(), extent = [sp_iter.min(), sp_iter.max(), ang_iter.max(), ang_iter.min()], interpolation=None)
        # subplot(212)
        # imshow(pharr, interpolation=None)
        # get maximum
        k, l = np.where(magarr == magarr.max())
        angmax = k[0] * d_ang - r_ang + angle
        spmax = l[0] * d_sp - r_sp + spacing
        return (angmax, spmax, magarr.max())

    def recon1(self, i, ang, spacing, phase, mag, zero_order=True):
        # construct 1 angle
        nx = self.nx
        ny = self.ny
        mu = self.mu

        imgf = np.zeros((nx, ny), dtype=np.complex64)
        otf = np.zeros((nx, ny), dtype=np.complex64)

        self.Snum = np.zeros((nx, ny), dtype=np.complex64)
        self.Sden = np.zeros((nx, ny), dtype=np.complex64)
        self.Sden += mu ** 2

        self.separate(i)
        self.shift0()
        self.shift1(ang[0], spacing[0])
        self.shift2(ang[1], spacing[1])
        ph1 = mag[0] * np.exp(1j * phase[0])
        ph2 = mag[1] * np.exp(1j * phase[1])
        if zero_order==True:
            # 0th order
            imgf = self.imgf0
            otf = self.otf0
            self.Snum += otf.conj() * imgf
            self.Sden += abs(otf) ** 2
        # +1st order
        imgf = self.imgf_1_0
        otf = self.otf_1_0
        self.Snum += ph1 * otf.conj() * imgf
        self.Sden += abs(otf) ** 2
        # -1st order
        imgf = self.imgf_1_1
        otf = self.otf_1_1
        self.Snum += ph1.conj() * otf.conj() * imgf
        self.Sden += abs(otf) ** 2
        # +2nd order
        imgf = self.imgf_2_0
        otf = self.otf_2_0
        self.Snum += ph2 * otf.conj() * imgf
        self.Sden += abs(otf) ** 2
        # -2nd order
        imgf = self.imgf_2_1
        otf = self.otf_2_1
        self.Snum += ph2.conj() * otf.conj() * imgf
        self.Sden += abs(otf) ** 2
        # # finish
        # self.S = self.Snum/self.Sden
        # self.finalimage = np.fft.fftshift(ifft2(S))

        ss = 4. * np.max(self.eh)
        A = self.apod(ss)
        self.S = self.Snum / self.Sden * A
        self.finalimage = fftshift(ifft2(self.S))
        tf.imsave('final_image.tif', self.finalimage.real.astype(np.float32), photometric='minisblack')
        tf.imsave('effective_OTF.tif', np.abs(fftshift(self.S)).astype(np.float32), photometric='minisblack')

    def recon(self, n, ang, spacing, phase, mag, path1, path2, zero_order=True):
        # construct n angles
        nx = self.nx
        ny = self.ny
        mu = self.mu

        imgf = np.zeros((nx, ny), dtype=np.complex64)
        otf = np.zeros((nx, ny), dtype=np.complex64)

        self.Snum = np.zeros((nx, ny), dtype=np.complex64)
        self.Sden = np.zeros((nx, ny), dtype=np.complex64)
        self.Sden += mu ** 2
        
        if zero_order==True:
            # 0th order
            imgf = self.imgf0
            otf = self.otf0
            self.Snum += otf.conj() * imgf
            self.Sden += abs(otf) ** 2
        for i in range(n):
            self.separate(i)
            self.shift0()
            self.shift1(ang[2*i], spacing[2*i])
            self.shift2(ang[2*i+1], spacing[2*i+1])
            ph1 = mag[2*i] * np.exp(1j * phase[2*i])
            ph2 = mag[2*i+1] * np.exp(1j * phase[2*i+1])
            # +1st order
            imgf = self.imgf_1_0
            otf = self.otf_1_0
            self.Snum += ph1 * otf.conj() * imgf
            self.Sden += abs(otf) ** 2
            # -1st order
            imgf = self.imgf_1_1
            otf = self.otf_1_1
            self.Snum += ph1.conj() * otf.conj() * imgf
            self.Sden += abs(otf) ** 2
            # +2nd order
            imgf = self.imgf_2_0
            otf = self.otf_2_0
            self.Snum += ph2 * otf.conj() * imgf
            self.Sden += abs(otf) ** 2
            # -2nd order
            imgf = self.imgf_2_1
            otf = self.otf_2_1
            self.Snum += ph2.conj() * otf.conj() * imgf
            self.Sden += abs(otf) ** 2
            # # finish
            # self.S = self.Snum/self.Sden
            # self.finalimage = np.fft.fftshift(ifft2(S))
        ss = 5. * np.max(self.eh)
        A = self.apod(ss)
        self.S = self.Snum / self.Sden * A
        # self.finalimage = fftshift(ifft2(self.S))
        self.finalimage = np.abs(fftshift(ifft2(self.S)))
        if self.farme_order == 1:
            if zero_order == True:
                tf.imsave(self.path + '/' + path1 + '/' + 'final_image_withzero.tif', self.finalimage.real.astype(np.float32), photometric='minisblack')
                tf.imsave(self.path + '/' + path1 + '/' + 'effective_OTF_withzero.tif', np.abs(fftshift(self.S)).astype(np.float32), photometric='minisblack')
            elif zero_order == False:
                tf.imsave(self.path + '/' + path1 + '/' +'final_image_nozero.tif', self.finalimage.real.astype(np.float32), photometric='minisblack')
                tf.imsave(self.path + '/' + path1 + '/' + 'effective_OTF_nozero.tif', np.abs(fftshift(self.S)).astype(np.float32), photometric='minisblack')
        elif self.farme_order == 2:
            if zero_order == True:
                tf.imsave(self.path + '/' + path2 + '/' +'final_image_withzero.tif', self.finalimage.real.astype(np.float32), photometric='minisblack')
                tf.imsave(self.path + '/' + path2 + '/' + 'effective_OTF_withzero.tif', np.abs(fftshift(self.S)).astype(np.float32), photometric='minisblack')
            elif zero_order == False:
                tf.imsave(self.path + '/' + path2 + '/' +'final_image_nozero.tif', self.finalimage.real.astype(np.float32), photometric='minisblack')
                tf.imsave(self.path + '/' + path2 + '/' + 'effective_OTF_nozero.tif', np.abs(fftshift(self.S)).astype(np.float32), photometric='minisblack')


    def recon_sim(self, n, ang, spacing, phase, mag, pathsim1, pathsim2, zero_order=True):
        # construct n angles
        nx = self.nx
        ny = self.ny
        mu = self.mu

        imgf = np.zeros((nx, ny), dtype=np.complex64)
        otf = np.zeros((nx, ny), dtype=np.complex64)

        self.Snum = np.zeros((nx, ny), dtype=np.complex64)
        self.Sden = np.zeros((nx, ny), dtype=np.complex64)
        self.Sden += mu ** 2
        
        if zero_order==True:
            # 0th order
            imgf = self.imgf0
            otf = self.otf0
            self.Snum += otf.conj() * imgf
            self.Sden += abs(otf) ** 2
        for i in range(n):
            self.separate(i)
            self.shift0()
            self.shift1(ang[2*i], spacing[2*i])
            ph1 = mag[2*i] * np.exp(1j * phase[2*i])
            # +1st order
            imgf = self.imgf_1_0
            otf = self.otf_1_0
            self.Snum += ph1 * otf.conj() * imgf
            self.Sden += abs(otf) ** 2
            # -1st order
            imgf = self.imgf_1_1
            otf = self.otf_1_1
            self.Snum += ph1.conj() * otf.conj() * imgf
            self.Sden += abs(otf) ** 2
            # # finish
            # self.S = self.Snum/self.Sden
            # self.finalimage = np.fft.fftshift(ifft2(S))
        # ss = 5. * np.max(self.eh)
        # A = self.apod(ss)
        self.S = self.Snum / self.Sden #* A
        # self.finalimage = fftshift(ifft2(self.S))
        self.finalimage = np.abs(fftshift(ifft2(self.S)))
        
        if self.farme_order == 1:
            tf.imsave(self.path + '/' + pathsim1 + '/' + 'final_image_withzero_sim.tif', self.finalimage.real.astype(np.float32), photometric='minisblack')
            tf.imsave(self.path + '/' + pathsim1 + '/' + 'effective_OTF_withzero_sim.tif', np.abs(fftshift(self.S)).astype(np.float32), photometric='minisblack')    
        elif self.farme_order == 2:
            tf.imsave(self.path + '/' + pathsim2 + '/' +'final_image_withzero_sim.tif', self.finalimage.real.astype(np.float32), photometric='minisblack')
            tf.imsave(self.path + '/' + pathsim2 + '/' + 'effective_OTF_withzero_sim.tif', np.abs(fftshift(self.S)).astype(np.float32), photometric='minisblack')
    



    def interp(self, arr):
        nx, ny = arr.shape
        px = int( nx ) 
        py = int( ny ) 
        arrf = fft2(arr)
        arro = np.pad(np.fft.fftshift(arrf), ((px, px), (py, py)), 'constant', constant_values=(0))
        out = ifft2(np.fft.fftshift(arro))
        return out


    
    def winfwindow(self, eta):
          nx = self.nx
          wxy = signal.tukey(nx, alpha=eta, sym=True)
          wx = np.tile(wxy, (nx, 1))
          wy = wx.swapaxes(0, 1) 
          # wy=wx.T
          w = wx * wy
          return w

    def zerosuppression(self, sx, sy):
        x = self.xv
        y = self.yv
        g = 1 - self.strength * np.exp(-((x - sx) ** 2. + (y - sy) ** 2.) / (2. * self.sigma ** 2.))
        g[g < 0.5] = 0.0
        g[g >= 0.5] = 1.0
        return g

    def apod(self, mp):
        r = (2 * mp) * self.radius
        ap = 1 - np.sqrt(self.xv ** 2 + self.yv ** 2) / r
        ap[ap < 0.] = 0.
        return ap
    
    

    def check_components(self, A = 1):
        ''' plot components in Fourier space '''
        fig, axs = subplots(1, 5)
        for m in range(5):
            temp = np.abs(fftshift(fft2(self.separr[m])))**(0.5)
            zoomInx = int(self.nx/2-self.nx/6)
            zoomIny = int(self.ny/2+self.ny/6)
            # axs[m].imshow(temp[zoomInx:zoomIny,zoomInx:zoomIny])
            axs[m].imshow(temp[zoomInx:zoomIny,zoomInx:zoomIny])
            # axs[m].imshow(temp)
            axs[m].axis('off')
        fig.savefig(self.path + '/' +'components_Angle' + str(A) +'.jpg',bbox_inches='tight',dpi=1500)
        return True
        
        