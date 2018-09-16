####
import glob, os
import sys
from PIL import Image
import numpy as np
import scipy.io
from skimage import io, transform
from scipy import ndimage
from scipy import signal
import math

def main():


    ## choose sizes divisible many times by 2 for conv layers
    h = 96
    w = 152
    size = (w,h)
    pix = h*w
    y = []
    stride = 1
    ary = np.array(np.zeros((1,w,h,3)))
    #print(ary.shape)
    dir = "IMG_DIR/HOTDOG/"
    cnt=0
    for infile in glob.glob(dir + "*.jpg"):
        file, ext = os.path.splitext(infile)
        img = io.imread(file+ext)  ## use scikit-image to read in 3D format
        #print(img[50,70,:])
        ## resize and mitigate effects of skimage resize on RGB
        img = transform.resize(img,(w,h,3),mode='constant',anti_aliasing=True)
        img = 255 * img
        img = img.astype(np.uint8) ## skimage is a numpy array
        n = Image.fromarray(img)
        n.show()

        #img = np.expand_dims(img,0) ## new axis for stacking
        #print(img.shape)
        #print(img)
        #ary = np.vstack((ary,img))
        ##print(ary.shape)
        ''' Convolution Test'''
        #k = np.array([[1,1,1],[1,1,1],[1,1,1]])  # 3x3 filter
        #k = np.vstack((k,k))
        #print(k.shape)
        #k1 = np.array([[[1,1,1],[1,1,1],[1,1,1]]]) ##1x3x3 filter
        #r = ndimage.filters.convolve(img[:,:,:],k,mode='constant',cval=0.0)
        #g = ndimage.filters.convolve(img[:,:,1],k,mode='constant',cval=0.0)
        #b = ndimage.filters.convolve(img[:,:,2],k,mode='constant',cval=0.0)
        ##############
        '''First activation map
        k1 is 3x3x3 filter
        k2 is another 3x3x3 filter
        V structure: r by c is number of 'blocks' wide by 'blocks' high
        From http://cs231n.github.io/convolutional-networks/#conv
        '''
        r = 0
        c = 0
        #P=(F−1)/2 pad the image
        print(img.shape)
        img = np.append(img,np.zeros((3,h,3)),axis=0)
        img = np.append(img,np.zeros((w+3,3,3)),axis=1)
        print(img.shape)

        k = np.array([[[1,1,1],[1,1,1],[1,1,1],[1,1,1],[1,1,1]]]) ##1x5x3
        k = np.vstack((k,k,k,k,k)) ##5x5x3 filter
        print(k.shape)
        ## w * h = r by c
        V = np.array(np.zeros(((math.ceil((w)/stride)),(math.ceil((h)/stride)))))
        print(V.shape)
        r_box = -1
        for r in range(0,w,stride):
            r_box = r_box + 1
            c_box = -1
            for c in range(0,h,stride):
                c_box = c_box + 1
                #print("r, c: ",r,c)
                #print("row block, c block: ", r_box,c_box)
                V[r_box,c_box] = np.sum(img[r:r+5,c:c+5,:] * k)
        print(V)
        print(V.shape)
        ## (r by c by d) >> 73 by 50 by 3

            #V[0,0,0] = np.sum(img[:5,:5,:] * k1)
            #V[1,0,0] = np.sum(img[2:7,:5,:] *k1)
            #V[2,0,0] = np.sum(img[4:9,:5,:] *k1)
            #.....
            #V[0,1,0] = np.sum(img[:5,2:7,:] * k1)
        '''Second activation map'''


        ###########################
        '''
        img = ndimage.filters.convolve(img,k,mode='reflect',cval=1.0)
        #img = img*255
        '''
        #print(n.shape)
        #n = Image.fromarray(n)
        #n.show()
        #x = np.vstack((r,g,b))
        #print(n.shape)
        #print(n.shape)
        #img = Image.open(file+ext)
        #img = img.resize(size)
        #n = list(img.getdata())
        #n = np.asarray(n)
        #print(n.shape)
        #n = np.reshape(n,[1,15000,3])
        #ary2 = np.atleast_3d(ary2)
        #print(ary.shape)
        cnt = cnt+1
        if cnt == 2: break

    '''
    ary = np.delete(ary,0,0)
    #print(ary.shape)

    ## ary now m by h by w by 3
    ##Convolution Step
    ## example kernel
    #k = np.array([[[-1,-1,-1],[-1,8,-1],[-1,-1,-1]]])
    k = np.array([[[1,1,1],[1,1,1],[1,1,1]]])
    k = np.vstack((k,k,k))
    #print(np.array(np.zeros((3,3,3))))
    #print(k)
    #print(k.shape)
    #print(img[:,:,0,0])
    #img = ndimage.filters.convolve(img[0,:,:,:],k,mode='constant',cval=0.0)
    #print(img.shape)
    img = Image.fromarray(img)
    #img.show()
    '''

if (__name__=="__main__"):
    main()
