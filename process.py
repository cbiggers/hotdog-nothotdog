'''
*CDKB 20180607
Python script as part of ML hotdog/nothotdog test

*TODO
    output 1xn row of features
	Then loop over directory and do this for all images to output a .txt file of features
'''

### Modules / Packages ###
import glob, os
import sys
from PIL import Image
import numpy as np
import scipy.io
#sys.path.append('../')
#import pickle


### source images stored in image directory
## Import initial image
'''
img = Image.open("IMG_DIR/HOTDOG/1.jpg")
img.show()
img = img.convert("L") ## mode L, black and white mode
print("Box: ",img.getbbox())
pix = list(img.getdata())
print(pix)  ## list of all pixel values grayscale
print("Length: ",len(pix))
## need one normalized wxh size
img = img.resize((400,200))
print(len(list(img.getdata())))
img.show()
'''



def main():
    '''
    ## converts color images into 1 by ~10000 lists of greyscale pixels
    '''
    h = 100
    w = 150
    features = h*w
    size = (w,h)

    pos_path = "IMG_DIR/HOTDOG/"
    neg_path = "IMG_DIR/NOTHOTDOG/"
    path = [] ## empty list
    path.append(pos_path)
    path.append(neg_path)

    '''Image counts
    Useful for splitting into train/test/cv sets
    and for array creation'''
    m = 0
    for i in path:
        for infile in glob.glob(i + "*.jpg"):
             m += 1

    ## initialize arrays. 1 big array and 3 subarrays
    ary = {}
    ary[0] = np.array(np.zeros((1,features+1)))
    ary[1] = np.array(np.zeros((1,features+1)))
    ary[2] = np.array(np.zeros((1,features+1)))
    ary[3] = np.array(np.zeros((1,features+1)))
    #print(imgary.dtype)
    #print("Features: ",features)
    #print("Elements in Array: ",imgary.size)
    #print("Shape of Array (r x c): ",imgary.shape)
    #imgary[0,10000] = 7  ## assigning elements
    ## print(imgary[0,10000])  ## last element, since numpy lists 0-indexed
    #np.ones((1,features+1)

    '''Process Images'''
    cnt = [0,0]
    for i in path:
        if i == pos_path: y = 1
        if i == neg_path: y = 0
        for infile in glob.glob(i + "*.jpg"):
            file, ext = os.path.splitext(infile)## file is name, ext is the .jpg
            img = Image.open(file+ext)
            img = img.convert("L")
            #img = img.convert("I")
            #print(list(img.getdata()))
            #w,h = img.size #img = img.rotate(90)
            img = img.resize(size) ## convert to standard size
            #img.show()
            n = list(img.getdata())
            n.append(y)
            ## generate randomish 60/20/20 spread of examples
            ## when assigning each example m to dataset
            ## 1=train, 2=cross-valid, 3=test
            ary[0] = np.vstack((ary[0],n))
            #choice = np.random.choice([3,2,1],1,1,[.2,.2,.6])[0]
            #ary[choice] = np.vstack((ary[choice],n))

            #print(ary[choice])
            #print(ary[1].flatten())
            #print(ary[choice].shape)
            if y==1: cnt[1] +=1
            if y==0: cnt[0] +=1
            print(cnt)


    print("Positive Images: ",cnt[1])
    print("Negative Images: ",cnt[0])
    total = cnt[1] + cnt[0]
    print("Total images processed: ",total)

    ## delete the first row, was placeholder of 0s to construct array
    ary[0] = np.delete(ary[0],0,0)
    ary[0] = np.random.permutation(ary[0])
    valid_pos = 0
    valid_neg = 0
    test_pos = 0
    test_neg = 0
    pos_slice = int(cnt[1]/5)
    neg_slice = int(cnt[0]/5)

    ## assign each row to train/valid/test sets

    for i in range(0,total):
        n = ary[0][i]
        ## this m is y==1
        if n[features] == 1:
            if valid_pos <= pos_slice:
                ary[2] = np.vstack((ary[2],n))
                valid_pos = valid_pos+1
            elif test_pos <= pos_slice:
                ary[3] = np.vstack((ary[3],n))
                test_pos = test_pos+1
            else: ary[1] = np.vstack((ary[1],n))
        ## this m is y==0
        elif n[features] == 0:
            if valid_neg <= neg_slice:
                ary[2] = np.vstack((ary[2],n))
                valid_neg = valid_neg+1
            elif test_neg <= neg_slice:
                ary[3] = np.vstack((ary[3],n))
                test_neg = test_neg+1
            else: ary[1] = np.vstack((ary[1],n))

    for i in (1,2,3):
        ary[i] = np.delete(ary[i],0,0)
        ary[i] = np.random.permutation(ary[i])

    print("Training dataset: ",ary[1].shape)
    print("Validation dataset: ",ary[2].shape)
    print("Testing dataset: ",ary[3].shape)

    scipy.io.savemat('data_train.mat', mdict={'data_train': ary[1]})
    scipy.io.savemat('data_valid.mat', mdict={'data_valid': ary[2]})
    scipy.io.savemat('data_test.mat', mdict={'data_test': ary[3]})


if (__name__=="__main__"):
    main()
