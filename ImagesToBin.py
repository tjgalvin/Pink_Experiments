import numpy
import warnings
import struct
import math
import os
import scipy.ndimage
import pyfits
import shutil


def calculateBackgroundStatistics(image, centerSize):
    size = image.shape
    empty = numpy.isnan(image)
    center = (slice(size[0]/2-size[0]/centerSize, size[0]/2+size[0]/centerSize), slice(size[1]/2-size[1]/centerSize, size[1]/2+size[1]/centerSize))
    mask = numpy.ones(image.shape, numpy.bool)
    mask[center] = 0
    mask[empty] = 0
    #print (numpy.sum(mask)/128.0/128.0, numpy.count_nonzero(~numpy.isnan(image)))
    backgroundMean = numpy.mean(image[mask])
    backgroundStd = numpy.std(image[mask])
    backgoundMin = numpy.min(image[mask])
    return backgroundMean, backgroundStd, backgoundMin

def preprocessImageF3(image, scale):
    empty = numpy.isnan(image)
    mu, sigma, _minimum = calculateBackgroundStatistics(image, 5)
    image[empty] = numpy.random.normal(loc=mu, scale=sigma, size=image.shape)[empty] # replace missing values with noise
#    mu, sigma, minimum = calculateBackgroundStatistics(image, 5)
    image = image - mu
    image = numpy.clip(image, 1.0*sigma, 1e10) # clip background
    image = image - 1.0*sigma
    image = image / numpy.max(image)
    result = image * scale
    return result


#def preprocessImageF1(image,scale):
#    empty = numpy.isnan(image)
#    image[empty] = numpy.random.normal(loc=numpy.mean(image[~empty]), scale=numpy.std(image[~empty]), size=image.shape)[empty]
#    image = image - numpy.min(image)
#    image = numpy.clip(image, 2.0*numpy.std(image), 1e10)
#    image = (image - numpy.min(image)) / (numpy.max(image) - numpy.min(image))
#    result = image * scale
#    return result

#def preprocessImageF2(image,scale):
#    empty = numpy.isnan(image)
#    image[empty] = numpy.random.normal(loc=numpy.mean(image[~empty]), scale=numpy.std(image[~empty]), size=image.shape)[empty]
#    image = image - numpy.min(image)
#    image = numpy.clip(image, 3.0*numpy.std(image), 1e10)
#    image = (image - numpy.min(image)) / (numpy.max(image) - numpy.min(image))
#    result = image * scale
#    return result

def preprocessImageW1(image,scale):
    empty = numpy.isnan(image)
    image[empty] = numpy.random.normal(loc=numpy.mean(image[~empty]), scale=numpy.std(image[~empty]), size=image.shape)[empty]
    image = image - numpy.min(image)
    image = (image - numpy.min(image)) / (numpy.max(image) - numpy.min(image))
    result = image * scale
    return result

def preprocessImageW2(image,scale):
    empty = numpy.isnan(image)
    image[empty] = numpy.random.normal(loc=numpy.mean(image[~empty]), scale=numpy.std(image[~empty]), size=image.shape)[empty]
    image = numpy.clip(image, 0.0001, 1e10)
    #image = image - numpy.min(image) + 0.0001 # equals -5 for log10 :-)
    image = numpy.log10(image)
    #image = numpy.clip(image, -5, 1e10) # clip after the log !
    image = (image - numpy.min(image)) / (numpy.max(image) - numpy.min(image))
    result = image * scale
    return result

numpy.random.seed(243535)

first0 = numpy.load('/hits/basement/ain/UKIDSS_FIRST/tim/data/first_arr_0.npy')
first1 = numpy.load('/hits/basement/ain/UKIDSS_FIRST/tim/data/first_arr_1.npy')
first2 = numpy.load('/hits/basement/ain/UKIDSS_FIRST/tim/data/first_arr_2.npy')
first3 = numpy.load('/hits/basement/ain/UKIDSS_FIRST/tim/data/first_arr_3.npy')

wise0 = numpy.load('/hits/basement/ain/UKIDSS_FIRST/tim/data/wise_arr_0.npy')
wise1 = numpy.load('/hits/basement/ain/UKIDSS_FIRST/tim/data/wise_arr_1.npy')
wise2 = numpy.load('/hits/basement/ain/UKIDSS_FIRST/tim/data/wise_arr_2.npy')
wise3 = numpy.load('/hits/basement/ain/UKIDSS_FIRST/tim/data/wise_arr_3.npy')

firstData = numpy.concatenate((first0,first1,first2,first3),axis=0)
wiseData = numpy.concatenate((wise0,wise1,wise2,wise3),axis=0)

imageCount = numpy.shape(firstData)[0]
height, width = (167,167)
warnings.filterwarnings('ignore')

output = open("/hits/basement/ain/UKIDSS_FIRST/tim/bin/F3W2_95_5.bin", 'wb') # output file opened for byte writing
output.write(struct.pack('i', imageCount)) # number of objects
output.write(struct.pack('i', 2)) # number of channels
output.write(struct.pack('i', width)) # width
output.write(struct.pack('i', height)) # height

print('start')

for j in range(imageCount):
    if j%10000==0:
        print(j)
    try:
        imageFirst = firstData[j]
        imageWise = wiseData[j]
        imageFirst = preprocessImageF3(imageFirst,0.95)
        imageWise = preprocessImageW2(imageWise,0.05)
        imageFirst.astype('f').tofile(output)
        imageWise.astype('f').tofile(output)
    except:
        print(j)

output.seek(0)
output.write(struct.pack('i',j)) # number of objects
output.close()

