import matplotlib.pyplot as plt 
import numpy as np 
import pandas as pd 
import struct

def no_ticks(ax):
    '''Disable ticks
    '''
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)


class heatmap(object):
    '''Helper to interact with a heatmap output
    '''
    def __init__(self, path):
        '''Path to the heatmap to load
        '''
        self.path = path
        self.fd = None # File descriptor to access if needed
        self.header_info = None

    @property
    def f(self):
        """Helper to open a persistent filedescriptor. Will always start
        at beginning of file
        """
        if self.fd is None:
            self.fd = open(self.path, 'rb')
        
        self.fd.seek(0)

        return self.fd

    @property
    def file_head(self):
        '''Return the file header information from PINK
        '''
        # Get file handler seeked to zero
        f = self.f
    
        no_images, som_width, som_height, som_depth = struct.unpack('i' * 4, f.read(4*4))
        
        return (no_images, som_width, som_height, som_depth)


    @property
    def details(self):
        if self.header_info is None:
            self.header_info = self.file_head
        
        return self.header_info


    def _ed_to_prob(self, ed, stretch=10, *args, **kwargs):
        '''Function to conver the euclidean distance to a likelihood
        '''
        prob = 1. / ed**stretch
        prob = prob / prob.sum()
        
        return prob


    def _get_ed(self, index=0, *args, **kwargs):
        '''Get the Euclidean distance of the i't page
        '''
        # Get file handler seeked to zero
        f = self.f

        # no_images, som_width, som_height, som_depth = struct.unpack('i' * 4, f.read(4*4))
        no_images, som_width, som_height, som_depth = self.details

        size = som_width * som_height * som_depth
        image_width = som_width
        image_height = som_depth * som_height

        # Seek the image number here
        # f.seek(index * size * 4, 1)
        f.seek(index * size * 4 + 4*4, 0)
        array = np.array(struct.unpack('f' * size, f.read(size * 4)))
        data = np.ndarray([som_width, som_height, som_depth], 'float', array)
        data = np.swapaxes(data, 0, 2)
        data = np.reshape(data, (image_height, image_width))

        return data


    def ed(self, index=0, prob=False, *args, **kwargs):
        '''Get the slice of a heatmap that has been mapped
        '''
        arr = self._get_ed(index=index)
        if prob:
            arr = self._ed_to_prob(arr, *args, **kwargs)
        
        return arr


class image_binary(object):
    '''Helper to interact with a heatmap output
    '''
    def __init__(self, path):
        '''Path to the image binary to load
        '''
        self.path = path

    def get_image(self, index=0, channel=0):
        '''Return the index-th image that was dumped to the binary image file that
        is managed by this instance of Binary
        
        index - int
            The source image to return
        channel - int
            The channel number of the image to return
        '''
        with open(self.path, 'rb') as f:
            no_images, no_channels, width, height = struct.unpack('i' * 4, f.read(4 * 4))
            if index > no_images:
                return None
            if channel > no_channels:
                return None

            size = width * height
            f.seek((index*no_channels + channel) * size*4, 1)
            array = np.array(struct.unpack('f' * size, f.read(size*4)))
            data = np.ndarray([width,height], 'float', array)

            return data

    @property
    def file_head(self):
        '''Return the file header information from PINK
        '''
        with open(self.path, 'rb') as f:
            no_images, no_channels, width, height = struct.unpack('i' * 4, f.read(4*4))
            
            return (no_images, no_channels, width, height)

class som(object):
    '''Class to interact with a SOM object
    '''
    def __init__(self, path):
        self.path = path

    @property
    def file_head(self):
        with open(self.path, 'rb') as som:
            # Unpack the header information
            numberOfChannels, SOM_width, SOM_height, SOM_depth, neuron_width, neuron_height = struct.unpack('i' * 6, som.read(4*6))

            return (numberOfChannels, SOM_width, SOM_height, SOM_depth, neuron_width, neuron_height)

    def get_som(self, channel=0):
        '''Get out a SOM given a channel
        channel - int
              The channel number to extract
        '''
        with open(self.path, 'rb') as som:
            # Unpack the header information
            numberOfChannels, SOM_width, SOM_height, SOM_depth, neuron_width, neuron_height = struct.unpack('i' * 6, som.read(4*6))
            SOM_size = np.prod([SOM_width, SOM_height, SOM_depth])

            # Check to ensure that the request channel exists. Remeber we are comparing
            # the index
            if channel > numberOfChannels - 1:
                # print(f'Channel {channel} larger than {numberOfChannels}... Returning...')
                return None

            dataSize = numberOfChannels * SOM_size * neuron_width * neuron_height
            array = np.array(struct.unpack('f' * dataSize, som.read(dataSize * 4)))

            image_width = SOM_width * neuron_width
            image_height = SOM_depth * SOM_height * neuron_height
            data = np.ndarray([SOM_width, SOM_height, SOM_depth, numberOfChannels, neuron_width, neuron_height], 'float', array)
            data = np.swapaxes(data, 0, 5) # neuron_height, SOM_height, SOM_depth, numberOfChannels, neuron_width, SOM_width
            data = np.swapaxes(data, 0, 2) # SOM_depth, SOM_height, neuron_height, numberOfChannels, neuron_width, SOM_width
            data = np.swapaxes(data, 4, 5) # SOM_depth, SOM_height, neuron_height, numberOfChannels, SOM_width, neuron_width
            data = np.reshape(data, (image_height, numberOfChannels, image_width))

            if channel < 0 or channel is None:
                # Leave data as is and return
                pass
            else:
                data = data[:,channel,:]

            return data

    def get_neuron(self, x, y, channel=0):
        '''Extract a neuron out of the SOM given the (x,y) and optional channel number
        
        x/y - int
             The integer positions of the neuron to slice out
        channel - int
             The channel of the SOM to return
        '''
        som_spec = self.file_head
        data = self.get_som(channel=channel)
    
        d = data[x*som_spec[5]:(x+1)*som_spec[5], 
                 y*som_spec[5]:(y+1)*som_spec[5]]

        return d
