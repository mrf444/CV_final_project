import numpy as np 
import matplotlib.pyplot as plt 
import time
import scipy
from PIL import Image
import math
import scipy.misc
from scipy.ndimage.interpolation import zoom
from scipy.stats import norm
from scipy.stats import multivariate_normal
import pywt
from scipy import signal
from scipy.ndimage import gaussian_filter
import scipy.stats as st
import scipy.ndimage as ndimage #image processing library
import itertools

def rgb2gray(rgb):
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray

def DxDy_filter(dim):
    mask = np.zeros((dim,dim))
    ref = int(math.ceil(dim/9))
    i1 = ref
    i2 = int(dim/3+(dim/3-3)/2+1)
    i3 = i2+1
    i4 = dim - ref
    
    mask[i1:i2, i1:i2] = 1
    mask[i3:i4, i3:i4] = 1
    mask[i1:i2, i3:i4] = -1
    mask[i3:i4, i1:i2] = -1
    
    return np.array(mask)

def DyDy_filter(dim):
    mask = np.zeros((dim,dim))
    ref = int(dim/3)
    ref2 = math.floor(dim/2)
    i1 = ref
    i2 = 2*ref
    i3 = math.ceil(ref2/2)
    i4 = dim - math.ceil(ref2/2)
    
    mask[0:i1, i3:i4] = 1
    mask[i1:i2, i3:i4] = -2
    mask[i2:dim+1, i3:i4] = 1
    
    return np.array(mask)

def DxDx_filter(dim):
    mask = np.zeros((dim,dim))
    ref = int(dim/3)
    ref2 = math.floor(dim/2)
    i1 = ref
    i2 = 2*ref
    i3 = math.ceil(ref2/2)
    i4 = dim - math.ceil(ref2/2)
    
    mask[i3:i4, 0:i1] = 1
    mask[i3:i4, i1:i2] = -2
    mask[i3:i4, i2:dim+1] = 1
    
    return np.array(mask)

def convolve(img,fltr):
    dim1,dim2 = fltr.shape
    half1,half2 = math.floor(dim1/2), math.floor(dim2/2)
    m,n = img.shape
    
    output = np.zeros((m,n))
    
    for i in np.arange(m-2*dim1)+dim1:
        for j in np.arange(n-2*dim2)+dim2:
            box = img[i-half1-1:i+half1, j-half2-1:j+half1]
            output[i][j] = np.sum(box*fltr)
            
    return output

def convolve_with_scipy(img, fltr):
    smaller_image = signal.convolve2d(img, fltr, mode = 'valid')
    height, width = np.array(img.shape) - np.array(smaller_image.shape)
    
    top = np.zeros((math.floor(height/2),smaller_image.shape[1]))
    bottom = np.zeros((math.ceil(height/2),smaller_image.shape[1]))
    
    left = np.zeros((img.shape[0], math.floor(width/2)))
    right = np.zeros((img.shape[0], math.ceil(width/2)))
    
    output = np.concatenate([top, smaller_image, bottom])
    output = np.concatenate([left, output, right], axis=1)
    
    return output

def create_first_octave(img, num_layers):
    layers = {}
    dims = np.linspace(9,9+6*(num_layers-1), num_layers)
    count = 0
    for di in dims:
        count += 1
        dim = int(di)
        Dxx = convolve_with_scipy(img, DxDx_filter(dim))
        Dxy = convolve_with_scipy(img, DxDy_filter(dim))
        Dyy = convolve_with_scipy(img, DyDy_filter(dim))
        
        this_scale = np.abs(Dxx*Dyy - ((.9**2)*Dxy*Dxy))
        s = ((dim - 9)/6 + 1)*1.2
        layers[count] = this_scale
    return layers

def create_second_octave(img, num_layers):
    layers = {}
    dims = np.linspace(33,33+12*(num_layers-1), num_layers)
    count = 0
    for di in dims:
        count += 1
        dim = int(di)
        Dxx = convolve_with_scipy(img, DxDx_filter(dim))
        Dxy = convolve_with_scipy(img, DxDy_filter(dim))
        Dyy = convolve_with_scipy(img, DyDy_filter(dim))
        
        this_scale = np.abs(Dxx*Dyy - ((.9**2)*Dxy*Dxy))
        s = ((dim - 9)/6 + 1)*1.2
        layers[count] = this_scale
    return layers

def get_ss_first_octave(num_layers):
    return np.linspace(1.2, 1.2*num_layers, num_layers)

def get_ss_second_octave(num_layers):
    return np.linspace(4.4, 1.6*num_layers, num_layers)

def threshold(filtered_image):
    max_pixel = np.max(filtered_image)
    mean = np.mean(filtered_image)
    sd = np.std(filtered_image)
    mask = 1*(filtered_image > (mean + 2*sd))
    return mask*filtered_image

#This cell is functions we utilize to perform the max suppression

#returns the points in the 9x9x9 square that we will be using for mas suppression
def surround_pts(j,k,lyr):
    return [lyr[j-1][k-1],lyr[j-1][k],lyr[j-1][k+1],lyr[j][k-1],lyr[j][k+1],lyr[j+1][k-1],lyr[j+1][k],lyr[j+1][k+1]]


#fuction that returns true if the passed in point at (j,k) has the maximum value in its neighborhood
def check_point(j,k,previous_layer, current_layer, next_layer):
    this_point = current_layer[j][k]
    
    previous_pts = surround_pts(j,k,previous_layer)
    previous_pts.append(previous_layer[j][k])
    
    current_pts = surround_pts(j,k, current_layer)
    
    next_pts = surround_pts(j,k, next_layer)
    next_pts.append(next_layer[j][k])
    
    all_points = previous_pts + current_pts + next_pts
    
    for point in all_points:
        if this_point <= point:
            return False
    
    return True

#Function that performs the max suppression on an octave and returns the interest points in a dictionary
#where keys are the interest points and values are the corresponding s values
def find_interest_points(octave, first_or_second):
    num_layers = len(octave.keys())
    if first_or_second ==1:
        scales = get_ss_first_octave(num_layers)
    elif first_or_second ==2:
        scales = get_ss_second_octave(num_layers)
    interest_points = {}
    m,n = octave[1].shape
    for i in np.arange(num_layers-2)+2:
        previous_layer = octave[i-1]
        current_layer = octave[i]
        next_layer = octave[i+1]
    
        for j in np.arange(m-21)+10:
            for k in np.arange(n-21)+10:
                if check_point(j,k,previous_layer,current_layer,next_layer):
                    interest_points[(j,k)] = scales[i]
                    
    return interest_points

#function that draws a white circle of specified radius around the specified points
def illustrate_point(img, pt, radius):
    r = math.ceil(radius)
    output = np.zeros(img.shape)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            output[i][j] = img[i][j]
    x,y = pt
    window = img[x-r-1:x+r+1, y-r-1:y+r+1]
    m,n = window.shape
    for i in np.linspace(x-r-1, x+r+1, x+r+1 - (x-r-1) + 1):
        for j in np.linspace(y-r-1, y+r+1, y+r+1 - (y-r-1) + 1):
            length = np.sqrt((i-x)**2 + (j - y)**2)
            if radius-1 < length < radius+1:
                output[int(i)][int(j)] = 255
                
    return output

#function that draws a white circle of specified radius around the specified points
def illustrate_point(img, pt, radius):
    r = math.ceil(radius)
    output = np.zeros(img.shape)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            output[i][j] = img[i][j]
    x,y = pt
    window = img[x-r-1:x+r+1, y-r-1:y+r+1]
    m,n = window.shape
    for i in np.linspace(x-r-1, x+r+1, x+r+1 - (x-r-1) + 1):
        for j in np.linspace(y-r-1, y+r+1, y+r+1 - (y-r-1) + 1):
            length = np.sqrt((i-x)**2 + (j - y)**2)
            if radius-1 < length < radius+1:
                output[int(i)][int(j)] = 255

def gkern(kernlen=5, nsig=3):
    """Returns a 2D Gaussian kernel."""

    x = np.linspace(-nsig, nsig, kernlen+1)
    kern1d = np.diff(st.norm.cdf(x))
    kern2d = np.outer(kern1d, kern1d)
    kernel= kern2d/kern2d.sum()
    
    center_coord = math.floor(kernlen/2)
    
    max_value = kernel[center_coord,center_coord]
    
    return kernel*(1/max_value)

#function that creates a mask to zeros out everything not in the inner circle of specified radius
def circular_mask(shape, center, radius):
    h,w = shape
    X, Y = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

    mask = dist_from_center <= radius
    pts = list(zip(*np.where(mask != 0)))
    
    return mask, pts

def haar_wavelet_filter(dim, direction='x'):
    fltr = np.ones((dim,dim))
    halfway = dim//2
    
    if direction == 'x':
        #set -1 region
        fltr[:,0:halfway] = -1
    
    elif direction == 'y':
        #set -1 region
        fltr[0:halfway,:] = -1
    
    return fltr

#returns the square of "radius" half_length around the center
def square_around_pt(img, center, half_length):
    half_length = int(round(half_length))
    i,j = center
    square_img = img[i-half_length:i+half_length+1, j-half_length:j+half_length+1] 
    new_center = (square_img.shape[0]//2,square_img.shape[0]//2) #should be a square
    
    return square_img, new_center

#function that returns whether a point is in the specified sector, given start and end angle in polor coordinates
def in_sector(x, y, new_center, radius, start_angle, end_angle): #x is y and y is x
    x_c,y_c = new_center
    
    #the x coordinates are reversed in images, so switch the signs for the x differences only
    x_diff = -(x-x_c)
    
    # Calculate polar coordinates 
    p_radius = math.sqrt(x_diff**2 + (y-y_c)**2)
    p_angle = np.arctan2(x_diff, y-y_c) #technically y/x but we're dealing with an image

    #returns -pi to pi, so if negative, add 2pi
    if p_angle < 0: p_angle += 2*np.pi
        
    return (p_angle >= start_angle and p_angle < end_angle and p_radius <= radius)


'''
@description: this function takes in a circular region and and an initial orientation window and returns
              all the pixels within that scanning region within that orientation window
@params: circular_region - list of pixels that comprise the circular region of radius 6s around
                               an interest point
         x_grad, y_grad - first order gaussian derivatives after haar wavelet has already been convolved
@return: max_orientation - orientation that corresponds to the largest vector after summing the
                           x and y responses
'''
def scanning_orientation(circle, new_center, radius, dx, dy):
    pi = np.pi
    response_dict = {i:0 for i in range(8)}
    orientations = [pi/4, pi/2, 3*pi/4, pi, 5*pi/4, 3*pi/2, 7*pi/4, 0]
    response_map = {i:d for i,d in enumerate(orientations)}
    string_map = {0:'pi/4', 1:'pi/2', 2:'3pi/4', 3:'pi', 4:'5pi/4', 5:'3pi/2', 6:'5pi/4', 7:'2pi'}
    
    _,circular_pts = circular_mask(circle.shape,new_center,radius=radius) #square with mask + gaussian applied
    for pt in circular_pts:
        i,j = pt
        if in_sector(i,j, new_center, radius, pi/8, 3*pi/8): #(pi/8,3pi/8)
            response_dict[0] += dx[i][j] + dy[i][j] #do i need to make the x's negative here too?

        elif in_sector(i,j, new_center, radius, 3*pi/8, 5*pi/8): #(3pi/8,5pi/8)
            response_dict[1] += dx[i][j] + dy[i][j]

        elif in_sector(i,j, new_center, radius, 5*pi/8, 7*pi/8): #(5pi/8,7pi/8)
            response_dict[2] += dx[i][j] + dy[i][j]

        elif in_sector(i,j, new_center, radius, 7*pi/8, 9*pi/8): #(7pi/8,9pi/8)
            response_dict[3] += dx[i][j] + dy[i][j]

        elif in_sector(i,j, new_center, radius, 9*pi/8, 11*pi/8): #(9pi/8,11pi/8)
            response_dict[4] += dx[i][j] + dy[i][j]

        elif in_sector(i,j, new_center, radius, 11*pi/8, 13*pi/8): #(11pi/8,13pi/8)
            response_dict[5] += dx[i][j] + dy[i][j]
            
        elif in_sector(i,j, new_center, radius, 13*pi/8, 15*pi/8): #(13pi/8,15pi/8)
            response_dict[6] += dx[i][j] + dy[i][j]
            
        else: #(15pi/8,pi/8)
            response_dict[7] += dx[i][j] + dy[i][j]
    
    #print(response_dict)
    max_orientation = max(response_dict,key=response_dict.get)
    #print("max orientation key: ",max_orientation)
    
    return string_map[max_orientation]

#put whole process in one function
def orientation(img, interest_pt, scale):
    #get the square of length 6*s around the interest point + new center
    square, new_center = square_around_pt(img, center=interest_pt, half_length=6*scale)

    #apply gaussian to square
    mask = gkern(square.shape[0],2.5)
    square = square*mask #2.5s per the paper
    
    #apply the 15x15 circular mask to get the circle points only
    mask,_ = circular_mask(square.shape, new_center, radius=6*scale)
    circle = square*mask
    
    #create 4x4 haar wavelet filter in each direction
    hw_x = haar_wavelet_filter(math.floor(4*scale),'x') #4x4 just because
    hw_y = haar_wavelet_filter(math.floor(4*scale),'y')
    
    #apply haar wavelet filter to square in each direction
    dx = scipy.signal.convolve2d(circle, hw_x, 'same')
    dy = scipy.signal.convolve2d(circle, hw_y, 'same')

    #get the 8x8 circular mask to get the circle points only (note new radius below)
    small_mask,_ = circular_mask(circle.shape, new_center, radius=6*scale - 3.5)
    dx = dx*small_mask
    dy = dy*small_mask
    
    #scan over shifting region of pi/3 around the interest point and return the largest orientation
    max_orientation = scanning_orientation(circle, new_center, 6*scale-3.5, dx, dy)

    return max_orientation

'''
@description: this function gets the points in a circular region around an interest point
@params: img - the image to get the circular points around
         center - the interest point around which to get the points
         scale - the scale s to multiply by 6 (radius = 6*s)
@return: circular_pts - list of tuples of the surrounding points in the circular region
'''
def get_circular_pts(img, center, scale):
    radius = 6*scale
    h,w = img.shape
    X, Y = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

    mask = dist_from_center <= radius
    circular_pts = list(zip(*np.where(mask != 0)))
    
    return circular_pts

#finds distance of pt to the line spanned by the vector 'perpendicular'
def find_distance(pt, center, perpendicular):
    v1 = pt-center
    v2 = perpendicular
    factor = np.dot(v1,v2)/(np.linalg.norm(v2)**2)
    projection = v1 - factor*v2
    return np.linalg.norm(projection)

#finds points in the square rotated by pi/2 degress centered around center with 'radius' specified
def rotated_square_around_pt(img, center, radius):
    center = np.array(center)
    r = radius
    int_r = math.floor(r)
    adjusted_r = math.floor(r/np.sqrt(2))
    circle = get_circular_pts(img, center, r)
    output = []
    
    perpendicular = np.array([1,-1])
    parallel = np.array([-1,-1])
    for pt in circle:
        d1 = find_distance(np.array(pt), center, perpendicular)
        d2 = find_distance(np.array(pt), center, parallel)
        if d1 < adjusted_r and d2 < adjusted_r:
            output.append(pt)   
    
    return np.array(output)

#helper function which returns the 16 centers of the descriptor blocks, according to the passed in orientation
def get_descriptor_centers(img, center, orientation, s):
    i,j = center
    box_dim = 4*s/np.sqrt(2)
    up1 = math.ceil(box_dim)
    up2 = math.ceil(2*box_dim)
    up3 = math.ceil(3*box_dim)
    
    centers = {}
    if orientation == '7pi/4':
        centers[4] = (i + up3,j)
        centers[3] = (i + up2, j+up1)
        centers[2] = (i + up1, j+up2)
        centers[1] = (i, j+ up3)
        
        centers[8] = (i + up2, j-up1)
        centers[7] = (i + up1, j)
        centers[6] = (i, j+up1)
        centers[5] = (i - up1, j+ up2)
        
        centers[12] = (i + up1, j - up2)
        centers[11] = (i, j-up1)
        centers[10] = (i - up1, j)
        centers[9] = (i - up2, j+up1)
        
        centers[13] = (i,j-up3)
        centers[14] = (i - up1, j-up2)
        centers[15] = (i -up2, j-up1)
        centers[16] = (i - up3, j)
        
    elif orientation == '5pi/4':
        centers[4] = (i,j - up3)
        centers[3] = (i +up1, j-up2)
        centers[2] = (i + up2, j-up1)
        centers[1] = (i + up3, j)
        
        centers[8] = (i - up1, j-up2)
        centers[7] = (i, j - up1)
        centers[6] = (i+up1, j)
        centers[5] = (i + up2, j+ up1)
        
        centers[12] = (i - up2, j - up1)
        centers[11] = (i-up1, j)
        centers[10] = (i, j+ up1)
        centers[9] = (i + up1, j+up2)
        
        centers[16] = (i - up3 ,j)
        centers[15] = (i - up2, j+ up1)
        centers[14] = (i -up1, j+up2)
        centers[13] = (i, j+up3)
        
    elif orientation == '3pi/4':
        centers[13] = (i + up3,j)
        centers[14] = (i + up2, j+up1)
        centers[15] = (i + up1, j+up2)
        centers[16] = (i, j+ up3)
        
        centers[9] = (i + up2, j-up1)
        centers[10] = (i + up1, j)
        centers[11] = (i, j+up1)
        centers[12] = (i - up1, j+ up2)
        
        centers[5] = (i + up1, j - up2)
        centers[6] = (i, j-up1)
        centers[7] = (i - up1, j)
        centers[8] = (i - up2, j+up1)
        
        centers[1] = (i,j-up3)
        centers[2] = (i - up1, j-up2)
        centers[3] = (i -up2, j-up1)
        centers[4] = (i - up3, j)
        
    elif orientation == 'pi/4':
        centers[13] = (i,j - up3)
        centers[14] = (i +up1, j-up2)
        centers[15] = (i + up2, j-up1)
        centers[16] = (i + up3, j)
        
        centers[9] = (i - up1, j-up2)
        centers[10] = (i, j - up1)
        centers[11] = (i+up1, j)
        centers[12] = (i + up2, j+ up1)
        
        centers[5] = (i - up2, j - up1)
        centers[6] = (i-up1, j)
        centers[7] = (i, j+ up1)
        centers[8] = (i + up1, j+up2)
        
        centers[1] = (i - up3 ,j)
        centers[2] = (i - up2, j+up1)
        centers[3] = (i -up1, j+up2)
        centers[4] = (i, j+up3)
    return centers

#Returns the 16 blocks of the descriptor in order of how they are counted according to the orientations

def get_straight_blocks(img, interest_pt, s, orientation):
    i,j = interest_pt
    
    up1 = math.ceil(5*s)
    up2 = math.ceil(10*s)
    
    blocks = {}
    
    blocks[4] = img[i+up1:i+up2 , j+up1: j+up2]
    blocks[3] = img[i:i+up1 , j+up1: j+up2]
    blocks[2] = img[i-up1:i , j+up1: j+up2]
    blocks[1] = img[i-up2:i-up1 , j+up1:j+up2]
    
    blocks[8] = img[i+up1:i+up2 , j: j+up1]
    blocks[7] = img[i:i+up1 , j: j+up1]
    blocks[6] = img[i-up1:i , j: j+up1]
    blocks[5] = img[i-up2:i-up1 , j: j+up1]
    
    blocks[12] = img[i+up1:i+up2, j-up1: j]
    blocks[11] = img[i:i+up1 , j-up1: j]
    blocks[10] = img[i-up1:i , j-up1: j]
    blocks[9] = img[i-up2:i-up1 , j-up1: j]

    blocks[16] = img[i+up1:i+up2, j-up2: j-up1]
    blocks[15] = img[i:i+up1 , j-up2: j-up1]
    blocks[14] = img[i-up1:i , j-up2: j-up1]
    blocks[13] = img[i-up2:i-up1 , j-up2: j-up1]
    
    output = {}
    
    if orientation == '2pi':
        output = blocks
    elif orientation == 'pi/2':
        output[1] = blocks[13]
        output[2] = blocks[9]
        output[3] = blocks[5]
        output[4] = blocks[1]
        
        output[5] = blocks[14]
        output[6] = blocks[10]
        output[7] = blocks[6]
        output[8] = blocks[2]
        
        output[9] = blocks[15]
        output[10] = blocks[11]
        output[11] = blocks[7]
        output[12] = blocks[3]
        
        output[13] = blocks[16]
        output[14] = blocks[12]
        output[15] = blocks[8]
        output[16] = blocks[4]
        
    elif orientation == 'pi':
        output[1] = blocks[16]
        output[2] = blocks[15]
        output[3] = blocks[14]
        output[4] = blocks[13]

        output[5] = blocks[12]
        output[6] = blocks[11]
        output[7] = blocks[10]
        output[8] = blocks[9]

        output[9] = blocks[8]
        output[10] = blocks[7]
        output[11] = blocks[6]
        output[12] = blocks[5]

        output[13] = blocks[4]
        output[14] = blocks[3]
        output[15] = blocks[2]
        output[16] = blocks[1]
        
    elif orientation == '3pi/2':
        output[1] = blocks[4]
        output[2] = blocks[8]
        output[3] = blocks[12]
        output[4] = blocks[16]
        
        output[5] = blocks[3]
        output[6] = blocks[7]
        output[7] = blocks[11]
        output[8] = blocks[15]
        
        output[9] = blocks[2]
        output[10] = blocks[6]
        output[11] = blocks[10]
        output[12] = blocks[14]
        
        output[13] = blocks[1]
        output[14] = blocks[5]
        output[15] = blocks[9]
        output[16] = blocks[13]
        
        
    return output
        

def get_descriptor(img, interest_pt, s, orientation):
    img_dx = convolve_with_scipy(lady, haar_wavelet_filter(math.floor(2*s), direction='x'))
    img_dy = convolve_with_scipy(lady, haar_wavelet_filter(math.floor(2*s), direction='y'))
    #applies the gaussian mask centered at interest_pt
    i,j = interest_pt
    bsd = math.ceil(10*s*np.sqrt(2))
    mask = gkern(2*bsd,3.3)
    big_square_dx = mask*img_dx[i-bsd:i+bsd, j-bsd:j+bsd]
    big_square_dy = mask*img_dy[i-bsd:i+bsd, j-bsd:j+bsd]
                
    img_dx[i-bsd:i+bsd, j-bsd:j+bsd] = big_square_dx
    img_dy[i-bsd:i+bsd, j-bsd:j+bsd] = big_square_dy    
    
    descriptor = []
    if orientation == 'pi/4' or orientation == '3pi/4' or orientation == '5pi/4' or orientation == '7pi/4':
        centers = get_descriptor_centers(img_dx, (i,j), orientation, s)
        for i in np.arange(16)+1:
            dx_sum = 0
            dy_sum = 0
            dx_abs_sum = 0
            dy_abs_sum = 0
            block = rotated_square_around_pt(img_dx, centers[i], 4*s)
            for pt in block:
                dx_value = img_dx[pt[0]][pt[1]]
                dy_value = img_dy[pt[0]][pt[1]]
                dx_sum += dx_value
                dy_sum += dy_value
                dx_abs_sum += np.abs(dx_value)
                dy_abs_sum += np.abs(dy_value)
            
            descriptor += [dx_sum, dy_sum, dx_abs_sum, dy_abs_sum]
    
    elif orientation == '2pi' or orientation == 'pi/2' or orientation == 'pi' or orientation == '3pi/2':
        blocks_dx =get_straight_blocks(img_dx, interest_pt, s, orientation)
        blocks_dy =get_straight_blocks(img_dy, interest_pt, s, orientation)
        
        for i in np.arange(16)+1:
            dx_sum = np.sum(blocks_dx[i])
            dy_sum = np.sum(blocks_dy[i])
            dx_abs_sum = np.sum(np.abs(blocks_dx[i]))
            dy_abs_sum = np.sum(np.abs(blocks_dy[i]))
            
            descriptor += [dx_sum, dy_sum, dx_abs_sum, dy_abs_sum]
        
    return np.array(descriptor)

def get_descriptor_for_all_points(img, first_octave_interest_points, second_octave_interest_points, first_octave_orientations, second_octave_orientations):
    
    descriptors = {}
    count = 0
    print('Getting descriptors for first octave...')
    start_time = time.time()
    for interest_pt in first_octave_interest_points.keys():
        try:
            descriptor = get_descriptor(img, interest_pt, first_octave_interest_points[interest_pt], first_octave_orientations[interest_pt])
            descriptors[interest_pt] = descriptor
            count += 1
            if count %10 == 0:
                print('Found descriptors for', count, 'interest points')
        except Exception as e:
            print('interest point', interest_pt, 'too close to edge')
            continue
    print('###################################################')
    print('First octave took', time.time()-start_time,'seconds')
    print('###################################################')
    print('')
    print('Getting descriptors for second octave...')
    start_time = time.time()
    for interest_pt in second_octave_interest_points.keys():
        try:
            descriptor = get_descriptor(img, interest_pt, second_octave_interest_points[interest_pt], second_octave_orientations[interest_pt])
            descriptors[interest_pt] = descriptor
            print(count)
            if count %10 == 0:
                print('Found descriptors for', count, 'interest points')
        except Exception as e:
            print('interest point', interest_pt, 'too close to edge')
            continue
            
    print('###################################################')
    print('Second octave took', time.time()-start_time,'seconds')
    print('###################################################')
    
    return descriptors

def SURF(img):
    #getting octaves
    start_time = time.time()
    print('creating first octave...')
    first_octave = create_first_octave(img, 8)
    print('time to make first octave:', time.time() - start_time, 'seconds')
    print('')
    print('creating second octave...')
    start_time = time.time()
    second_octave = create_second_octave(img, 8)
    print('time to make second octave:' ,time.time() - start_time, 'seconds')
    print('')
    print('s (sigma) values for the first octave:')
    print(get_ss_first_octave(8))
    print('')
    print('s (sigma) values for the second octave:')
    print(get_ss_second_octave(8))
    
    #thresholding
    thresholded_first_octave = {}
    for layer in first_octave.keys():
        thresholded_first_octave[layer] = threshold(first_octave[layer])

    thresholded_second_octave = {}
    for layer in second_octave.keys():
        thresholded_second_octave[layer] = threshold(second_octave[layer])
    
    #finding interest points
    print('')
    print('finding interest points in 1st octave...')
    start_time = time.time()
    first_octave_interest_points = find_interest_points(thresholded_first_octave, 1)
    print('finding interest points took', time.time() - start_time, 'seconds')

    print('')

    print('finding interest points in 2st octave...')
    start_time = time.time()
    second_octave_interest_points = find_interest_points(thresholded_second_octave, 2)
    print('finding interest points took', time.time() - start_time, 'seconds')

    print('')

    print('found', len(first_octave_interest_points.keys()), 'interest points in first octave and ', len(second_octave_interest_points.keys()), 'interest points in second octave')
    copy = img
    for pt in first_octave_interest_points.keys():
        copy = illustrate_point(copy, pt, first_octave_interest_points[pt])
    for pt in second_octave_interest_points.keys():
        copy = illustrate_point(copy, pt, second_octave_interest_points[pt])

    plt.figure(figsize=((9,9)))
    plt.imshow(copy)
    plt.show()
    print('')
    print('finding orientations for first octave...')
    start_time = time.time()
    first_octave_orientations = {}

    new_first_octave = first_octave_interest_points.copy()
    for interest_pt in new_first_octave.keys():
        try:
            first_octave_orientations[interest_pt] = orientation(img, interest_pt, new_first_octave[interest_pt])
            #print(first_octave_orientations[interest_pt])
        except:
            print('interest point too close to edge', interest_pt)
            first_octave_interest_points.pop(interest_pt)
            continue
    print('Orientations for first octave took', time.time() - start_time, 'seconds')

    print('')

    print('finding orientations for second octave...')
    start_time = time.time()
    second_octave_orientations = {}
    new_second_octave = second_octave_interest_points.copy()
    for interest_pt in new_second_octave.keys():
        try:
            second_octave_orientations[interest_pt] = orientation(img, interest_pt, new_second_octave[interest_pt])
            #print(second_octave_orientations[interest_pt])
        except:
            print('interest point too close to edge', interest_pt)
            second_octave_interest_points.pop(interest_pt)
            continue
    print('Orientations for second octave took', time.time() - start_time, 'seconds')
    
    print('')
    
    descriptors = get_descriptor_for_all_points(img, first_octave_interest_points, second_octave_interest_points, first_octave_orientations, second_octave_orientations)
    
    return descriptors