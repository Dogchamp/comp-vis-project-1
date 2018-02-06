# Computer Vision Project 1 - Glass Plate Color Channel Alignment
# Uses an exhaustive Gaussian Pyramid coarse-to-fine search to align 

import scipy
from scipy import signal
import imageio as imgio
import numpy as np
import cv2

# ======================================  Gaussian Pyramid   ======================================  
# Returns a Gaussian kernel meant for use as a Gaussian low pass filter
def make_g_kernel(a):
	kernel = np.array([0.25-(a/2.0), 0.25, a, 0.25, 0.25-(a/2.0)])
	return np.outer(kernel, kernel)


# Passes image through a gaussian filter and then reduces it by half
def scaledown(image):
	g_kernel = make_g_kernel(0.4)
	convolved = signal.convolve2d(image, g_kernel, 'same')
	reduced_im = convolved[::2, ::2]
	return reduced_im

# Returns gaussian pyramid of image 'image' levels 'levels' deep
def g_pyr(image, levels):
	out = []
	out.append(image)
	tmp = image
	for level in range(0, levels):
		tmp = scaledown(tmp)
		out.append(tmp)
	return out
 
# ==================================== SCORING ========================================
# Scores image a relative to image b via the sum of the squared differences
# Smaller ssd score is better
def score(a, b):
	squared_diff = (a - b)**2
	return np.sum(squared_diff)
	
# Exhaustively searches over a diameter of 2r to find displacement that results in best match according to score()
def match(a, b, r):
	scores = np.zeros((r*2, r*2))
	best_score, best_x, best_y = 100000, 0, 0
	for i in xrange(-r, r):
		for j in xrange(-r, r):
			a_new = np.roll(a, j, 0)	# move on x-axis one pixel
			a_new = np.roll(a, i, 1)	# move on y-axis one pixel
			current_score = score(a_new, b)
			if current_score < best_score:
				best_score = current_score
				best_x = i+r
				best_y = j+r
	print( "lowest ssd: %s " % best_score)
	print( best_x, best_y )
	return (best_x, best_y)

#  ======================================  MAIN  ====================================== 
# name of the input image
imname = 'cathedral'
fname = imname + '.jpg'	

# read in the image (opencv2 natively reads images as a numpy array), as grayscale
im = cv2.imread(fname, cv2.IMREAD_GRAYSCALE)

# get the height and width of each image
total_sz = im.shape
height = int(total_sz[0]/3)
width = total_sz[1]
print(" height: %s " % height)
print(" width: %s " % width)
cv2.imshow(imname, im)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Bells and Whistles - Autocontrast using CLAHE 
# (Contrast Limited Adaptive Histograme Equalization)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
clim = clahe.apply(im)
print( "Applying autocontrast..." )
cv2.imshow('contrast', clim)
cv2.waitKey(0)
cv2.destroyAllWindows()
imgio.imwrite(imname + '_clahe.jpg', clim)

# Separate channels for each color
r = im[:height]
g = im[height:2*height]
b = im[2*height:3*height]
r = r.astype('float')/255
g = g.astype('float')/255
b = b.astype('float')/255
rc_pyr = g_pyr(r, 4)
gc_pyr = g_pyr(g, 4)
bc_pyr = g_pyr(b, 4)

cv2.imshow("red channel", r)
cv2.imshow("0", rc_pyr[0])
cv2.imshow("1", rc_pyr[1])
cv2.imshow("2", rc_pyr[2])
cv2.imshow("3", rc_pyr[3])
cv2.waitKey(0)
cv2.destroyAllWindows()

# vectors to hold displacements found during alignment

# align the images
# functions that might be useful for aligning the images include:
# np.roll, np.sum, sk.transform.rescale (for multiscale)
r_to_g_total = np.zeros(2)
g_to_b_total = np.zeros(2)
for i in xrange(0,3):
	r_to_g = match(rc_pyr[i], gc_pyr[i], 15)
	r_to_g_total[0] += r_to_g[0]
	r_to_g_total[1] += r_to_g[1]
	g_to_b = match(gc_pyr[i], bc_pyr[i], 15)
	g_to_b_total[0] += g_to_b[0]
	g_to_b_total[1] += g_to_b[1]
	ar = np.roll(r, r_to_g[0], axis=0)
	ar = np.roll(r, r_to_g[1], axis=1)
	ag = np.roll(g, g_to_b[0], axis=0)
	ag = np.roll(g, g_to_b[1], axis=1)
	 
# create a color image
im_out = np.dstack([ar, ag, b])

# display the image, wait for a keystroke, save the image, and exit
cv2.imshow(imname, im_out)
cv2.waitKey(0)
cv2.destroyAllWindows()
imgio.imwrite(imname + '_colored.jpg' , im_out)
