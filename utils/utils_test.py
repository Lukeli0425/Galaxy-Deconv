import torch
from torchvision.transforms import Compose, Resize, Grayscale, ToTensor
from torch.utils.data import Dataset, DataLoader
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from utils_poisson_deblurring.utils_torch import conv_fft, img_to_tens, scalar_to_tens
from utils_poisson_deblurring.utils_deblur import gauss_kernel, pad, crop
from PIL import Image
import cv2 as cv

def p4ip_wrapper(y, k, M, p4ip):
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	
	Ht = img_to_tens(k).to(device).float()	
	yt = img_to_tens(y).to(device)
	Mt = scalar_to_tens(M).to(device)

	with torch.no_grad():
		x_rec_list = p4ip(yt, Ht, Mt)
	x_rec = x_rec_list[-1]
	x_rec = np.clip(x_rec.cpu().detach().numpy(),0,1)
	x_out = x_rec[0,0,:,:]
	return x_out

def p4ip_wrapper_pad(y, k, M, p4ip):
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	H, W = np.shape(y)[0], np.shape(y)[1]
	H1, W1 =  np.int32(H/2), np.int32(W/2)  
	y_pad = np.pad(y, ((H1,H1),(W1,W1)), mode='symmetric')

	Ht = img_to_tens(k).to(device).float()	
	yt = img_to_tens(y_pad).to(device)
	Mt = scalar_to_tens(M).to(device)

	with torch.no_grad():
		x_rec_list = p4ip(yt, Ht, Mt)
	
	x_rec = x_rec_list[-1]
	x_rec = np.clip(x_rec.cpu().detach().numpy(),0,1)

	x_out = x_rec[0,0,:,:]
	x_out = x_out[H1:H1+H, W1:W1+W]
	return x_out


def p4ip_bayer(y, k, p4ip):
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

	y_list = []
	M_list = []
	y_list.append(y[0::2, 0::2])
	y_list.append(y[1::2, 0::2])
	y_list.append(y[0::2, 1::2])
	y_list.append(y[1::2, 1::2])

	H, W = np.shape(y_list[0])
	H1, W1 =  np.int32(H/2), np.int32(W/2)  
	
	H2, W2 =  np.shape(k)
	H3, W3 =  np.int32(np.ceil(H2/2)*2+1), np.int32(np.ceil(W2/2)*2+1)
	N3 = np.max([H3,W3])
	k = pad(k, [N3,N3])
	kt = img_to_tens(k).to(device).float()	
	out_rggb = np.zeros([4,1,H,W], dtype=np.float32)
	out = np.zeros(np.shape(y))

	idx = 0
	for y in y_list:

		y1 = np.clip(y,0,np.inf)
		M  = np.mean(np.ravel(y1))/0.33 

		M_list.append(M)
		y_pad = np.pad(y1, ((H1,H1),(W1,W1)), mode='symmetric')

		yt = img_to_tens(y_pad).to(device)
		Mt = scalar_to_tens(M).to(device)

		with torch.no_grad():
			x_rec_list = p4ip(yt, kt, Mt)
		x_rec = x_rec_list[-1]
		x_rec = np.clip(x_rec.cpu().detach().numpy(),0,1)
		x_out = x_rec[0,0,:,:]
		out_rggb[idx,0,:,:] = x_out[H1:H1+H, W1:W1+W]
		idx +=1

	out[0::2, 0::2] = out_rggb[0]*M_list[0]
	out[1::2, 0::2] = out_rggb[1]*M_list[1]
	out[0::2, 1::2] = out_rggb[2]*M_list[2]
	out[1::2, 1::2] = out_rggb[3]*M_list[3]

	return out, out_rggb, M_list

def rggb_to_rgb(rggb, H, W, mode='BGGR'):
	out = np.zeros([H,W], dtype=np.float32)
	out[0::2, 0::2] = rggb[0]
	out[1::2, 0::2] = rggb[1]
	out[0::2, 1::2] = rggb[2]
	out[1::2, 1::2] = rggb[3]

	out_rgb = bayer_to_rgb(out, mode)
	return out_rgb	

def bayer_to_rgb(x, mode='BGGR'):
	min_x, max_x = np.min(np.ravel(x)), np.max(np.ravel(x))
	x1 = np.uint16((x - min_x)/(max_x - min_x) * (2**16 - 1))

	if mode == 'BGGR':
		x_rgb = cv.cvtColor( x1, cv.COLOR_BayerRG2BGR)
		x_rgb = x_rgb.astype('float32')*(max_x - min_x)/(2**16 - 1) + min_x
		x_rgb[:,:,0] *= 2310/1024
		x_rgb[:,:,2] *= 1843/1024

	if mode == 'RGGB':
		x_rgb = cv.cvtColor( x1, cv.COLOR_BayerGB2BGR)
		x_rgb = x_rgb.astype('float32')*(max_x - min_x)/(2**16 - 1) + min_x
		x_rgb[:,:,0] *= 2310/1024
		x_rgb[:,:,2] *= 1843/1024

	return x_rgb


def img_register_file(file_true, file_est):
	img1_color = cv.imread(file_true)  # Image to be aligned
	img2_color = cv.imread(file_est)  # Image to be aligned

	img1 = cv.cvtColor(img1_color, cv.COLOR_BGR2GRAY)
	img2 = cv.cvtColor(img2_color, cv.COLOR_BGR2GRAY)
	height, width = img2.shape

	# Create ORB detector with 5000 features.
	orb_detector = cv.ORB_create(5000)

	# Find keypoints and descriptors.
	# The first arg is the image, second arg is the mask
	#  (which is not reqiured in this case).
	kp1, d1 = orb_detector.detectAndCompute(img1, None)
	kp2, d2 = orb_detector.detectAndCompute(img2, None)

	# Match features between the two images.
	# We create a Brute Force matcher with
	# Hamming distance as measurement mode.
	matcher = cv.BFMatcher(cv.NORM_HAMMING, crossCheck = True)


	# Match the two sets of descriptors.
	matches = matcher.match(d1, d2)

	# Sort matches on the basis of their Hamming distance.
	matches.sort(key = lambda x: x.distance)

	# Take the top 90 % matches forward.
	matches = matches[:int(len(matches)*100)]
	no_of_matches = len(matches)

	# Define empty matrices of shape no_of_matches * 2.
	p1 = np.zeros((no_of_matches, 2))
	p2 = np.zeros((no_of_matches, 2))

	for i in range(len(matches)):
		p1[i, :] = kp1[matches[i].queryIdx].pt
		p2[i, :] = kp2[matches[i].trainIdx].pt

	# Find the homography matrix.
	homography, mask = cv.findHomography(p1, p2, cv.RANSAC)

	# Use this matrix to transform the
	# colored image wrt the reference image.
	transformed_img = cv.warpPerspective(img1_color, homography, (width, height))

	return transformed_img, img2_color



def img_register(im_true, im_est):
	img1_color = im_true # Image to be aligned
	img2_color = im_est  # Reference image to which img1_color is aligned

	img1 = cv.cvtColor(img1_color, cv.COLOR_BGR2GRAY)
	img2 = cv.cvtColor(img2_color, cv.COLOR_BGR2GRAY)
	height, width = img2.shape

	# Create ORB detector with 5000 features.
	orb_detector = cv.ORB_create(5000)

	# Find keypoints and descriptors.
	# The first arg is the image, second arg is the mask
	#  (which is not reqiured in this case).
	kp1, d1 = orb_detector.detectAndCompute(img1, None)
	kp2, d2 = orb_detector.detectAndCompute(img2, None)

	# Match features between the two images.
	# We create a Brute Force matcher with
	# Hamming distance as measurement mode.
	matcher = cv.BFMatcher(cv.NORM_HAMMING, crossCheck = True)


	# Match the two sets of descriptors.
	matches = matcher.match(d1, d2)

	# Sort matches on the basis of their Hamming distance.
	list(matches).sort(key = lambda x: x.distance)

	# Take the top 90 % matches forward.
	matches = matches[:int(len(matches)*100)]
	no_of_matches = len(matches)

	# Define empty matrices of shape no_of_matches * 2.
	p1 = np.zeros((no_of_matches, 2))
	p2 = np.zeros((no_of_matches, 2))

	for i in range(len(matches)):
		p1[i, :] = kp1[matches[i].queryIdx].pt
		p2[i, :] = kp2[matches[i].trainIdx].pt

	# Find the homography matrix.
	homography, mask = cv.findHomography(p1, p2, cv.RANSAC)

	# Use this matrix to transform the
	# colored image wrt the reference image.
	transformed_img = cv.warpPerspective(img1_color,
										  homography, (width, height))

	return transformed_img, img2_color