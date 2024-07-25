import cv2
import numpy as np
from tqdm import tqdm
import itertools

def extend_image_left_top_loop(image, size=2):
	height, width, channels = image.shape
	# NOTE: the extra rightmost 2 columns are required for the extended image of the original image (that area will be masked by the patch)
	extended_img = np.zeros((height+size, width+size+size, channels), np.uint8)
	# Fill the top 2 rows with the same values as the bottom 2 rows of `image`
	for i in range(size):
		for j in range(size, width+size):
			for k in range(channels):
				extended_img[i][j][k] = image[i-size][j-size][k]

	# Fill the left 2 columns with the same values as the right 2 columns of `image`
	for j in range(size):
		for i in range(size, height+size):
			for k in range(channels):
				extended_img[i][j][k] = image[i-size][j-size][k]

	# Fill the top-left 2x2 corner with the same values as the bottom-right 2x2 corner of `image`
	for i in range(size):
		for j in range(size):
			for k in range(channels):
				extended_img[i][j][k] = image[i-size][j-size][k]

	# Fill the rest of the `image` in the extended image
	for i in range(size, height+size):
		for j in range(size, width+size):
			for k in range(channels):
				extended_img[i][j][k] = image[i-size][j-size][k]

	# The following is only required for the extended image of the original image
	# Fill the rightmost 2 columns with the same values as the left 2 rows of `image`
	for j in range(width-size, width):
		for i in range(size, height+size):
			for k in range(channels):
				extended_img[i][j][k] = image[size-(width-i)][j-size][k]

	# No need to fill the bottom 2 rows (as the patch will never masks the bottom 2 rows)

def patch_dist(patch1, patch2):
	if len(patch1[0]) != 1:
		patch1 = np.array(patch1).flatten()
		patch2 = np.array(patch2).flatten()
	# L2 norm
	return np.linalg.norm(patch1 - patch2)

def distance(patches_T, patches_S):
	dist = 0
	for t in patches_T:
		dist_patches = float('inf')
		argmin_s = patches_S[0]
		for s in patches_S:
			if patch_dist(t, s) < dist_patches:
				dist_patches = patch_dist(t, s)**2
				argmin_s = s
		dist += dist_patches
	return dist

def extract_patch(extended_img, i, j):
		"""
		grid = np.zeros(5+5+3)

		0  1  2  3  4
		5  6  7  8  9
		10 11 12

		12 = (i, j)

		(i-2, j-2) (i-2, j-1)
		"""

		patch_gen_img = []
		patch_gen_img.append(extended_img[i-2][j-2])
		patch_gen_img.append(extended_img[i-2][j-1])
		patch_gen_img.append(extended_img[i-2][j])
		patch_gen_img.append(extended_img[i-2][j+1])
		patch_gen_img.append(extended_img[i-2][j+2])
		patch_gen_img.append(extended_img[i-1][j-2])
		patch_gen_img.append(extended_img[i-1][j-1])
		patch_gen_img.append(extended_img[i-1][j])
		patch_gen_img.append(extended_img[i-1][j+1])
		patch_gen_img.append(extended_img[i-1][j+2])
		patch_gen_img.append(extended_img[i][j-2])
		patch_gen_img.append(extended_img[i][j-1])
		patch_gen_img.append(extended_img[i][j])
		
		return patch_gen_img

# read the image
img = cv2.imread('imgs/161.jpg')

print(img.size) # 64*64*3

height = 128
width = 128
channels = 3
new_img = np.random.randint(0, 255, (height, width, channels), np.uint8)
extended_new_img = extend_image_left_top_loop(new_img, size=2)

extend_org_img = extend_image_left_top_loop(img, size=2)




for k in range(10):
	## Seach Step (fix T, the target patch)
	T = []
	S = []
	s_t_pairs = []
	for i_org in range(2, img.shape[0]+2):
		for j_org in range(2, img.shape[1]+2):
				# Extract the patch centered at (i, j)
				s = extract_patch(extend_org_img, i, j)
				s.append(S)

	for i in range(2, height+2):
		for j in range(2, width+2):
			# Extract the patch centered at (i, j)
			t = extract_patch(extended_new_img, i, j)
			t.append((T, (i, j)))

			min_dist = float('inf')
			best_patch = None
			for s in S:
				# Compute the distance between the patches
				dist = distance(t, s)
				# Update the patch if the distance is smaller
				if dist < min_dist:
					min_dist = dist
					best_patch = s
			
			s_t_pairs.append((best_patch, t))

	## Mixtures Step (fix S)
	for i in range(2, height+2):
		for j in range(2, width+2):
			# extract all filters related to the patch
			patches_T = []

			for s, t in s_t_pairs:
				t_x = t.second[0]
				t_y = t.second[1]

				if (t_x == i-2 or t_x == i-1) and (j-2 <= t_y <= j+2):
					patches_T.append(t.first)
					# average += t.first[i-1]
				if t_x == i and (j-2 <= t_y <= j):
					patches_T.append(t.first)

				



