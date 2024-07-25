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

	
	return extended_img

def patch_dist(patch1, patch2):
	if len(patch1[0]) != 1:
		patch1 = np.array(patch1).flatten()
		patch2 = np.array(patch2).flatten()
	# L2 norm
	return np.linalg.norm(patch1 - patch2)

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

# Params
height = 128
width = 128

# image array for new image
new_img = np.zeros((height, width, 3), np.uint8)

# grid 5-5-3 (partial 2D array)
grid = np.zeros(5+5+3)

# randomly assign rightmost 2 columns and bottom 2 rows
for i in range(height-2, height):
	for j in range(width):
		for k in range(3):
			new_img[i][j][k] = np.random.randint(0, 256, dtype=np.uint8)
	"""
		The following code does not work as expected.
		```python
		for i in range(height-2, height):
			# for j in range(width):
			for k in range(3):
				new_img[i][:][k] = np.random.randint(0, 256, width, dtype=np.uint8)
		```

		This is due to the numpy array slicing in python.
		print(np.shape(new_img[:][:][:])) # 128*128*3
		print(np.shape(new_img[i][:][:])) # 128*3
		print(np.shape(new_img[:][i][:])) # 128*3
		print(np.shape(new_img[i][k][:])) # 3
		print(np.shape(new_img[i][:][k])) # 3 (but not 128!)
		
		Let us interpret the 3D array as a 2D array of 3 channels (k).
		```new_image[i][:][k]``` does not mean extracting the k-th channel of the i-th row.
		
		This is because in python, 
		- new_image[i] means extracting the i-th 2D slice of the 3D array `new_image`.
		- new_image[i][:] means extracting the i-th row of the 1D slice of the 2D array `new_image[i]`.
		- new_image[i][:][k] means extracting the k-th element of the 1D slice of the 2D array `new_image[i][:]`.
	"""

for j in range(width-2, width):
	for i in range(height):	
		for k in range(3):
			new_img[i][j][k] = np.random.randint(0, 256, dtype=np.uint8)

# Save the image (initial image with rightmost 2 columns and bottom 2 rows filled with random values)
cv2.imwrite('imgs/161_128.jpg', new_img)

# Create an extended image with leftmost 2 columns and top 2 rows are filled with the same values as the rightmost 2 columns and bottom 2 rows of the new image
extended_new_img = extend_image_left_top_loop(new_img)
# Save the extended image of the new image
cv2.imwrite('imgs/161_128_2.jpg', extended_new_img)

extended_org_img = extend_image_left_top_loop(img)
# Save the extended image of the original image
cv2.imwrite('imgs/161_ext.jpg', extended_org_img)

# The rightmost 2 columns and bottom 2 rows of the new image are filled with random values, so no need to calc in the above
for i, j in tqdm(itertools.product(range(2, height), range(2, width)), total=(height-2)*(width-2)):
		patch_ext = extract_patch(extended_new_img, i, j)

		dist = float('inf')
		argmin_i = 0
		argmin_j = 0

		# # Search the most similar 5-5-3 patch in the original image
		for i_org in range(2, img.shape[0]+2):
			for j_org in range(2, img.shape[1]+2):
				patch_org = extract_patch(extended_org_img, i_org, j_org)

				# Calculate the distance between the two patches
				if patch_dist(patch_ext, patch_org) < dist:
					dist = patch_dist(patch_ext, patch_org)
					argmin_i = i_org
					argmin_j = j_org
		
		# Update the pixel value of the new image
		for k in range(3):
			new_img[i-2][j-2][k] = extended_org_img[argmin_i][argmin_j][k]
		
		# Save the image while updating the pixel value
		cv2.imwrite('imgs/161_128_reconstructed.jpg', new_img)

# Save the image (final)
cv2.imwrite('imgs/161_128_reconstructed.jpg', new_img)