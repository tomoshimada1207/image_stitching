import os

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

### Functions ###

def find_good_matches(des_ref, des_1):
    ''' 
    takes in descriptor matrix (# of descriptors, # of descriptor bins)
    and uses k-nearest neighbors to find the best k matches.
    Then will compare which match is best by calculating the euclidean distances
    '''

    # k nearest neighbors
    k=2

    # BF Matcher
    bf = cv.BFMatcher()
    matches = bf.knnMatch(des_ref,des_1,k=2)

    # Apply Lowe's ratio test
    good_matches = []
    for m,n in matches:
        if m.distance < 0.75*n.distance:
            good_matches.append([m])

    return good_matches

def coordinates_of_matches(good_matches, kp_ref, kp_1):
    '''
    Takes the list (with a list) of DMatch types and finds which kp matches up with eachother.
    Then returns two ordered 2xN numpy array of coordinates that are matches with eachother. 
    In other words, the coordinates kp_ref[0, :] is a match with the coordinates kp_1[0, :]
    in the two different images. 
    '''

    # List of Coordinates for key points
    kp_ref_coords = np.empty((0, 2))
    kp_1_coords = np.empty((0, 2))

    for matches in good_matches:
        # Get the matching key points from both images
        img_ref_idx = matches[0].queryIdx
        img_1_idx = matches[0].trainIdx

        # Getting the coordinates 
        (x_ref, y_ref) = kp_ref[img_ref_idx].pt
        (x_1, y_1) = kp_1[img_1_idx].pt

        # Put all coordinates of match pairs into two list 
        kp_ref_coords = np.row_stack((kp_ref_coords,(x_ref, y_ref)))
        kp_1_coords = np.row_stack((kp_1_coords,(x_1, y_1)))
    
    return kp_ref_coords, kp_1_coords

def find_homography_matrix(kp_ref_coords, kp_1_coords):
    '''
    Uses input of 2xN numpy arrays which contain the coordinates of match pairs.
    Uses RANSAC to find which Homography gives the most matches b/w matches (Filters out
    False Positives). 
    Optional refinding of Homography with just using the good points included (commented out)
    '''

    # Note, where stat == 1 is a TP, otherwise FP (aka not good match according to RANSAC)
    H, mask = cv.findHomography(kp_1_coords, kp_ref_coords, cv.RANSAC, ransacReprojThreshold=2)
    assert mask.sum() >= 4 # must be atleast 4 matches

    # Only getting the good points, method: & -> find non-zeros, then reshape
    temp_ref = kp_ref_coords*mask
    kp_ref_good = temp_ref[np.where(temp_ref != 0)]
    kp_ref_good = kp_ref_good.reshape((mask.sum(), 2))

    temp_1 = kp_1_coords*mask
    kp_1_good = temp_1[np.where(temp_1 != 0)]
    kp_1_good = kp_1_good.reshape((mask.sum(), 2))

    # Uncomment for robustness to redo find Homography with the good points (Optional)
    # H, _ = cv.findHomography(kp_ref_good, kp_1_good)

    return H

def stitch_pos_checker(H, img_ref, img_1):

    '''
    Uses the calculated Homography matrix to check if the top right and bottom right
    corners of the stitching image falls on to the right or left of top right and bottom
    right of the reference. If it falls on the left, the sitiching image should be on the
    left side of the refernce image.
    '''
    
    ref_rows = img_ref.shape[0]
    ref_cols = img_ref.shape[1]

    stitch_rows = img_1.shape[0]
    stitch_cols = img_1.shape[1]

    # Calculating where the corners of image 1 lands
    hom_ref_topright = np.array([ref_cols, 0, 1])
    hom_ref_botright = np.array([ref_cols, ref_rows, 1])

    hom_1_topright = np.array([stitch_cols, 0, 1])
    hom_1_botright = np.array([stitch_cols, stitch_rows, 1])

    t_hom_1_topright = np.matmul(H, hom_1_topright)
    t_hom_1_botright = np.matmul(H, hom_1_botright)
    t_hom_1_topright /= t_hom_1_topright[2]
    t_hom_1_botright /= t_hom_1_botright[2]

    test_topright = t_hom_1_topright - hom_ref_topright
    test_botright = t_hom_1_botright - hom_ref_botright

    # checking if the stitch images are on the left or right of the reference
    if test_topright[0] > 0 and test_botright[0] > 0:
        return False
    
    elif test_topright[0] < 0 and test_botright[0] < 0:
        return True

    else: 
        print(test_topright[0])
        print(test_botright[0])
        raise AssertionError ("Cannot properly check whether ref img is on left or right")

def blend_images(img_ref, img_1):
    '''
    Blending two images using simple weights
    '''
    
    # blending parameters
    alpha = 0.5
    beta = 1 - alpha

    # Blending Images
    blended1_images = cv.subtract(img_ref, img_1)
    blended2_images = cv.subtract(img_1, img_ref)
    blended_images = blended1_images + blended2_images

    return blended_images

# Overall Function that will be called from Main
def stitch_images(img_ref, img_1):
    '''
    Function takes in two images and returns a stitched image of the two
    '''

    # Number of SIFT features to be detected intially
    n_features = 100

    # Create SIFT Object for the reference image
    sift_ref = cv.SIFT_create()
    sift_ref.setNFeatures(n_features)
    kp_ref, des_ref = sift_ref.detectAndCompute(img_ref, None)

    # Create SIFT Object for the stitching image
    sift_1 = cv.SIFT_create()
    sift_1.setNFeatures(n_features)
    kp_1, des_1 = sift_1.detectAndCompute(img_1, None)

    # Find the good matches according to Lowe's Ratio test
    good_matches = find_good_matches(des_ref, des_1)

    # Get array of ordered match coordinates for kp_ref and kp_1
    kp_ref_coords, kp_1_coords = coordinates_of_matches(good_matches, kp_ref, kp_1)

    # Find Homography and Transform the stiching image to reference coordinates
    H = find_homography_matrix(kp_ref_coords, kp_1_coords)
    img_1_transform = cv.warpPerspective(img_1, H, (img_1.shape[1] + img_ref.shape[1], img_ref.shape[0]))

    # Check where the stitching image is right or left in relation to Original
    left = stitch_pos_checker(H, img_ref, img_1)
    print(left)
    if left == True:
        return stitch_images(img_ref[:, ::-1], img_1[:, ::-1])[:,::-1]

    # Stitching Procedure and Blending
    stitched_image = img_1_transform.copy()
    stitched_image[0:img_ref.shape[0], 0:img_ref.shape[1]] = img_ref

    return stitched_image, img_1_transform


if __name__ == '__main__':
    
    # Reading in the images
    ref_path = './Test_Images/Set_3/scene_ref.jpg'
    stitch1_path = './Test_Images/Set_3/scene_r1.jpg'
    stitch2_path = './Test_Images/Set_3/scene_l1.jpg'

    # BGR scale
    img_ref_bgr = cv.imread(os.path.join(ref_path))
    img_1_bgr = cv.imread(os.path.join(stitch1_path))
    img_2_bgr = cv.imread(os.path.join(stitch2_path))
    # img_ref_bgr = img_ref_bgr[:, ::-1]
    # img_1_bgr = img_1_bgr[:, ::-1]
    img_ref_bgr = cv.resize(img_ref_bgr, (len(img_1_bgr[0,:])//2, len(img_1_bgr[:,0])//2))
    img_1_bgr = cv.resize(img_1_bgr, (len(img_1_bgr[0,:])//2, len(img_1_bgr[:,0])//2))
    img_2_bgr = cv.resize(img_2_bgr, (len(img_2_bgr[0,:])//2, len(img_2_bgr[:,0])//2))

    # RGB scale
    img_ref_rgb = cv.cvtColor(img_ref_bgr, cv.COLOR_BGR2RGB)
    img_1_rgb = cv.cvtColor(img_1_bgr, cv.COLOR_BGR2RGB)
    img_2_rgb = cv.cvtColor(img_2_bgr, cv.COLOR_BGR2RGB)

    # Gray scale
    img_ref = cv.cvtColor(img_ref_bgr, cv.COLOR_BGR2GRAY)
    img_1 = cv.cvtColor(img_1_bgr, cv.COLOR_BGR2GRAY)
    img_2 = cv.cvtColor(img_2_bgr, cv.COLOR_BGR2GRAY)

    # Create the stitched image
    stitched_image, img_1_transfrom = stitch_images(img_ref, img_1)
    # stitched2_image = stitch_images(stitched_image, img_2_rgb)

    # cv.imwrite('stitching_rslt.jpg', stitched2_image)

    # Plotting the images in RGB scale
    fig, axes = plt.subplots(1, 2, figsize=(10,5))

    # axes[0].imshow(img_ref_rgb)
    axes[0].imshow(img_ref, cmap='gray')
    axes[0].set_title('Original Reference Image')

    # axes[1].imshow(img_1_rgb)
    axes[1].imshow(img_1, cmap='gray')
    axes[1].set_title('Image to be Stitched to Original Reference Image')
    plt.show()


    ### Image Blending testing
    
    img_1_mask = img_1_transfrom.copy().astype(float)

    img_ref_test = img_ref.copy().astype(float)
    img_1_test = img_1_transfrom[0:img_ref.shape[0], 0:img_ref.shape[1]].copy().astype(float)

    img_ref_test = img_ref_test
    img_1_test =img_1_test

    img_1_mask[img_1_mask>0] == 1
    img_1_mask = img_1_mask[0:img_ref.shape[0], 0:img_ref.shape[1]]

    w_img_1_mask = cv.GaussianBlur(img_1_mask, (5,5), 4)/255
    new_img = cv.add(img_ref_test*(1-w_img_1_mask), img_1_test*(w_img_1_mask))


    # Plotting the stitched image
    plt.figure(figsize=(10,5))
    plt.imshow(stitched_image, cmap='gray')
    plt.title('Stitched Image')
    plt.show()