import cv2
import numpy as np
from functools import cmp_to_key

#part I: find the key points
#build up a pyramid
# blur accumulation equation is 'blur_now ** 2 == blur_origin**2 + blur_needed ** 2'
def buidupPyramid(image, sigma=1.6, blur=0.5, S=3):      #image is the input_image, sigma is the scaling factor of Gaussian kernel, blur is the original blur factor, S is the number of scales per octave
    # this part, not necessarily needed, is under the assumption that the original image is pre-blurred and we need to restore the image to retrieve the image with the blur factor we want
    # image = image.astype('float32')
    image = cv2.resize(image, (0, 0), fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
    restoreDsigma = np.sqrt(max((sigma ** 2) - ((2 * blur) ** 2), 0.01))
    processedImage = cv2.GaussianBlur(image, (0, 0), sigmaX=restoreDsigma, sigmaY=restoreDsigma)

    octaveNum = int(np.round(np.log(min(processedImage.shape))/np.log(2)-1))

    k = 2 ** (1. / S)
    gaussianKernel = np.zeros(S+3) #S+3 is the number of images per octave
    for i in range(S+3):
        if i == 0:
            gaussianKernel[i] = sigma
        else:
            gaussianKernel[i] = np.sqrt(((k ** i ) * sigma) ** 2 - ((k ** (i-1)) * sigma) ** 2)

    #with gaussianKernels for each octave settled follows the gaussian_pyramid
    gaussian_pyramid = []
    OctImage = processedImage
    for i in range(octaveNum):
        gaussian_image = []
        gaussian_image.append(OctImage)
        for j in gaussianKernel[1:]:
            OctImage = cv2.GaussianBlur(OctImage, (0, 0), sigmaX=j, sigmaY=j)
            gaussian_image.append(OctImage)
        gaussian_pyramid.append(gaussian_image)
        OctImage0 = gaussian_image[3]
        OctImage = cv2.resize(OctImage0, (int(OctImage0.shape[1] / 2), int(OctImage0.shape[0]/2)), interpolation=cv2.INTER_NEAREST)
    print('finishing buidupPyramid')
    return np.array(gaussian_pyramid, dtype=object)

def buildupDOG(gaussian_pyramid, S=3):
    DOG_pyramid = []
    for OctImages in gaussian_pyramid:
        DOGImages = []
        for i in range(1, S+3):
            DOGImages.append(cv2.subtract(OctImages[i], OctImages[i-1]))
        DOG_pyramid.append(DOGImages)
    print('finishing buildupDOG')
    return np.array(DOG_pyramid, dtype=object)

######################################################################
                                                                    ##
                                                                    ##
######################################################################

#part II: locate key points
def findExtrema(gaussian_pyramid, DOG_pyramid, S=3, sigma=1.6, image_border_width=5, threshold=0.03):
    threshold = np.floor(0.5 * threshold / S * 255)  # from OpenCV implementation
    KeyPoints = []
    cnt0 = 0
    cnt1 = 0
    for OctIndex, DOGImages in enumerate(DOG_pyramid):
        for ImIndex, (image0, image1, image2) in enumerate(zip(DOGImages, DOGImages[1:], DOGImages[2:])):
            for i in range(image_border_width, image0.shape[0] - image_border_width): # eliminate the border effect
                for j in range(image_border_width, image0.shape[1] - image_border_width):
                    if isExtremum(image0[i-1:i+2, j-1:j+2], image1[i-1:i+2, j-1:j+2], image2[i-1:i+2, j-1:j+2], threshold):
                        location = locate(i, j, ImIndex + 1, OctIndex, S, DOGImages, sigma, threshold, image_border_width)
                        cnt0 += 1
                        if location is not None:
                            keypoint, descriptor_ImIndex = location
                            # one keypoint may have multi son-keypoints with different direction
                            KPs = computeMainDir(keypoint, OctIndex, gaussian_pyramid[OctIndex][descriptor_ImIndex])
                            if KPs == []:
                                print('bug')
                            for k in KPs:
                                KeyPoints.append(k)
    print('keypoints found')
    return KeyPoints

def isExtremum(image0, image1, image2, threshold):
#identifies whether the center pixel is extremum
    centerPixel = image1[1, 1]
    if abs(centerPixel) > threshold:
        if centerPixel > 0:
            if centerPixel >= max(np.stack([image1, image2, image0]).flatten()):
                return True
        elif centerPixel < 0:
            if centerPixel <= min(np.stack([image1, image2, image0]).flatten()):
                return True
        else:
            return False
    return False

def locate(i, j, ImIndex, OctIndex, S, DOGImages, sigma, threshold, image_border_width):
    ImShape = DOGImages[0].shape
    OuterExtremum = False
    for Index in range(5):
        # need to convert from uint8 to float32 to compute derivatives and need to rescale pixel values to [0, 1] to apply Lowe's thresholds
        image0, image1, image2 = DOGImages[ImIndex - 1:ImIndex + 2]
        pixel_cube = np.stack([ image0[i - 1:i + 2, j - 1:j + 2],
                                image1[i - 1:i + 2, j - 1:j + 2],
                                image2[i - 1:i + 2, j - 1:j + 2]]).astype('float32') / 255.
        gradient = ComGradient(pixel_cube)
        hessArray = ComHenssianArray(pixel_cube)
        extremum_update = -np.linalg.lstsq(hessArray, gradient, rcond=None)[0]
        if abs(extremum_update[0]) < 0.5 and abs(extremum_update[1]) < 0.5 and abs(extremum_update[2]) < 0.5:
            break
        else:
            j += int(np.round(extremum_update[0]))
            i += int(np.round(extremum_update[1]))
            ImIndex += int(np.round(extremum_update[2]))
        if i < image_border_width or i >= ImShape[0] - image_border_width or j < image_border_width or j >= ImShape[1] - image_border_width or ImIndex < 1 or ImIndex > S:
            OuterExtremum = True
            break

    if OuterExtremum or ImIndex >= 4:
        return None

    UpdatedExtremum = pixel_cube[1, 1, 1] + 0.5 * np.dot(gradient, extremum_update)
    #restore the keypoint's position, determining a descriptor
    if abs(UpdatedExtremum) * S >= 0.03:
        hessian = hessArray[:2, :2]
        trace = np.trace(hessian)
        det = np.linalg.det(hessian)
        if det > 0 and 10 * (trace ** 2) < ((10 + 1) ** 2) * det:
            # using OpenCV KeyPoint object
            keypoint = cv2.KeyPoint()
            keypoint.pt = ((j + extremum_update[0]) * (2 ** OctIndex), (i + extremum_update[1]) * (2 ** OctIndex))
            keypoint.octave = OctIndex + ImIndex * (2 ** 8) + int(np.round((extremum_update[2] + 0.5) * 255)) * (2 ** 16)
            keypoint.size = sigma * (2 ** ((ImIndex + extremum_update[2]) / np.float32(S))) * (2 ** (OctIndex + 1))  # octave_index + 1 because the input image was doubled
            keypoint.response = abs(UpdatedExtremum)
            return keypoint, ImIndex
    return None

def computeMainDir(keypoint, OctIndex, gaussian_image):
    # the following coefficients are used in OpenCV
    scale = 1.5 * keypoint.size / np.float32(2 ** (OctIndex + 1))  # compare with keypoint.size computation in localizeExtremumViaQuadraticFit()
    radius = int(np.round(3 * scale))
    weight_factor = -0.5 / (scale ** 2)
    hist = np.zeros(36)
    #histogram of keypoints and main direction
    for i in range(-radius, radius+1):
        y = int(np.round(keypoint.pt[1] / np.float32(2 ** OctIndex))) + i
        if y > 0 and y < gaussian_image.shape[0] - 1:
            for j in range(-radius, radius+1):
                x = int(np.round(keypoint.pt[0] / np.float32(2 ** OctIndex))) + j
                if x > 0 and x < gaussian_image.shape[1] - 1:
                    dx = (gaussian_image[y, x + 1] - gaussian_image[y, x - 1])
                    dy = (gaussian_image[y - 1, x] - gaussian_image[y + 1, x])
                    theta = int(np.round(np.rad2deg(np.arctan2(dy, dx)) * (36 / 360.))) % 36
                    mxy = np.sqrt((dx ** 2) + (dy ** 2)) * np.exp(weight_factor * (i ** 2 + j ** 2))
                    hist[theta] += mxy
    smooth_histogram = np.zeros(36)
    for n in range(36):
        smooth_histogram[n] = (6 * hist[n] + 4 * (hist[n - 1] + hist[(n + 1) % 36]) + hist[n - 2] + hist[(n + 2) % 36]) / 16.
    maxDirValue = max(smooth_histogram)

    DirPeaks = []
    for i, j in enumerate(smooth_histogram):
        if i < 35:
            if j > smooth_histogram[i - 1] and j > smooth_histogram[i + 1]:
                DirPeaks.append(i)
        else:
            if j > smooth_histogram[i - 1] and j > smooth_histogram[0]:
                DirPeaks.append(i)


    if DirPeaks == []:
        print('BUG', smooth_histogram )


    KP = []
    for i in DirPeaks:
        if smooth_histogram[i] >= 0.8 * maxDirValue:
            NeighborL = smooth_histogram[(i - 1) % 36]
            NeighborR = smooth_histogram[(i + 1) % 36]
            updatedDir = (i + 0.5 * (NeighborL - NeighborR) / (NeighborL - 2 * i + NeighborR)) % 36
            direction = 360. - updatedDir * (360. / 36)
            if abs(360. - direction) < 1e-7:
                direction = 0.
            updatedKP = cv2.KeyPoint(*keypoint.pt, keypoint.size, direction, keypoint.response, keypoint.octave)
            KP.append(updatedKP)
    return KP

#OpenCV has removed some duplicated KeyPoints with the following identical codes
#which are very trivial, so I just copied from https://github.com/rmislam/PythonSIFT/blob/master/pysift.py
def removeDupKP(keypoints):
    if len(keypoints) < 2:
        return keypoints

    keypoints.sort(key = cmp_to_key(compareKeypoints))
    unique_keypoints = [keypoints[0]]

    for next_keypoint in keypoints[1:]:
        last_unique_keypoint = unique_keypoints[-1]
        if last_unique_keypoint.pt[0] != next_keypoint.pt[0] or \
           last_unique_keypoint.pt[1] != next_keypoint.pt[1] or \
           last_unique_keypoint.size != next_keypoint.size or \
           last_unique_keypoint.angle != next_keypoint.angle:
            unique_keypoints.append(next_keypoint)
    return unique_keypoints

def compareKeypoints(keypoint1, keypoint2):
    if keypoint1.pt[0] != keypoint2.pt[0]:
        return keypoint1.pt[0] - keypoint2.pt[0]
    if keypoint1.pt[1] != keypoint2.pt[1]:
        return keypoint1.pt[1] - keypoint2.pt[1]
    if keypoint1.size != keypoint2.size:
        return keypoint2.size - keypoint1.size
    if keypoint1.angle != keypoint2.angle:
        return keypoint1.angle - keypoint2.angle
    if keypoint1.response != keypoint2.response:
        return keypoint2.response - keypoint1.response
    if keypoint1.octave != keypoint2.octave:
        return keypoint2.octave - keypoint1.octave
    return keypoint2.class_id - keypoint1.class_id

##return the key points from gaussian position to the original position
def restorePos(keypoints):
    KP = []
    for k in keypoints:
        k.pt = tuple(0.5 * np.array(k.pt))
        k.size *= 0.5
        k.octave = (k.octave & ~255) | ((k.octave - 1) & 255)
        KP.append(k)
    return KP


def ComGradient(pixel_cube):
    # f'(x) = f(x + 1) -  f(x - 1) / 2
    dx = 0.5 * (pixel_cube[1, 1, 2] - pixel_cube[1, 1, 0])
    dy = 0.5 * (pixel_cube[1, 2, 1] - pixel_cube[1, 0, 1])
    ds = 0.5 * (pixel_cube[2, 1, 1] - pixel_cube[0, 1, 1])
    return np.array([dx, dy, ds])

def ComHenssianArray(pixel_cube):
    #f''(x) = f(x + 1) - 2 * f(x) + f(x - 1)
    # (d^2) f(x, y) / (dx dy) = (f(x + 1, y + 1) - f(x + 1, y - 1) - f(x - 1, y + 1) + f(x - 1, y - 1)) / 4
    center_pixel_value = pixel_cube[1, 1, 1]
    dxx = pixel_cube[1, 1, 2] - 2 * center_pixel_value + pixel_cube[1, 1, 0]
    dyy = pixel_cube[1, 2, 1] - 2 * center_pixel_value + pixel_cube[1, 0, 1]
    dss = pixel_cube[2, 1, 1] - 2 * center_pixel_value + pixel_cube[0, 1, 1]
    dxy = 0.25 * (pixel_cube[1, 2, 2] - pixel_cube[1, 2, 0] - pixel_cube[1, 0, 2] + pixel_cube[1, 0, 0])
    dxs = 0.25 * (pixel_cube[2, 1, 2] - pixel_cube[2, 1, 0] - pixel_cube[0, 1, 2] + pixel_cube[0, 1, 0])
    dys = 0.25 * (pixel_cube[2, 2, 1] - pixel_cube[2, 0, 1] - pixel_cube[0, 2, 1] + pixel_cube[0, 0, 1])
    return np.array([[dxx, dxy, dxs],
                    [dxy, dyy, dys],
                    [dxs, dys, dss]])