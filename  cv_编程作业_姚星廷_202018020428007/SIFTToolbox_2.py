import cv2
import numpy as np
from matplotlib import pyplot as plt

#part III: description
def KP2Des(keypoints, gaussian_images):
    descriptors = []
    for keypoint in keypoints:
    # restore gaussImage(octave, layer, scale) for the key point from the OpenCV object KeyPoint
        octave = keypoint.octave & 255
        layer = (keypoint.octave >> 8) & 255
        if octave >= 128:
            octave = octave | -128
        scale = 1 / np.float32(1 << octave) if octave >= 0 else np.float32(1 << -octave)
        gaussian_image = gaussian_images[octave + 1, layer]

        cosTheta = np.cos(np.deg2rad(360. - keypoint.angle))# for the following rotation
        sinTheta = np.sin(np.deg2rad(360. - keypoint.angle))
        resPos = np.round(scale * np.array(keypoint.pt)).astype('int')
        desHist = np.zeros((4, 4, 8))
        for X in range(4):
            for Y in range(4):
                for i in range(4):
                    for j in range(4):
                        rotation_matrix = np.array([[cosTheta, -sinTheta], [sinTheta, cosTheta]])
                        RealXDiff, RealYDiff = rotation_matrix.dot(np.array([i + 4 * (X - 2), j + 4 * (Y - 2)]).T)
                        RealX = np.round(resPos[0] + RealXDiff).astype(int)
                        RealY = np.round(resPos[1] + RealYDiff).astype(int)
                        # in case the real axis is outside the gaussian_images
                        if RealX >= 5 and RealX <= gaussian_image.shape[1] - 5 and RealY >= 5 and RealY <= gaussian_image.shape[0] - 5:
                            diffX = gaussian_image[RealY, RealX + 1] - gaussian_image[RealY, RealX - 1]
                            diffY = gaussian_image[RealY - 1, RealX] - gaussian_image[RealY + 1, RealX]
                            weight = np.exp(- 1. / 8. * ((RealXDiff / 8) ** 2 + (RealYDiff / 8) ** 2))
                            Mag = np.sqrt(diffX ** 2 + diffY ** 2) * weight
                            Dir = np.rad2deg(np.arctan2(diffY, diffX)) % 360.

                            DirBin = ((Dir - keypoint.angle) * (8. / 360.)) % 8.
                            DirBinFloor = np.floor(DirBin)
                            DirBinDiff = DirBin - DirBinFloor
                            mag1 = Mag * DirBinDiff
                            mag2 = Mag * (1 - DirBinDiff)
                            desHist[X, Y, DirBinFloor.astype(int)] += mag1
                            if DirBinFloor.astype(int) == 7:
                                a = 0
                            else:
                                a = DirBinFloor.astype(int) + 1
                            desHist[X, Y, a] += mag2

        descriptor_vector = desHist.flatten()  # Remove histogram borders
        # Threshold and normalize descriptor_vector
        threshold = np.linalg.norm(descriptor_vector) * 0.2
        descriptor_vector[descriptor_vector > threshold] = threshold
        descriptor_vector /= max(np.linalg.norm(descriptor_vector), 1e-7)
        # Multiply by 512, round, and saturate between 0 and 255 to convert from float32 to unsigned char (OpenCV convention)
        descriptor_vector = np.round(512 * descriptor_vector)
        descriptor_vector[descriptor_vector < 0] = 0
        descriptor_vector[descriptor_vector > 255] = 255
        descriptors.append(descriptor_vector)

    print('descriptors computed')
    return np.array(descriptors, dtype='float32')


def FlannPolt(img1, kp1, des1, img2, kp2, des2):
    # Initialize and use FLANN
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    # Lowe's ratio test
    good = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good.append(m)

    if len(good) > 10:
        # Estimate homography between template and scene
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        M = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)[0]

        # Draw detected template in scene image
        h, w = img1.shape
        pts = np.float32([[0, 0],
                          [0, h - 1],
                          [w - 1, h - 1],
                          [w - 1, 0]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, M)

        img2 = cv2.polylines(img2, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)

        h1, w1 = img1.shape
        h2, w2 = img2.shape
        nWidth = w1 + w2
        nHeight = max(h1, h2)
        hdif = int((h2 - h1) / 2)
        newimg = np.zeros((nHeight, nWidth, 3), np.uint8)

        for i in range(3):
            newimg[hdif:hdif + h1, :w1, i] = img1
            newimg[:h2, w1:w1 + w2, i] = img2

        # Draw SIFT keypoint matches
        for m in good:
            pt1 = (int(kp1[m.queryIdx].pt[0]), int(kp1[m.queryIdx].pt[1] + hdif))
            pt2 = (int(kp2[m.trainIdx].pt[0] + w1), int(kp2[m.trainIdx].pt[1]))
            cv2.line(newimg, pt1, pt2, (255, 0, 0))

        plt.imshow(newimg)
        plt.show()
        print("the number of the matched Keypoint pairs:%d" %len(good) )
    else:
        print("Not enough matches are found - %d/%d" % (len(good),10))


