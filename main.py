import SIFTToolbox as sift
import SIFTToolbox_2 as sift2
import cv2
import numpy as np

mode = 0
# mode = 'OpenCV'
class descriptor(object):
    def __init__(self, image):
        image = image.astype('float32')
        self.GaussPyra = sift.buidupPyramid(image)
        self.DOGPyra = sift.buildupDOG(self.GaussPyra)
        self.Keypoints = sift.findExtrema(self.GaussPyra, self.DOGPyra)
        self.Keypoints_clean = sift.removeDupKP(self.Keypoints)
        self.usefulKP = sift.restorePos(self.Keypoints_clean)
        self.Descriptors = sift2.KP2Des(self.usefulKP, self.GaussPyra)
    def activate(self):

        return self.usefulKP, self.Descriptors

if mode == 'OpenCV':
    img1 = cv2.imread('book1.png', 0)   # queryImage
    img2 = cv2.imread('book2.png', 0)  # trainImage

    sift = cv2.xfeatures2d.SIFT_create()

    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    bf = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_BRUTEFORCE)
    matches = bf.match(des1, des2)

    matches = sorted(matches, key=lambda x: x.distance)
    good_matches = matches[:200]  # 只取前XX个匹配
    result = cv2.drawMatches(img1, kp1, img2, kp2, good_matches,
                             None, matchColor=(0, 0, 255), singlePointColor=(255, 0, 0))  # 只画前XX个匹配
    cv2.imshow('result', result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    img1 = cv2.imread('book1.png', 0)  # queryImage
    img2 = cv2.imread('book2.png', 0)  # trainImage

    kp1, des1 = descriptor(img1).activate()
    kp2, des2 = descriptor(img2).activate()

    sift2.FlannPolt(img1, kp1, des1, img2, kp2, des2)