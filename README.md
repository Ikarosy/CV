## Implementation of SIFT in Python

#### Dependencies

Python 3.8	Numpy	OpenCV-Python		matplotlib			   for 'Personal Codes'

Python 3.8	Numpy	opencv_contrib_python 	matplotlib	for 'OpenCV Codes'





#### Usage of python Codes (just open main.py and run)

There are three python scripts: main.py, SIFTToolbox.py, and SIFTToolbox_2. The latter two scripts are cotainers of used functions in SIFT algorithm.  Specifically, SIFTToolbox.py is the function box used for key-point detection while SIFTToolbox_2.py is the one used for descriptor computation.



Run main.py with the `mode ='OpenCV ` commented out, the result will be from this implementation, i.e. 'Personal Codes'. Otherwise, the result will be from an OpenCV exemple, i.e. 'OpenCV Codes', which is used as an comparison.





#### Usage of .exe (failed)

After several attempts, I gave up the idea that I should pack the codes and env. into an .exe file because many fails on running .exe and unsolved errors occurred and incapcity of compressing the .zip below 6MB.





#### Assistances

[SIFT tutorial](https://medium.com/@russmislam/implementing-sift-in-python-a-complete-guide-part-1-306a99b50aa5) 

For details in implementation of SIFT algorthm, this project mainly refers to [OpenCV](https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_feature_homography/py_feature_homography.html) . And for some very trivial parts, I adopted OpenCV codes, e.g. blur, plot, and etc.. 

Paper:[Distinctive Image Features from Scale-Invariant Keypoints", David G. Lowe](https://www.cs.ubc.ca/~lowe/papers/ijcv04.pdf) 





#### Results

OpenCV exemple:

![image-20210416172135425](README(%E8%AF%B4%E6%98%8E+%E5%AE%9E%E9%AA%8C%E7%BB%93%E6%9E%9C).assets/image-20210416172135425.png)

This exemple uses brutal search and displays the top 50 pairs of matches.



This implementation: 

​	![ex1](README(%E8%AF%B4%E6%98%8E+%E5%AE%9E%E9%AA%8C%E7%BB%93%E6%9E%9C).assets/ex1.png)

Obviously , this raw result isn't good at all.  After reading the original paper, I added interpolation to the computation on descriptors.  And, most importantly, weights for each magnitude vectors are used to emphasize more on the magnitude near the key-points.

Meanwhile, some modifications on the adoption of matching algorithm are also adapted.  The following is the final result:

![ex2](README(%E8%AF%B4%E6%98%8E+%E5%AE%9E%E9%AA%8C%E7%BB%93%E6%9E%9C).assets/ex2.png)

Although there are many outer points perceived as matches indicating this implementation is not good enough, useful matches on the object , the book in this case, are more clear which manifests this implementation is complete in revealing the invariance of SIFT in feature representation.

Because of the limitation of houmework file size, I run this implementation on smaller pictures.  The result is as follows, which is more clear, yet the problem of outer points remains.

![e1](README(%E8%AF%B4%E6%98%8E+%E5%AE%9E%E9%AA%8C%E7%BB%93%E6%9E%9C).assets/e1.png)

[SIFT tutorial](https://medium.com/@russmislam/implementing-sift-in-python-a-complete-guide-part-1-306a99b50aa5) ：

![image-20210425214231510](README(%E8%AF%B4%E6%98%8E+%E5%AE%9E%E9%AA%8C%E7%BB%93%E6%9E%9C).assets/image-20210425214231510.png)

![e2](README(%E8%AF%B4%E6%98%8E+%E5%AE%9E%E9%AA%8C%E7%BB%93%E6%9E%9C).assets/e2.png)



#### Defects:

Due to ignorance of many detailed techniques in the original paper, e.g. some useful interpolation algorithms aren't applied, the result doesn't match many other implementations'.  Therefore, this implementation have a lot of problems remaining mending up, e.g., many outer points.



The running time is too long (around 11 mins for a pair of high resolution photos on 2.3 GHz 4 cores Intel Core i7).  Therefore, I used a pair of low resolution photos from OpenCV for consideration of runtime and file size.











 