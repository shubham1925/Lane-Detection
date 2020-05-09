# Lane Detection for Self Driving Cars

## Authors
* **Shubham Sonawane**
* **Prasheel Renkuntla**
* **Raj Prakash Shinde**

 
## Description
Using the concept of hough lines and histogram for lanes, lane detection has been carried out on 2 different datasets.

## Dependencies
* Ubuntu 16
* Python 3.7
* OpenCV 4.2
* Numpy
* copy
* sys
* argparse

## Run
To run the Video enhancement on Night ride video

```
python3.7 P1-video_correction.py
```
To run the lane detection on images, 2(a) Udacity Dataset -
```
python3.7 P2-1-Homography.py
```
To run the lane detection on challenge video, 2(b) KITTI Dataset -
```
python3.7 P2-2-challenge_accepted.py
```
## Results
![Image description](https://github.com/shubham1925/Lane-Detection/blob/master/dataset1.png)

![Image description](https://github.com/shubham1925/Lane-Detection/blob/master/dataset2.png)
## Reference
* https://stackoverflow.com/questions/19890054/how-to-sharpen-an-image-in-opencv
* https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_filtering/py_filtering.html
* https://www.pyimagesearch.com/2015/10/05/opencv-gamma-correction/
* https://medium.com/@rndayala/image-histograms-in-opencv-40ee5969a3b7
* https://answers.opencv.org/question/193276/how-to-change-brightness-of-an-image-increase-or-decrease/
* https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_histograms/py_histogram_equalization/py_histogram_equalization.html
* https://www.programcreek.com/python/example/89353/cv2.createCLAHE
* http://amroamroamro.github.io/mexopencv/opencv/clahe_demo_gui.html
* https://en.wikipedia.org/wiki/Kernel_(image_processing)
* https://answers.opencv.org/question/175912/how-to-display-multiple-images-in-one-window/
