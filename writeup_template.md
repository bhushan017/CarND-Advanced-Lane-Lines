**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./camera_cal/corners_found8.jpg "draw corners"
[image2]: ./test_images/tracked0.jpg "Undistored"
[image3]: ./test_images/preprocessImage2.jpg "Binary Example"
[image4]: ./test_images/warped0.jpg "Warp Example"
[image5]: ./test_images/line1.jpg "Fit Visual"
[image6]: ./test_images/line_road0.jpg "Warped Line"
[image7]: ./test_images/line_orginal_image1.jpg "Actualy Image with lane lines"
[image8]: ./test_images/line_orginal_image3.jpg "radius of curvature"
[video1]: ./AdvancedLaneFinding.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points
Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
Writeup / README

1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

Camera Calibration

1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

    The code for this step is located in "./camera_cal/camera_calibrate.py"

    I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection. I am using the chessboard corners detection to draw the chessboard ccorners using drawChessboardCorners.Here's an example of my output for this step.

    ![alt text][image1]


1. Provide an example of a distortion-corrected image.

    I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
    
    
    ![alt text][image2]


2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.


    I used a combination of color and gradient thresholds to generate a binary image (thresholding steps at lines 80 through 96 in `image_gen.py`).  Here's an example of my output for this step.

    ![alt text][image3]

3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

    The code for my perspective transform which appears in lines 117 through 128 in the file `image_gen.py`. Below is the code snippet for the source and destination points for the perspective transform.

```
    # Perspective transformation
    img_size = (img.shape[1],img.shape[0])
    bot_width = .76 # bottom trapizoid height
    mid_width = .08 # middle trapizoid height
    height_pct = .62 # trapizoid height
    bottom_trim = .935 # avoid care hood

    # work on defining perspective transformation area
    img_size = (img.shape[1],img.shape[0])
    src = np.float32([[img.shape[1]*(.5-mid_width/2),img.shape[0]*height_pct],
                      [img.shape[1]*(.5+mid_width/2),img.shape[0]*height_pct],
                      [img.shape[1]*(.5+bot_width/2),img.shape[0]*bottom_trim],
                      [img.shape[1]*(.5-bot_width/2),img.shape[0]*bottom_trim]])
    offset = img_size[0]*.25
    dst = np.float32([[offset, 0], [img_size[0]-offset, 0],[img_size[0]-offset, img_size[1]], [offset ,img_size[1]]])

```
This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 588, 446      | 320, 0        | 
| 691, 446      | 960, 0        |
| 1126, 673     | 960, 720      |
| 153, 673      | 320, 720      |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

    ![alt text][image4]

4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

    I used the line tracker function in the line_tracker.py to generate the line graphically of the input warped image

    ![alt text][image5]
    
   Then I use the above image to draw a smooth right and left line on the warped image.
   
   ![alt text][image6]

   
5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

    I did this in lines 203 through 207 in my code in `image_gen.py`

6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

    I implemented this step in lines 200 through 218 in my code in `image_gen.py` Here is an example of my result on a test image:

    ![alt text][image7]

---

###Video

1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

    Here's a [link to my video result](./AdvancedLaneFinding.mp4)

---

###Discussion

1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

    The code I have submitted works smoothly on the project_video.mp4 video. But not the challenge and harder challenge videos. I have to tune the values for the perspective transform. The line tracker is very basic and need some improvement to track the curves more smoothly. My current pipeline is likey to fail if the are shape curve on the road. 

