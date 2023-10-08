#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  6 09:33:12 2022

@author: nastysushi
"""


import numpy as np
import cv2 
import dlib
from math import hypot


cap = cv2.VideoCapture('assets/footage.mp4')

fps = 50
curr_frame = 0

# Default resolutions of the frame are obtained and system dependent.
# Convert the resolutions from float to integer.
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
    
    
# The code does output .avi because MPEG-4 codec gave troubles on mac, but the resulting video is
# converted and downsampled to a MPEG-4 format.
# Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
out = cv2.VideoWriter('outpy.avi',cv2.VideoWriter_fourcc('M','J','P','G'), fps, (frame_width,frame_height))

detector = dlib.get_frontal_face_detector()


def verify_alpha_channel(frame):
      try:
          frame.shape[3] # 4th position
      except IndexError:
          frame = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
      return frame
  

def apply_color_overlay(frame, intensity=0.3, blue = 0, green = 0, red = 0):
     frame = verify_alpha_channel(frame)
     frame_h, frame_w, frame_c = frame.shape
     color_bgra = (blue, green, red, 1)
     overlay = np.full((frame_h, frame_w, 4), color_bgra, dtype='uint8')
     cv2.addWeighted(overlay, intensity, frame, 1.0, 0, frame)
     frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
     return frame
 


def grayscale():
    """
    Turns video to grayscale. Convertion back to BGR to make grayscale
    export possible.

    """
    mask = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    
    return mask



def gaussianBlur():
    """
    Gaussian filter removes high-frequency components 
    from the image, images become more smooth.
    
    """
    val = round((curr_frame-200)/10, 1)
    mask = cv2.GaussianBlur(frame, (9,9), val)

    return mask


def bilateralBlur(x):
    """
    Bilateral filter does not avarege across edges.
    Weighted by spatial distance and intensity difference.
    
    """
    mask = cv2.bilateralFilter(frame,x,75,75)
    return mask    

 
def grabtest():
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)  
    
    lower_yellow = np.array([22, 93, 0])
    upper_yellow = np.array([45, 255, 255])
    
    kernel = np.ones((10,10), np.uint8)
    kernel2 = np.ones((30,30), np.uint8)
    
    erode = cv2.erode(hsv_frame, kernel, iterations=1) 
    dilate = cv2.dilate(hsv_frame, kernel, iterations=1)
    closing = cv2.morphologyEx(hsv_frame, cv2.MORPH_CLOSE, kernel2)
    opening = cv2.morphologyEx(hsv_frame, cv2.MORPH_OPEN, kernel2)
    
    
    # compute difference
    difference = cv2.subtract(dilate, erode)
    
    
    Conv_hsv_Gray = cv2.cvtColor(difference, cv2.COLOR_BGR2GRAY)
    
    ret, mask = cv2.threshold(Conv_hsv_Gray, 50, 255, cv2.THRESH_BINARY_INV)
    
    difference[mask != 255] = [0, 0, 255]
    difference[mask == 255] = [0, 0, 0]
    
    # add the red mask to the images to make the differences obvious
    #erode[mask != 255] = [0, 0, 255]
    #dilate[mask != 255] = [0, 0, 255]
    
    #img2 = cv2.bitwise_and(difference,difference,mask = mask)
    
    mask_fund = cv2.inRange(dilate, lower_yellow, upper_yellow)
    mask_fund = cv2.cvtColor(mask_fund, cv2.COLOR_BGR2RGB)
    
    
    
    res = cv2.add(mask_fund, difference)

    
    
    #return difference
    return res

def grabObjectHSV(morphOp, spectrum):
    """
    Grabs an object in RGB and HSV color space. 
    Show binary frames with the foreground object 
    in white and background in black.

    """
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)   

    lower_yellow = np.array([22, 93, 0])
    upper_yellow = np.array([45, 255, 255])
    
       
    """
    MORPHOLOGICAL OPERATIONS
    
    ----
    
    Erosion: 
        It is useful for removing small white noises.
        Used to detach two connected objects etc.
        
    Dilation:
        In cases like noise removal, erosion is followed by dilation. Because, erosion removes white noises, but it also shrinks our object. So we dilate it. Since noise is gone, they won’t come back, but our object area increases.
        It is also useful in joining broken parts of an object.
        
    Closing:
        A dilation followed by an erosion 
        (i.e. the reverse of the operations for an opening). 
        closing tends to close gaps in the image.
    Closing:
        Opposite of closing (erosion followed by dilation)
    """
    
    if(morphOp == 'erosion'):
        kernel = np.ones((10,10), np.uint8)
        morph_op = cv2.erode(hsv_frame, kernel, iterations=1) 
        #print(morph_op)
       # print(morph_op) -> 0 of 255 - RED
    elif(morphOp == 'dilation'):
        kernel = np.ones((10,10), np.uint8)
        morph_op = cv2.dilate(hsv_frame, kernel, iterations=3)
        #print(morph_op) -> 0 of 255 - GREEN ETC EN OVERLAPPING IS WIT!! *[255, 0,0] doen etc
        
    elif(morphOp == 'closing'):
        kernel = np.ones((30,30), np.uint8)
        morph_op = cv2.morphologyEx(hsv_frame, cv2.MORPH_CLOSE, kernel)
    elif(morphOp == 'opening'):
        kernel = np.ones((30,30), np.uint8)
        morph_op = cv2.morphologyEx(hsv_frame, cv2.MORPH_OPEN, kernel)



    mask = cv2.inRange(morph_op, lower_yellow, upper_yellow)
    
     
     # coversion to make export to video possible
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)

    if(morphOp == 'erosion'):
        for m in mask:
            for i in range(len(m)):
                # Because from gray scale, so will be white
                if(m[i][0] == 255):
                    m[i][0] = 0
                    m[i][1] = 0
    elif(morphOp == 'dilation'):
        for m in mask:
             for i in range(len(m)):
                 # Because from gray scale, so will be white
                 if(m[i][0] == 255):
                     m[i][1] = 0
                     m[i][2] = 0
            
            
   # print(mask)
  
    return mask


def sobel(detection):
    """
    Sobel filter is a 1D discrete derivative filter for edge detection.
    Change in color intensity to detect edges by taking first derivative.

    """

	# Convert to graycsale
    img_gray = grayscale()

    # Blur the image for better edge detection
    img_blur = cv2.GaussianBlur(img_gray, (3,3), 0)
    
    # sobelx - base = not blurred img
    if(detection == 'sobelx_noblur'):
        return cv2.Sobel(frame, cv2.CV_8U, 1,0, ksize=3)
    elif(detection == 'sobelx'):
        return cv2.Sobel(img_blur, cv2.CV_8U, 1,0, ksize=3)
    elif(detection == 'sobely'):
        return cv2.Sobel(img_blur, cv2.CV_8U, 0,1, ksize=3)
    elif(detection == 'sobelxy_noblur'):
        return cv2.Sobel(frame, cv2.CV_8U, dx=1, dy=1, ksize=5)
    elif(detection == 'sobelxy'):
        return cv2.Sobel(img_blur, cv2.CV_8U, dx=1, dy=1, ksize=5)
    else:
        print('forgot to add a detection parameter')


def houghTransform(dp, mindst):
    """
    Hough transform is a feature extraction method for detecting simple 
    shapes such as circles, lines etc in an image.

    """
    gray =cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Finite difference filters respond strongly to noise, so smoothing edges,
    # by forcing pixels different from their nieghbors to look more like neighbors, helps forecome the problem
    img = cv2.medianBlur(gray, 5)
    
    # convert gray back to BGR
    cimg = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    
    circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, dp, mindst, param1=100, param2=30,minRadius=0, maxRadius=0) 
    
    if circles is not None:
        circles = np.uint16(np.around(circles))
    
        for i in circles[0,:]:
            #draw the outer circle
            cv2.circle(cimg, (i[0], i[1]), i[2],(0,255,0), 3)
            #draw the center of the circle
            cv2.circle(cimg, (i[0], i[1]), 2, (0,0,255), 5)
        
    return cimg


def objectDetection(part):
    """
    template matching
    
    """
    gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    template = cv2.imread('assets/duck.png', cv2.IMREAD_GRAYSCALE)
    
    h, w = template.shape
    # returns chances (i.e. intensity values proportional to the likelihood) in black and white
    result = cv2.matchTemplate(gray_img, template, cv2.TM_CCOEFF_NORMED)
    
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

    bottom_right = (max_loc[0] + w, max_loc[1] + h)
    
    image = cv2.rectangle(frame, max_loc, bottom_right, (255, 0, 255), 2)
        
    template = cv2.cvtColor(template, cv2.COLOR_GRAY2BGR) 
    gray_img = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR) 
    
    result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR) 
    result = result - result.min() # between 0 and 8674
    result = result / result.max() * 255
    
    arr = np.uint8(result)
    # result gives other resolutions (result of substraction between original image - template)
    # so have to be resized to keep same sizes
    resized = cv2.resize(arr, (1600,900), interpolation = cv2.INTER_AREA)

    if(part == 'rect'):
        return image
    elif(part == 'gray'):
        return resized
    else:
        return print('something went wrong')


def instafilter(param):
    """
    This filter recognizes faces by using a shape predictor (face)
    and replaces the original face with Richard Feynman's

    """
    face_img = cv2.imread("assets/insta/richard2.png")
    
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
    faces = detector(frame)
    predictor= dlib.shape_predictor("assets/insta/shape_predictor_68_face_landmarks.dat")
    
    for face in faces:
        landmarks = predictor(gray_frame, face)
        
        left_head = (landmarks.part(0).x, landmarks.part(0).y)
        right_head = (landmarks.part(16).x, landmarks.part(16).y)
        center_head = (landmarks.part(30).x, landmarks.part(30).y)
        
        # hypot : library to calculate distance between two points
        head_width = int(hypot(left_head[0] - right_head[0], left_head[1] - right_head[1]) + 130)
        head_height = int(head_width * 1.25)
        
        top_left = (int(center_head[0] - head_width / 2),
                    int(center_head[1] - head_height / 2))
            
        bottom_right = (int(center_head[0] + head_width / 2),
                        int(center_head[1] + head_height / 2))
        
        if(param == 'rect'):
            cv2.rectangle(frame, top_left, bottom_right,(0, 255, 0), 2 )
        elif(param == 'face'):
            head_img = cv2.resize(face_img, (head_width, head_height))
            head_img_gray = cv2.cvtColor(head_img, cv2.COLOR_BGR2GRAY)
            
            _, head_mask = cv2.threshold(head_img_gray, 5, 255, cv2.THRESH_BINARY_INV)
            
            # cuts out head area
            head_area = frame[top_left[1]:top_left[1] + head_height, top_left[0]:top_left[0] + head_width]
            # mask for transparency
            head_area_no_head = cv2.bitwise_and(head_area, head_area, mask=head_mask)
            final_head = cv2.add(head_area_no_head, head_img)
            
            frame[top_left[1]:top_left[1] + head_height, 
                        top_left[0]:top_left[0] + head_width] = final_head
        
    return frame
    
    
def sharpen():   
    """
    Sharpening filters are used to enhance the edges of objects 
    and adjust the contrast and the shade characteristics.

    """
    sharpening_filter = np.array([[-1, -1, -1],
                                  [-1, 9, -1],
                                  [-1, -1, -1]])
    
    curr_filter = cv2.filter2D(frame, -1, sharpening_filter)

    return curr_filter 


def sepia(curr_filter, intensity=0.5):
    """
    Sepia effect is one of the most used filter when editing images.
    It adds a warm brown tone to the pictures.

    """
    blue = 20
    green = 66 
    red = 112
    curr_filter = apply_color_overlay(frame, 
                                       intensity=intensity, 
                                       blue=blue, green=green, red=red)
     
    return curr_filter

    
def videoPartOne(fps, curr_frame):
    ## color to grayscale several times (0-4s) 
    if(fps < curr_frame <= fps*2 or fps*3 < curr_frame <= fps*4):
        curr_filter = grayscale()
        labeled = cv2.putText(img=curr_filter, text='grayscale', org=(150, 250), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=3, color=(255, 255, 255),thickness=3)    
    elif(fps*2 < curr_frame <= fps*3):
        curr_filter = frame
        labeled = cv2.putText(img=curr_filter, text='normal frame', org=(150, 250), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=3, color=(255, 255, 255),thickness=3)    
    #start: 0, 200, 800   
    
    ## Blur (4-12s) 
    # gaussian blur (frame 200-400)
    elif(fps*4 < curr_frame <= fps*8):
        val = round((curr_frame-200)/10, 1)
        curr_filter = cv2.GaussianBlur(frame, (9,9), val)
        labeled = cv2.putText(img=curr_filter, text='Gaussian blur - smooth everything', org=(150, 250), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=2, color=(255, 255, 255),thickness=3)    

    # Bi-lateral filter 
        #curr_frame: start: 400 - 800
        # highly effective in noise removal while keeping edges sharp. But the operation is slower compared to other filters.
        # Mus be an integer between 2 and 36
    elif(fps*8 < curr_frame <= fps*9):
        curr_filter = bilateralBlur(2)
        labeled = cv2.putText(img=curr_filter, text='BilateralBlur (2) - smooth except edges', org=(150, 250), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=2, color=(255, 255, 255),thickness=3)    
    elif(fps*9 < curr_frame <= fps*10):
        curr_filter = bilateralBlur(15)   
        labeled = cv2.putText(img=curr_filter, text='BilateralBlur (15) - smooth except edges', org=(150, 250), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=2, color=(255, 255, 255),thickness=3)    

    elif(fps*10 < curr_frame <= fps*11):
        curr_filter = bilateralBlur(21)
        labeled = cv2.putText(img=curr_filter, text='BilateralBlur (21) - smooth except edges', org=(150, 250), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=2, color=(255, 255, 255),thickness=3)    
    elif(fps*11 < curr_frame <= fps*12):
        curr_filter = bilateralBlur(31) # Time consuming
        labeled = cv2.putText(img=curr_filter, text='BilateralBlur (31) - smooth except edges', org=(150, 250), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=2, color=(255, 255, 255),thickness=3)    
               
    # Black and white ------------------------------------------
    # TODO: CHANGES IN OTHER COLORS
    elif(fps*12 < curr_frame <= fps*14): # 0-2s
        curr_filter = grabObjectHSV('dilation', 'binary')    
        labeled = cv2.putText(img=curr_filter, text='Dilation', org=(150, 250), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=3, color=(255, 255, 255),thickness=3)    
    elif(fps*14 < curr_frame <= fps*16): # 2-4s
        curr_filter = grabObjectHSV('erosion', 'binary')
        labeled = cv2.putText(img=curr_filter, text='Erosion', org=(150, 250), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=3, color=(255, 255, 255),thickness=3)    
    elif(fps*16 < curr_frame <= fps*18):
        curr_filter = grabObjectHSV('closing', 'binary')
        labeled = cv2.putText(img=curr_filter, text='Closing', org=(150, 250), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=3, color=(255, 255, 255),thickness=3)    
    elif(fps*18 < curr_frame <= fps*20):
        curr_filter = grabObjectHSV('opening', 'binary')
        labeled = cv2.putText(img=curr_filter, text='Opening', org=(150, 250), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=3, color=(255, 255, 255),thickness=3)     
   
    ## Default
    else:
        curr_filter = frame
        labeled = cv2.putText(img=curr_filter, text='normal frame', org=(150, 250), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=3, color=(255, 255, 255),thickness=3)    
        
    return labeled



def videoPartTwo(offset, fps, curr_frame):
    # sobel filter
    if(fps*20 < curr_frame <= fps*21):
        curr_filter = sobel('sobelx_noblur')
        labeled = cv2.putText(img=curr_filter, text='Sobel x', org=(150, 250), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=3, color=(255, 255, 255),thickness=3) 
    elif(fps*21 < curr_frame <= fps*22):
        curr_filter = sobel('sobelx')
        labeled = cv2.putText(img=curr_filter, text='Sobel x (blurred base)', org=(150, 250), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=3, color=(255, 255, 255),thickness=3) 
    elif(fps*22 < curr_frame <= fps*23):
        curr_filter = sobel('sobely')
        labeled = cv2.putText(img=curr_filter, text='Sobel y (blurred base)', org=(150, 250), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=3, color=(255, 255, 255),thickness=3) 
    elif(fps*23 < curr_frame <= fps*24):
        curr_filter = sobel('sobelxy')
        labeled = cv2.putText(img=curr_filter, text='Sobel xy (blurred base)', org=(150, 250), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=3, color=(255, 255, 255),thickness=3) 
    elif(fps*24 < curr_frame <= fps*25):
        curr_filter = sobel('sobelxy_noblur')    
        labeled = cv2.putText(img=curr_filter, text='Sobel xy', org=(150, 250), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=3, color=(255, 255, 255),thickness=3) 
  
    # Hough transform
    elif(fps*25 < curr_frame <= fps*27):
        curr_filter = houghTransform(3, 120)
        labeled = cv2.putText(img=curr_filter, text='Hough Transform(dp:3, mindst:120)', org=(150, 250), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=2, color=(255, 255, 255),thickness=3) 

    elif(fps*27 < curr_frame <= fps*29):
        curr_filter = houghTransform(2, 120)
        labeled = cv2.putText(img=curr_filter, text='Hough Transform(dp:2, mindst:120)', org=(150, 250), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=2, color=(255, 255, 255),thickness=3) 
        
    elif(fps*29 < curr_frame <= fps*31):
        curr_filter = houghTransform(1, 120)
        labeled = cv2.putText(img=curr_filter, text='Hough Transform(dp:1, mindst:120)', org=(150, 250), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=2, color=(255, 255, 255),thickness=3) 
    elif(fps*31 < curr_frame <= fps*33):
        curr_filter = houghTransform(1, 50)
        labeled = cv2.putText(img=curr_filter, text='Hough Transform(dp:1, mindst:50)', org=(150, 250), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=2, color=(255, 255, 255),thickness=3) 
    elif(fps*33 < curr_frame <= fps*35):
        curr_filter = houghTransform(1, 20)
        labeled = cv2.putText(img=curr_filter, text='Hough Transform(dp:1, mindst:20)', org=(150, 250), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=2, color=(255, 255, 255),thickness=3) 


    # template matching
    elif(fps*35 < curr_frame <= fps*37):
        curr_filter = objectDetection('rect')
        labeled = cv2.putText(img=curr_filter, text='Template matching', org=(150, 250), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=3, color=(255, 255, 255),thickness=3) 
    elif(fps*37 < curr_frame <= fps*40):
        curr_filter = objectDetection('gray')
        labeled = cv2.putText(img=curr_filter, text='Template matching (method: CCOEFF_NORMED)', org=(150, 250), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=2, color=(255, 255, 255),thickness=3) 
    else:   
        curr_filter = frame
        labeled = cv2.putText(img=curr_filter, text='something went wrong', org=(150, 250), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=3, color=(255, 255, 255),thickness=3) 
        
    return labeled
        
        
def videoPartThree(offset, fps, curr_frame):    
    if(fps*40 < curr_frame <= fps*45):
        curr_filter = sharpen()
        labeled = cv2.putText(img=curr_filter, text='Sharpen filter', org=(150, 250), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=3, color=(255, 255, 255),thickness=3) 
    elif(fps*45 < curr_frame <= fps*50):
        curr_filter = sepia(frame, intensity=0.5)
        labeled = cv2.putText(img=curr_filter, text='Sepia filter', org=(150, 250), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=3, color=(0, 0, 0),thickness=3) 
    elif(fps*50 < curr_frame <= fps*55):
        curr_filter = instafilter('rect')
        labeled = cv2.putText(img=curr_filter, text='Face recognition', org=(150, 250), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=3, color=(255, 255, 255),thickness=3) 
    elif(fps*55 < curr_frame <= fps*60):
        curr_filter = instafilter('face')
        labeled = cv2.putText(img=curr_filter, text='Face replacement', org=(150, 250), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=3, color=(255, 255, 255),thickness=3) 
    else:   
        curr_filter = frame 
        labeled = cv2.putText(img=curr_filter, text='Something went wrong', org=(150, 250), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=3, color=(255, 255, 255),thickness=3) 
        
    return labeled

    
def videoTests(fps, curr_frame):
    # to test seperate filters
    #curr = instafilter('face')
    #if(curr_frame <= fps*20):
    #    curr = grabObjectHSV('erosion', 'binary')
    #else:
    #    curr = grabObjectHSV('dilation', 'binary')
        
    curr = grabtest()

    return curr



def video(fps, curr_frame):
    # 3 parts of the assignment; divided in seperate functions
    if(curr_frame <= fps*20):
        vid = videoPartOne(fps, curr_frame)
    elif(fps*20 < curr_frame <= fps*40):
        vid = videoPartTwo(20, fps, curr_frame)
    elif(fps*40 < curr_frame):
        vid = videoPartThree(40, fps, curr_frame)
    
    return vid

     
while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    #curr_filter = video(fps, curr_frame)
    curr_filter = videoTests(fps, curr_frame)
    
    
    cv2.imshow('filter', curr_filter)
    curr_frame += 1

    # output video
    out.write(curr_filter) 
    
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
