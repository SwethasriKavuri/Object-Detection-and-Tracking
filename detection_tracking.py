import os
import sys
import cv2
import matplotlib.pyplot as plt
import numpy as np
from math import cos, sin, sqrt

face_cascade = cv2.CascadeClassifier('/usr/local/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml')

def help_message():
   print("Usage: [Question_Number] [Input_Video] [Output_Directory]")
   print("[Question Number]")
   print("1 Camshift")
   print("2 Particle Filter")
   print("3 Kalman Filter")
   print("4 Optical Flow")
   print("[Input_Video]")
   print("Path to the input video")
   print("[Output_Directory]")
   print("Output directory")
   print("Example usages:")
   print(sys.argv[0] + " 1 " + "02-1.avi " + "./")


def detect_one_face(im):
    gray=cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.2, 3)
    if len(faces) == 0:
        return (0,0,0,0)
    return faces[0]

def hsv_histogram_for_window(frame, window):
    # set up the ROI for tracking
    c,r,w,h = window
    roi = frame[r:r+h, c:c+w]
    hsv_roi =  cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_roi, np.array((0., 60.,32.)), np.array((180.,255.,255.)))
    roi_hist = cv2.calcHist([hsv_roi],[0],mask,[180],[0,180])
    cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)
    return roi_hist


def resample(weights):
    n = len(weights)
    indices = []
    C = [0.] + [sum(weights[:i+1]) for i in range(n)]
    u0, j = np.random.random(), 0
    for u in [(u0+i)/n for i in range(n)]:
      while u > C[j]:
          j+=1
      indices.append(j-1)
    return indices

def camShift_tracker(v, file_name):
    # Open output file
    output_name = sys.argv[3] + file_name
    output = open(output_name,"w")

    frameCounter = 0
    # read first frame
    ret ,frame = v.read()
    if ret == False:
        return

    # detect face in first frame
    c,r,w,h = detect_one_face(frame)

    # Write track point for first frame
    output.write("%d,%d,%d\n" %(0, c+w/2,r+h/2)) # Write as 0,pt_x,pt_y
    frameCounter = frameCounter + 1

    # set the initial tracking window
    track_window = (c,r,w,h)

    # calculate the HSV histogram in the window
    # NOTE: you do not need this in the Kalman, Particle or OF trackers
    roi_hist = hsv_histogram_for_window(frame, (c,r,w,h)) # this is provided for you
    term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )
    # initialize the tracker
    

    # e.g. kf = cv2.KalmanFilter(4,2,0)
    # or: particles = np.ones((n_particles, 2), int) * initial_pos

    while(1):
        ret ,frame = v.read() # read another frame
        if ret == False:
            break

        # perform the tracking
	hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
	dst = cv2.calcBackProject([hsv],[0],roi_hist,[0,180],1)
	ret, track_window = cv2.CamShift(dst, track_window, term_crit)
        # e.g. cv2.meanShift, cv2.CamShift, or kalman.predict(), kalman.correct()
	
	pt = (track_window[0]+track_window[2]/2, track_window[1]+track_window[3]/2)
        cv2.circle(frame, (int(pt[0]), int(pt[1])), 5, (0,255,0), -1)
        output.write("%d,%d,%d\n" % (frameCounter, pt[0], pt[1])) # Write as frame_index,pt_x,pt_y
	pts = cv2.boxPoints(ret)
        pts = np.int0(pts)
        camSh = cv2.polylines(frame,[pts],True, 255,2)
        cv2.imshow('camSh',camSh)
	k = cv2.waitKey(60) & 0xff 
        
	
	
	# use the tracking result to get the tracking point (pt):
        # if you track a rect (e.g. face detector) take the mid point,
        # if you track particles - take the weighted average
        # the Kalman filter already has the tracking point in the state vector

        # write the result to the output file
       
	#output.write("%d,%d,%d\n" % (frameCounter, pt[0], pt[1]) # Write as frame_index,pt_x,pt_y
        frameCounter = frameCounter + 1

    output.close()


def kalman_filter(v, file_name):
    # Open output file
    output_name = sys.argv[3] + file_name
    output = open(output_name,"w")

    frameCounter = 0
    # read first frame
    ret ,frame = v.read()
    if ret == False:
        return

    # detect face in first frame
    c,r,w,h = detect_one_face(frame)

    # set the initial tracking window
    track_window = (c,r,w,h)

    # calculate the HSV histogram in the window
    # NOTE: you do not need this in the Kalman, Particle or OF trackers
    roi_hist = hsv_histogram_for_window(frame, (c,r,w,h)) # this is provided for y   
    cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)
   
    state = np.array([c+w/2,r+h/2,0,0], dtype='float64') # initial position
    kalman = cv2.KalmanFilter(4,2,0)
    kalman.transitionMatrix = np.array([[1., 0., .1, 0.],
                                    [0., 1., 0., .1],
                                    [0., 0., 1., 0.],
                                    [0., 0., 0., 1.]])
    kalman.measurementMatrix = 1. * np.eye(2, 4)
    kalman.processNoiseCov = 1e-5 * np.eye(4, 4)
    kalman.measurementNoiseCov = 1e-3 * np.eye(2, 2)
    kalman.errorCovPost = 1e-1 * np.eye(4, 4)
    kalman.statePost = state

    # tracking
    prediction = kalman.predict()

    # obtain measurement
    measurement = np.dot(kalman.measurementNoiseCov, np.random.randn(2, 1))
    measurement = np.dot(kalman.measurementMatrix, state) + measurement.reshape(-1)
    kalman.correct(measurement)
    process_noise = sqrt(kalman.processNoiseCov[0,0]) * np.random.randn(4, 1)
    test = np.dot(kalman.transitionMatrix, state)
    state = np.dot(kalman.transitionMatrix, state) + process_noise.reshape(-1)
    
    pt = (int(state[0]), int(state[1]))
    output.write("%d,%d,%d\n" % (frameCounter, pt[0], pt[1]))	
    frameCounter = frameCounter + 1
    while(1):	
	ret, frame = v.read() # read another frame
        if ret == False:
            break

        # detect face in first frame
        c,r,w,h = detect_one_face(frame)
        # set the initial tracking window
        track_window = (c,r,w,h)

        prediction = kalman.predict()
        if c == 0 and r == 0 and w == 0 and h == 0:
            measurement = np.dot(kalman.measurementNoiseCov, np.random.randn(2, 1))
            measurement = np.dot(kalman.measurementMatrix, state) + measurement.reshape(-1)
            pos = (int(prediction[0]), int(prediction[1]))
        else:
            state = np.array([c+w/2,r+h/2,0,0], dtype='float64') # initial position
            measurement = np.dot(kalman.measurementNoiseCov, np.random.randn(2, 1))
            measurement = np.dot(kalman.measurementMatrix, state) + measurement.reshape(-1)
            posterior = kalman.correct(measurement)
            pos = (int(posterior[0]), int(posterior[1]))

        process_noise = sqrt(kalman.processNoiseCov[0,0]) * np.random.randn(4, 1)
        state = np.dot(kalman.transitionMatrix, state) + process_noise.reshape(-1)

        cv2.circle(frame, pos, 5, (0,0,255), -1)
        cv2.rectangle(frame,(int(pos[0])-w/2, int(pos[1])-h/2),(int(pos[0]+w/2), int(pos[1])+h/2),(0,255,0),3)
        cv2.imshow('Kalman',frame)
        k = cv2.waitKey(60) & 0xff
        #if k == 27:
        #    break
        
        output.write("%d,%d,%d\n" % (frameCounter, pos[0], pos[1])) # Write as frame_index,pt_x,pt_y
        frameCounter = frameCounter + 1

    output.close()


def of_tracker(v, file_name):
    # Open output file
    output_name = sys.argv[3] + file_name
    output = open(output_name,"w")

    frameCounter = 0
    # read first frame
    ret ,frame = v.read()
    if ret == False:
        return

    # params for ShiTomasi corner detection
    feature_params = dict( maxCorners = 100,
                           qualityLevel = 0.3,
                           minDistance = 7,
                           blockSize = 7 )
    # Parameters for lucas kanade optical flow
    lk_params = dict( winSize  = (15, 15),
                      maxLevel = 2,
                      criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    # Create some random colors
    color = np.random.randint(0,255,(100,3))
    # detect face in first frame
    c,r,w,h = detect_one_face(frame)

    old_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # p0 = cv2.goodFeaturesToTrack(old_gray, mask = mask, **feature_params)
    # p0 = np.array([[[c+w/2, r+h/2]], [[c+w/2, r+h/2]]], np.float32)

    p0 = []
    num_points = 5
    for i in range(num_points):
        temp = np.random.rand(1,2)*(w/2,h/2) + (c+w/4,r+h/4)
        p0.append(temp)

    p0.append([[c+w/2, r+h/2]])
    p0 = np.array(p0).astype('float32')
    # Create a mask image for drawing purposes
    #print p0.shape, p0
    mask = np.zeros_like(frame)

    pt_wt_avg = (int(c+w/2), int(r+h/2))
    pt = (int(pt_wt_avg[0]), int(pt_wt_avg[1]))
    output.write("%d,%d,%d\n" % (frameCounter, pt[0], pt[1])) # Write as frame_index,pt_x,pt_y
    frameCounter = frameCounter + 1

    while(1):
        ret ,frame = v.read() # read another frame
        if ret == False:
            break

        # detect face in first frame
        c,r,w,h = detect_one_face(frame)

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # calculate optical flow
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
        # Select good points
        good_new = p1[st==1]
        good_old = p0[st==1]

        if c == 0 & w == 0 & r == 0 & h == 0:
            pt_wt_avg = sum(zip(*p0)[0])/len(zip(*p0)[0])
            cv2.circle(frame, (int(pt_wt_avg[0]), int(pt_wt_avg[1])), 5, (0,0,255), -1)
	    cv2.imshow('frame', frame)
        else:
            pt_wt_avg = (int(c+w/2), int(r+h/2))
            cv2.imshow('frame', frame)

        # k = cv2.waitKey(30) & 0xff
        #if k == 27:
        #    break
        # Now update the previous frame and previous points
        old_gray = frame_gray.copy()
        p0 = good_new.reshape(-1,1,2)

        k = cv2.waitKey(60) & 0xff
        if k == 27:
            break

        pt = (int(pt_wt_avg[0]), int(pt_wt_avg[1]))
        output.write("%d,%d,%d\n" % (frameCounter, pt[0], pt[1])) # Write as frame_index,pt_x,pt_y
        frameCounter = frameCounter + 1

    output.close()




def particle_filter(v, file_name): 

    # Open output file
    output_name = sys.argv[3] + file_name
    output = open(output_name,"w")

    frameCounter = 0
    # read first frame
    ret ,frame = v.read()
    if ret == False:
        return

    # detect face in first frame
    c,r,w,h = detect_one_face(frame)

    # Write track point for first frame
    output.write("%d,%d,%d\n" %(0, c+w/2,r+h/2)) # Write as 0,pt_x,pt_y
    frameCounter = frameCounter + 1

    # set the initial tracking window
    track_window = (c,r,w,h)

    # calculate the HSV histogram in the window
    # a function that, given a particle position, will return the particle's "fitness"
    roi_hist = hsv_histogram_for_window(frame, (c,r,w,h))
    def particleevaluator(back_proj, particle):
        return back_proj[particle[1],particle[0]]

    # hist_bp: obtain using cv2.calcBackProject and the HSV histogram
    # c,r,w,h: obtain using detect_one_face()
    n_particles = 400

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hist_bp = cv2.calcBackProject([hsv],[0],roi_hist,[0,180],1)	
    init_pos = np.array([c + w/2.0,r + h/2.0], int) # Initial position
    particles = np.ones((n_particles, 2), int) * init_pos # Init particles to init position
    f0 = particleevaluator(hist_bp,init_pos) * np.ones(n_particles) # Evaluate appearance model
    weights = np.ones(n_particles) / n_particles   # weights are uniform (at first)
    
    stepsize = 15
    # e.g. kf = cv2.KalmanFilter(4,2,0)
    # or: particles = np.ones((n_particles, 2), int) * initial_pos
    while(1):
	ret ,frame = v.read() # read another frame
        if ret == False:
            break
	w_prev = w
        h_prev = h
        # detect face in first frame
        c,r,w,h = detect_one_face(frame)
        # set the initial tracking window
        track_window = (c,r,w,h)

        if c == 0 and r == 0 and w == 0 and h == 0:
            w = w_prev
            h = h_prev

        #roi_hist = hsv_histogram_for_window(frame, (c,r,w,h))
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hist_bp = cv2.calcBackProject([hsv],[0],roi_hist,[0,180],1)

        # Particle motion model: uniform step (TODO: find a better motion model)
        np.add(particles, np.random.uniform(-stepsize, stepsize, particles.shape), out=particles, casting="unsafe")

        (im_h, im_w) = frame.shape[:2]
        # Clip out-of-bounds particles
        particles = particles.clip(np.zeros(2), np.array((frame.shape[1],frame.shape[0]))-1).astype(int)

        f = particleevaluator(hist_bp, particles.T) # Evaluate particlesi
        weights = np.float32(f.clip(1))             # Weight ~ histogram response
        weights /= np.sum(weights)                  # Normalize w
        pos = np.sum(particles.T * weights, axis=1).astype(int) # expected position: weighted average
        
        if 1. / np.sum(weights**2) < n_particles / 2.: # If particle cloud degenerate:
            particles = particles[resample(weights),:]  # Resample particles according to weights

        for i in range(len(particles)):
            cv2.circle(frame, (int(particles[i][0]), int(particles[i][1])), 1, (0,0,255), -1)
        #cv2.circle(frame, (int(pos[0]), int(pos[1])), 5, (0,255,0), -1)
        #cv2.rectangle(frame,(int(pos[0])-w/2, int(pos[1])-h/2),(int(pos[0]+w/2), int(pos[1])+h/2),(0,255,0),3)
        cv2.imshow('img2',frame)
        k = cv2.waitKey(60) & 0xff
        if k == 27:
            break

        output.write("%d,%d,%d\n" % (frameCounter, pos[0], pos[1])) # Write as frame_index,pt_x,pt_y
        frameCounter = frameCounter + 1

    output.close()

if __name__ == '__main__':
    question_number = -1
   
    # Validate the input arguments
    if (len(sys.argv) != 4):
        help_message()
        sys.exit()
    else: 
        question_number = int(sys.argv[1])
        if (question_number > 4 or question_number < 1):
            print("Input parameters out of bound ...")
            sys.exit()

    # read video file
    video = cv2.VideoCapture(sys.argv[2]);

    if (question_number == 1):
        camShift_tracker(video, "output_camshift.txt")
    elif (question_number == 2):
         particle_filter(video, "output_particle.txt")
    elif (question_number == 3):
        kalman_filter(video, "output_kalman.txt")
    elif (question_number == 4):
        of_tracker(video, "output_of.txt")
    
    video.release()
'''
For Kalman Filter:

# --- init

state = np.array([c+w/2,r+h/2,0,0], dtype='float64') # initial position
kalman.transitionMatrix = np.array([[1., 0., .1, 0.],
                                    [0., 1., 0., .1],
                                    [0., 0., 1., 0.],
                                    [0., 0., 0., 1.]])
kalman.measurementMatrix = 1. * np.eye(2, 4)
kalman.processNoiseCov = 1e-5 * np.eye(4, 4)
kalman.measurementNoiseCov = 1e-3 * np.eye(2, 2)
kalman.errorCovPost = 1e-1 * np.eye(4, 4)
kalman.statePost = state


# --- tracking

prediction = kalman.predict()

# ...
# obtain measurement

if measurement_valid: # e.g. face found
    # ...
    posterior = kalman.correct(measurement)

# use prediction or posterior as your tracking result
'''

'''
For Particle Filter:

# --- init

# a function that, given a particle position, will return the particle's "fitness"
def particleevaluator(back_proj, particle):
    return back_proj[particle[1],particle[0]]

# hist_bp: obtain using cv2.calcBackProject and the HSV histogram
# c,r,w,h: obtain using detect_one_face()
n_particles = 200

init_pos = np.array([c + w/2.0,r + h/2.0], int) # Initial position
particles = np.ones((n_particles, 2), int) * init_pos # Init particles to init position
f0 = particleevaluator(hist_bp, pos) * np.ones(n_particles) # Evaluate appearance model
weights = np.ones(n_particles) / n_particles   # weights are uniform (at first)


# --- tracking

# Particle motion model: uniform step (TODO: find a better motion model)
np.add(particles, np.random.uniform(-stepsize, stepsize, particles.shape), out=particles, casting="unsafe")

# Clip out-of-bounds particles
particles = particles.clip(np.zeros(2), np.array((im_w,im_h))-1).astype(int)

f = particleevaluator(hist_bp, particles.T) # Evaluate particles
weights = np.float32(f.clip(1))             # Weight ~ histogram response
weights /= np.sum(weights)                  # Normalize w
pos = np.sum(particles.T * weights, axis=1).astype(int) # expected position: weighted average

if 1. / np.sum(weights**2) < n_particles / 2.: # If particle cloud degenerate:
    particles = particles[resample(weights),:]  # Resample particles according to weights
# resample() function is provided for you
'''	
