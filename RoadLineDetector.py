
import numpy as np
import cv2

def canny(image):
  gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
  blur_image = cv2.GaussianBlur(gray_image, (5,5), 0) # Reduce noise
  gradient_image = cv2.Canny(blur_image, 50, 150)
  return gradient_image
  
# Region of interest -> Area in which to detect road lane lines
def region_of_interest(image):
  height = image.shape[0]
  polygons = np.array([
    [(200, height), (1100, height), (550, 250)]
    ]) # Vertices of polygons. (There is only one polygon)
  mask = np.zeros_like(image) 
  cv2.fillPoly(mask, polygons, color=255) 
  masked_image = cv2.bitwise_and(image, mask)
  return masked_image
  
def display_lines(image, lines):
  line_image = np.zeros_like(image) 
  if lines is not None:
    for line in lines:
      x1, y1, x2, y2 = line
      cv2.line(line_image, (x1,y1), (x2,y2), color=(255,0,0), thickness=10) # Draw a line connecting two points
  return line_image

def average_slope_intercept(image, lines):
  left_fit = [] # Coordinates of the left line
  right_fit = [] # Coordinates of the right line

  # Group all lines into right_fit or left_fit.
  for line in lines:
    x1, y1, x2, y2 = line.reshape(4)
    parameters = np.polyfit((x1, x2), (y1, y2), 1) 
    slope = parameters[0]
    intercept = parameters[1] 
    if slope < 0:
      left_fit.append((slope, intercept))
    else:
      right_fit.append((slope, intercept))
    
  if left_fit: # If left_fit IS NOT empty
    left_fit_average = np.average(left_fit, axis=0) # Obtain a single line for the left line of the road lane
    # Coordinates to place it in the image
    left_line = make_coordinates(image, left_fit_average)
  else:
    left_line = np.array([0, 0, 0, 0]) # There is no left line
    
  if right_fit: # if right_fit IS NOT empty
    right_fit_average = np.average(right_fit, axis=0) # Obtain a single line for the right line of the road lane
    # Coordinates to place it in the image
    right_line = make_coordinates(image, right_fit_average)
  else:
    right_line = np.array([0, 0, 0, 0]) # There is no right line
    
  return np.array([left_line, right_line])

def make_coordinates(image, line_parameters):
  slope, intercept = line_parameters
  y1 = image.shape[0]
  y2 = int(y1*(3/5))
  x1 = int((y1 - intercept)/slope)
  x2 = int((y2 - intercept)/slope)
  return np.array([x1, y1, x2, y2])


image = cv2.imread("road.png") 
canny_image = canny(image)
cropped_image = region_of_interest(canny_image)
lines = cv2.HoughLinesP(cropped_image, rho=2, theta=np.pi/180, threshold=100, lines=np.array([]), minLineLength=40, maxLineGap=5) # Detect Lines
averaged_lines = average_slope_intercept(image, lines) # Display only one line instead of multiple lines overlapping
      
line_image = display_lines(image, averaged_lines)
combo_image = cv2.addWeighted(src1=image, alpha=0.8, src2=line_image, beta=1, gamma=1)

cv2.imwrite("output.png", combo_image)
cv2.imshow("Output", combo_image)
cv2.waitKey(0)
cv2.destroyAllWindows()