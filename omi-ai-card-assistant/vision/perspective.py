import cv2
import numpy as np

def flatten_card(frame, pts):
    """
    Applies a 4-point perspective transform to "flatten" a card into a standard
    top-down 200x300 image.
    """
    # 1. Format the points from the contour into a numpy array of shape (4, 2)
    # The contour points come in as (4, 1, 2), so we reshape them
    pts = pts.reshape(4, 2)
    
    # 2. Add the points together to find the top-left and bottom-right corners.
    # The top-left point will have the smallest sum (x+y).
    # The bottom-right point will have the largest sum.
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    
    # 3. Take the difference between points to find top-right and bottom-left.
    # The top-right will have the smallest difference (y-x).
    # The bottom-left will have the largest difference.
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    
    # 4. Define the dimensions of our new "flattened" card image.
    # Standard playing cards are 2.5 x 3.5 inches. 
    # 200x300 pixels gives a nice clear resolution while keeping a similar ratio.
    maxWidth = 200
    maxHeight = 300
    
    # 5. Define the destination points corresponding to the 4 corners 
    # of our new 200x300 image: (Top-Left, Top-Right, Bottom-Right, Bottom-Left)
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")
        
    # 6. Calculate the perspective transform matrix and warp the image
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(frame, M, (maxWidth, maxHeight))
    
    return warped

def crop_corner(warped_img):
    """
    Takes the flattened 200x300 card image and crops out the top-left corner
    where the Rank and Suit symbols are located.
    """
    # For a 200x300 image, the top-left corner is roughly the top 90 pixels 
    # and the left 35 pixels. (y_start:y_end, x_start:x_end)
    # These dimensions can be tweaked for better accuracy later.
    corner = warped_img[0:95, 0:35]
    
    # Resize it slightly larger just so it's easier to see on the screen
    # (Optional, but helps for debugging)
    corner_zoomed = cv2.resize(corner, (0, 0), fx=3, fy=3)
    
    return corner_zoomed
