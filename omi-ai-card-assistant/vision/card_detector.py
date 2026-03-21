import cv2
import numpy as np

def detect_card(frame):
    """
    Detects rectangular contours representing playing cards.
    Filters by area to ignore noise, and returns the largest 4-point contour.
    """
    # 1. Convert to Grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # 2. Add Gaussian Blur to reduce noise
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # 3. Apply an adaptive threshold to separate cards from background
    # (Cards are typically bright/white against a darker background)
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                 cv2.THRESH_BINARY_INV, 11, 2)
    
    # 4. Find all contours in the thresholded image
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return []
        
    # 5. Sort contours by area in descending order
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    
    cards = []
    
    for contour in contours:
        # Ignore very small contours that are clearly not cards
        area = cv2.contourArea(contour)
        if area < 5000:
            continue
            
        # 6. Approximate the contour to a polygon
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        
        # 7. If our approximated contour has 4 points, we found a card!
        if len(approx) == 4:
            cards.append(approx)
            
    # Return all detected 4-point card contours
    return cards
