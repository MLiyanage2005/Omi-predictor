import cv2
import os
import numpy as np
import glob

# Load templates once when the module is imported to save time
RANK_TEMPLATES = {}
SUIT_TEMPLATES = {}

def load_templates():
    """
    Loads all reference images from templates/ranks and templates/suits.
    Filenames like 'A.jpg', 'K.jpg', 'Hearts.jpg' become the dict keys.
    """
    ranks_path = "templates/ranks/*.jpg"
    suits_path = "templates/suits/*.jpg"
    
    for filepath in glob.glob(ranks_path):
        filename = os.path.basename(filepath)
        name = os.path.splitext(filename)[0]
        template = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        RANK_TEMPLATES[name] = template
        
    for filepath in glob.glob(suits_path):
        filename = os.path.basename(filepath)
        name = os.path.splitext(filename)[0]
        template = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        SUIT_TEMPLATES[name] = template

# Initialize templates
load_templates()

def match_card(corner_crop):
    """
    Matches a cropped corner image against the loaded Rank and Suit templates.
    Returns the best matching Rank and Suit names as strings (e.g. "A", "Spades").
    """
    if not isinstance(corner_crop, np.ndarray) or corner_crop.size == 0:
        return "Unknown", "Unknown"
        
    # Convert corner to grayscale for matching
    if len(corner_crop.shape) == 3:
        gray_corner = cv2.cvtColor(corner_crop, cv2.COLOR_BGR2GRAY)
    else:
        gray_corner = corner_crop
        
    # Apply Otsu's thresholding to binarize the image for clean matching
    _, thresh_corner = cv2.threshold(gray_corner, 128, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

    best_rank = "Unknown"
    max_rank_val = 0
    
    # Check Ranks
    for name, template in RANK_TEMPLATES.items():
        # Make sure the template isn't larger than the corner crop
        if template.shape[0] > thresh_corner.shape[0] or template.shape[1] > thresh_corner.shape[1]:
            continue
            
        res = cv2.matchTemplate(thresh_corner, template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(res)
        
        if max_val > max_rank_val:
            max_rank_val = max_val
            best_rank = name
            
    best_suit = "Unknown"
    max_suit_val = 0
    
    # Check Suits
    for name, template in SUIT_TEMPLATES.items():
        if template.shape[0] > thresh_corner.shape[0] or template.shape[1] > thresh_corner.shape[1]:
            continue
            
        res = cv2.matchTemplate(thresh_corner, template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(res)
        
        if max_val > max_suit_val:
            max_suit_val = max_val
            best_suit = name
            
    # Set a threshold - if the match is too poor (< 50% confidence), return Unknown
    if max_rank_val < 0.5:
        best_rank = "Unknown"
    if max_suit_val < 0.5:
        best_suit = "Unknown"
        
    return best_rank, best_suit
