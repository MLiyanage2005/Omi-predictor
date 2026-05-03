import cv2
import sys
import os
import numpy as np
import config
from vision.card_detector import detect_card
from vision.perspective import flatten_card, crop_corner, get_player_zone
from vision.matcher import match_card
from game.state import GameState, Card
from game.strategy import choose_card

def main():
    # Initialize Game State
    game = GameState()
    
    # Open the camera using the source from config.py
    # Use FFMPEG backend which is much more stable for HTTP streams
    if isinstance(config.CAMERA_SOURCE, str) and config.CAMERA_SOURCE.startswith("http"):
        cap = cv2.VideoCapture(config.CAMERA_SOURCE, cv2.CAP_FFMPEG)
    else:
        cap = cv2.VideoCapture(config.CAMERA_SOURCE)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        sys.exit(1)

    print("Camera successfully opened. Press ESC to exit.")

    stable_card_counter = 0
    last_seen_card = None

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        # If frame is read correctly ret is True
        if not ret or frame is None:
            print("Error: Can't receive frame (stream end or connection failed). Exiting...")
            break

        # Detect all card contours on the table
        card_contours = detect_card(frame)
        
        # We will only show the flattened view/corner 
        # of the FIRST card detected so we don't spam windows,
        # but we will draw boxes around ALL of them.
        first_card = True
        
        for contour in card_contours:
            # Draw a thick green box around the detected card
            cv2.drawContours(frame, [contour], -1, (0, 255, 0), 3)
            
            # Flatten the card
            warped_card = flatten_card(frame, contour)
            
            # Crop the corner
            corner_crop = crop_corner(warped_card)
            
            # Get player zone
            frame_height, frame_width = frame.shape[:2]
            player_id = get_player_zone(contour, frame_width, frame_height)
            
            # Predict the card rank and suit
            rank, suit = match_card(corner_crop)
            player_names = {0: "Me", 1: "Left Opponent", 2: "Teammate", 3: "Right Opponent"}
            card_name = f"{rank} of {suit} - {player_names.get(player_id, 'Unknown')}"
            current_card = Card(rank, suit) if rank != "Unknown" and suit != "Unknown" else None
            
            # Print to console so the user can see log output
            print(f"Identified Card: {card_name}")
            
            # Get AI suggestion
            best_move = None
            if current_card and current_card in game.my_hand:
                best_move = choose_card(game.my_hand, game.current_trick, game.trump, game.lead_suit, game.teammate_index)
            
            # Find the top-left point of the contour to draw text near the specific card
            pts = contour.reshape(4, 2)
            top_left = tuple(map(int, pts[np.argmin(pts.sum(axis=1))]))
            
            # Draw the predicted name near the specific card
            label_color = (0, 0, 255)
            if best_move and current_card == best_move:
                label_color = (255, 0, 0) # Blue for AI recommended move
                card_name += " [AI SUGGESTION]"
                
            cv2.putText(frame, card_name, (top_left[0], top_left[1] - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, label_color, 2)
            
            # Only show the debug windows for the primary/largest card
            if first_card:
                cv2.imshow('Warped (Top-Down) Card', warped_card)
                cv2.imshow('Card Corner', corner_crop)
                
                # Save into a variable so the 's' and 'p' key logic below can access it
                primary_corner_crop = corner_crop
                primary_card = current_card
                primary_player_id = player_id
                first_card = False

        # Auto-play stable cards
        if 'primary_card' in locals() and primary_card:
            if primary_card == last_seen_card and primary_card not in game.played_cards:
                stable_card_counter += 1
                if stable_card_counter > 30: # About 1 second at 30fps
                    game.current_trick.append({"player": primary_player_id, "card": primary_card})
                    game.played_cards.append(primary_card)
                    
                    if not game.lead_suit:
                        game.lead_suit = primary_card.suit
                        
                    if primary_card in game.my_hand:
                        game.my_hand.remove(primary_card)
                        
                    print(f"Auto-Played Card: {primary_card} by {player_names.get(primary_player_id, 'Unknown')}")
                    stable_card_counter = 0 # reset
            elif primary_card != last_seen_card:
                stable_card_counter = 0
                last_seen_card = primary_card
        else:
            stable_card_counter = 0
            last_seen_card = None

        # Display the main camera frame
        cv2.imshow('Omi AI Card Assistant - Live Feed', frame)

        # Wait for 1 ms and check keys
        key = cv2.waitKey(1)
        if key == 27: # ESC key
            print("ESC pressed. Exiting...")
            break
        elif key == ord('s'): # 's' key to save template
            if 'primary_corner_crop' in locals() and primary_corner_crop is not None:
                # Binarize corner crop for saving as a clean black & white template
                gray_corner = cv2.cvtColor(primary_corner_crop, cv2.COLOR_BGR2GRAY)
                _, thresh_corner = cv2.threshold(gray_corner, 128, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
                
                num_saved = len([f for f in os.listdir('.') if f.startswith('captured_template')])
                filename = f"captured_template_{num_saved + 1}.jpg"
                cv2.imwrite(filename, thresh_corner)
                print(f"Captured {filename}! Move this to templates/ranks/ or templates/suits/ and rename it (e.g. 'A.jpg' or 'Hearts.jpg').")
        elif key == ord('h'): # 'h' key to add card to hand
            if 'primary_card' in locals() and primary_card:
                if primary_card not in game.my_hand:
                    game.my_hand.append(primary_card)
                    print(f"Added to Hand: {primary_card}")
        elif key == ord('p'): # 'p' key to play a card on the table
            if 'primary_card' in locals() and primary_card and primary_card not in game.played_cards:
                game.current_trick.append({"player": primary_player_id, "card": primary_card})
                game.played_cards.append(primary_card)
                
                if not game.lead_suit:
                    game.lead_suit = primary_card.suit
                    
                if primary_card in game.my_hand:
                    game.my_hand.remove(primary_card)
                    
                print(f"Manually Played Card: {primary_card} by Player {primary_player_id}")
        elif key == ord('r'): # 'r' key to reset trick or game
            game.reset()
            print("Game Reset")
        elif key == ord('t'): # 't' key to set trump
            if 'primary_card' in locals() and primary_card:
                game.trump = primary_card.suit
                print(f"Trump set to: {game.trump}")

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
