from game.state import RANK_ORDER

def is_valid_move(card, hand, lead_suit):
    """
    Checks if playing a specific card is a legal move.
    """
    # If this is the first card of the trick, any card is valid
    if lead_suit is None:
        return True
        
    # If the player has cards of the lead suit, they MUST play one of them
    same_suit_cards = [c for c in hand if c.suit == lead_suit]
    if same_suit_cards:
        return card.suit == lead_suit
        
    # If the player is void in the lead suit, they can play any card
    return True

def get_winner(trick_cards, trump, lead_suit):
    """
    Determines which card won the trick.
    trick_cards: List of Card objects (in the order they were played)
    Returns: The Card object that won the trick.
    """
    if not trick_cards:
        return None
        
    winning_card = trick_cards[0]

    for i in range(1, len(trick_cards)):
        card = trick_cards[i]
        
        # 1. If both cards are the same suit, the higher rank wins
        if card.suit == winning_card.suit:
            if RANK_ORDER.get(card.rank, 0) > RANK_ORDER.get(winning_card.rank, 0):
                winning_card = card
        
        # 2. If the current winning card is NOT trump, but the new card IS trump, new card wins
        elif card.suit == trump:
            winning_card = card
            
        # 3. If the new card is a different suit (and not trump), it cannot beat the current winner
        # Note: We assume the winning_card either follows suit or is trump.
        
    return winning_card
