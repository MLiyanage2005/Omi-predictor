from game.rules import is_valid_move, get_winner
from game.state import RANK_ORDER

def choose_card(hand, trick, trump, lead_suit, teammate_index):
    """
    Core AI decision function for Omi.
    """
    # 1. Get all valid moves
    valid_moves = [c for c in hand if is_valid_move(c, hand, lead_suit)]
    
    if not valid_moves:
        return None

    # Helper functions
    def get_lowest(cards):
        return min(cards, key=lambda c: RANK_ORDER.get(c.rank, 0))
    
    def get_highest(cards):
        return max(cards, key=lambda c: RANK_ORDER.get(c.rank, 0))

    def beats_current(card, current_trick, trump, lead_suit):
        if not current_trick:
            return True
        trick_cards = [t["card"] for t in current_trick]
        current_winner = get_winner(trick_cards, trump, lead_suit)
        new_winner = get_winner(trick_cards + [card], trump, lead_suit)
        return new_winner == card

    # Case 1: Leading the trick
    if not trick:
        # Try to play a high card if it's likely to win, otherwise play low
        # Simple strategy: play highest card for now
        return get_highest(valid_moves)

    # Check if teammate is currently winning
    trick_cards_only = [t["card"] for t in trick]
    current_winner_card = get_winner(trick_cards_only, trump, lead_suit)
    
    teammate_winning = False
    for t in trick:
        if t["card"] == current_winner_card and t["player"] == teammate_index:
            teammate_winning = True
            break

    # Case 2: Teammate is winning
    if teammate_winning:
        # Do not waste high cards, play lowest card
        return get_lowest(valid_moves)

    # Case 3: We can win the trick
    winning_moves = [c for c in valid_moves if beats_current(c, trick, trump, lead_suit)]
    if winning_moves:
        # Play the smallest card that still wins the trick
        return get_lowest(winning_moves)

    # Case 4: We cannot win the trick
    # Play the lowest possible card (discard)
    return get_lowest(valid_moves)
