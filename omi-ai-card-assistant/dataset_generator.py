import random
import csv
import json
import os
from collections import Counter

# Defining the Deck
RANKS = ["6", "7", "8", "9", "10", "J", "Q", "K", "A"]
RANK_ORDER = {rank: i for i, rank in enumerate(RANKS)}
SUITS = ["Clubs", "Diamonds", "Hearts", "Spades"]

class Card:
    def __init__(self, rank, suit):
        self.rank = rank
        self.suit = suit
        self.value = RANK_ORDER[rank]
        
    def __repr__(self):
        return f"{self.rank}_{self.suit}"

def get_winner(trick, trump, lead_suit):
    """Returns the index (in the trick list) of the winning card."""
    winning_idx = 0
    winning_card = trick[0]
    
    for i in range(1, len(trick)):
        card = trick[i]
        if card.suit == winning_card.suit:
            if card.value > winning_card.value:
                winning_card = card
                winning_idx = i
        elif card.suit == trump:
            winning_card = card
            winning_idx = i
            
    return winning_idx

class ProBot:
    def __init__(self, player_id):
        self.id = player_id
        self.hand = []
        
    def choose_card(self, current_trick, trump, lead_suit, played_cards):
        """
        Heuristics-based bot to simulate 'pro natural moves'.
        """
        # 1. Determine valid moves
        if lead_suit is None:
            valid_moves = self.hand
        else:
            same_suit = [c for c in self.hand if c.suit == lead_suit]
            valid_moves = same_suit if same_suit else self.hand
            
        if len(valid_moves) == 1:
            card = valid_moves[0]
            self.hand.remove(card)
            return card

        # Helper functions
        def get_lowest(cards): return min(cards, key=lambda c: c.value)
        def get_highest(cards): return max(cards, key=lambda c: c.value)

        # Case A: Leading the trick
        if not current_trick:
            # Pro move: Try to lead with a high non-trump to draw out enemy trumps,
            # or lead the highest card we have.
            non_trumps = [c for c in valid_moves if c.suit != trump]
            if non_trumps:
                choice = get_highest(non_trumps)
            else:
                choice = get_highest(valid_moves)
            self.hand.remove(choice)
            return choice
            
        # Analyze current trick
        trick_cards = [t["card"] for t in current_trick]
        winning_idx = get_winner(trick_cards, trump, lead_suit)
        winning_player = current_trick[winning_idx]["player"]
        
        teammate_id = (self.id + 2) % 4
        teammate_is_winning = (winning_player == teammate_id)
        
        current_winner_card = trick_cards[winning_idx]

        # Case B: Teammate is winning the trick
        if teammate_is_winning:
            # Pro move: Conserve high cards, throw the lowest valid card
            choice = get_lowest(valid_moves)
            self.hand.remove(choice)
            return choice
            
        # Case C: Enemy is winning, we MUST follow suit
        if valid_moves[0].suit == lead_suit:
            # Can we mathematically beat the current winner?
            winning_moves = [c for c in valid_moves if c.value > current_winner_card.value]
            if winning_moves:
                # Pro move: Play the LOWEST card that still beats the enemy to save our highest cards
                choice = get_lowest(winning_moves)
            else:
                # We can't beat them, throw away the lowest garbage card
                choice = get_lowest(valid_moves)
            self.hand.remove(choice)
            return choice
            
        # Case D: Enemy is winning, we are VOID in the lead suit
        # Pro move: We can 'Ruff' (Trump) it!
        trumps_in_hand = [c for c in valid_moves if c.suit == trump]
        if trumps_in_hand:
            # Does the enemy already have a trump down?
            enemy_trump_value = current_winner_card.value if current_winner_card.suit == trump else -1
            winning_trumps = [c for c in trumps_in_hand if c.value > enemy_trump_value]
            
            if winning_trumps:
                # Play the smallest trump that wins
                choice = get_lowest(winning_trumps)
            else:
                # Can't beat their trump, throw lowest off-suit garbage
                none_trumps = [c for c in valid_moves if c.suit != trump]
                choice = get_lowest(none_trumps) if none_trumps else get_lowest(trumps_in_hand)
        else:
            # We don't have trumps either. Throw lowest garbage.
            choice = get_lowest(valid_moves)

        self.hand.remove(choice)
        return choice

def simulate_game(game_id):
    # Setup Deck
    deck = [Card(r, s) for r in RANKS for s in SUITS]
    random.shuffle(deck)
    
    # Setup Players
    players = [ProBot(i) for i in range(4)]
    
    # Deal 9 cards to each player (36 cards total)
    for i in range(4):
        players[i].hand = deck[i*9 : (i+1)*9]
        
    # The player who acts first picks the trump (simplified)
    # Usually trump is picked by inspecting the first 4 cards. We simulate it
    # by picking the suit they have the most of.
    suit_counts = Counter([c.suit for c in players[0].hand])
    trump = suit_counts.most_common(1)[0][0]
    
    # Save Initial Hands
    initial_hands = [[str(c) for c in players[i].hand] for i in range(4)]
    
    played_cards = []
    game_history = []
    
    current_leader = 0
    scores = {0:0, 1:0} # Team 0 (Players 0 & 2) vs Team 1 (Players 1 & 3)
    
    # Play 9 Tricks
    for trick_num in range(9):
        trick = []
        lead_suit = None
        
        # 4 Players play in order starting from the leader
        for turn in range(4):
            active_player_idx = (current_leader + turn) % 4
            bot = players[active_player_idx]
            
            # Bot thinks...
            card = bot.choose_card(trick, trump, lead_suit, played_cards)
            
            if not lead_suit:
                lead_suit = card.suit
                
            played_cards.append(card)
            trick.append({"player": active_player_idx, "card": card})
            
        # Determine Trick Winner
        trick_cards = [t["card"] for t in trick]
        win_idx = get_winner(trick_cards, trump, lead_suit)
        winner_id = trick[win_idx]["player"]
        
        # Update Leader & Score
        current_leader = winner_id
        team_winner = winner_id % 2
        scores[team_winner] += 1
        
        # Record trick log string
        log_str = ", ".join([f"P{t['player']} played {t['card']}" for t in trick])
        game_history.append(f"Trick {trick_num+1}: {log_str}. Winner: P{winner_id}")

    # Determine final game result
    if scores[0] > scores[1]:
        result = "Team 0 Wins"
    elif scores[1] > scores[0]:
        result = "Team 1 Wins"
    else:
        result = "Draw"

    # Return the row of data for the CSV
    return {
        "game_id": game_id,
        "trump": trump,
        "p0_start": json.dumps(initial_hands[0]),
        "p1_start": json.dumps(initial_hands[1]),
        "p2_start": json.dumps(initial_hands[2]),
        "p3_start": json.dumps(initial_hands[3]),
        "history": json.dumps(game_history),
        "result": result
    }

def generate_dataset(num_games=1000, output_file="omi_dataset.csv"):
    print(f"Simulating {num_games} games...")
    
    headers = ["game_id", "trump", "p0_start", "p1_start", "p2_start", "p3_start", "history", "result"]
    with open(output_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        
        for i in range(num_games):
            row = simulate_game(i)
            writer.writerow(row)
            
            if (i+1) % 10000 == 0:
                print(f"Finished {i+1} games...")
                
    print(f"Dataset generated and saved to {output_file}!")

if __name__ == "__main__":
    # Start with 10,000 games to see how fast it is.
    # To train a deep model, eventually change this to 1,000,000.
    generate_dataset(num_games=10000, output_file="omi_dataset.csv")
