import random
import sys
import os

# Add the project root to sys.path so it can find the 'game' module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from game.state import GameState, Card, SUITS, RANK_ORDER
from game.rules import is_valid_move, get_winner
from game.strategy import choose_card

def simulate_round():
    print("--- Starting Omi Game Simulation ---")
    game = GameState()
    
    # 1. Create and Shuffle Deck (Short Deck: 6 to A)
    deck = [Card(r, s) for r in RANK_ORDER.keys() for s in SUITS]
    random.shuffle(deck)
    
    # 2. Deal Cards (8 cards each for 4 players)
    player_hands = [[] for _ in range(4)]
    for _ in range(8):
        for i in range(4):
            player_hands[i].append(deck.pop())
            
    game.my_hand = player_hands[0]
    print(f"My Hand: {game.my_hand}")
    
    # 3. Simulate Trump Selection (Player 0 chooses)
    # In a real game, this happens after first 4 cards, but let's simplify
    game.trump = random.choice(SUITS)
    print(f"Trump chosen: {game.trump}")
    
    # 4. Play 8 Tricks
    current_leader = 0 # Player 0 starts
    
    for trick_num in range(1, 9):
        print(f"\n--- Trick {trick_num} ---")
        game.current_trick = []
        game.lead_suit = None
        
        # Players play in turn
        for i in range(4):
            p_idx = (current_leader + i) % 4
            hand = player_hands[p_idx]
            
            # AI (or simplified logic) chooses card
            if p_idx == 0:
                # Use our actual AI strategy for "me"
                card_to_play = choose_card(game.my_hand, game.current_trick, game.trump, game.lead_suit, game.teammate_index)
            else:
                # Other players use a simple "must follow suit" logic for simulation
                valid_moves = [c for c in hand if is_valid_move(c, hand, game.lead_suit)]
                # Simple rule: try to win if possible, else play random valid
                # For simulation, just pick the first valid move
                card_to_play = valid_moves[0]
                
            # Play the card
            if not game.lead_suit:
                game.lead_suit = card_to_play.suit
                
            game.current_trick.append({"player": p_idx, "card": card_to_play})
            game.played_cards.append(card_to_play)
            hand.remove(card_to_play)
            
            print(f"Player {p_idx} plays: {card_to_play}")
            
        # Determine Winner
        trick_cards = [t["card"] for t in game.current_trick]
        winner_card = get_winner(trick_cards, game.trump, game.lead_suit)
        
        # Find player record associated with winner_card
        winner_player = -1
        for t in game.current_trick:
            if t["card"] == winner_card:
                winner_player = t["player"]
                break
        
        print(f"Winner of trick {trick_num}: Player {winner_player} with {winner_card}")
        
        # Update score
        if winner_player in [0, 2]:
            game.scores["team_me"] += 1
        else:
            game.scores["team_opp"] += 1
            
        # Winner leads next trick
        current_leader = winner_player
        
    print("\n--- Game Over ---")
    print(f"Final Scores: {game.scores}")
    if game.scores["team_me"] > game.scores["team_opp"]:
        print("Our team wins!")
    elif game.scores["team_me"] < game.scores["team_opp"]:
        print("Opponent team wins!")
    else:
        print("It's a tie!")

if __name__ == "__main__":
    simulate_round()
