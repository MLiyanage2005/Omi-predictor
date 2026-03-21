import json

# Card Rankings (Short Deck Omi: 6 to Ace)
RANK_ORDER = {
    "6": 1, "7": 2, "8": 3, "9": 4, "10": 5, 
    "J": 6, "Q": 7, "K": 8, "A": 9
}

SUITS = ["Clubs", "Diamonds", "Hearts", "Spades"]

class GameState:
    def __init__(self):
        self.reset()

    def reset(self):
        self.my_hand = []
        self.played_cards = []  # List of all cards played in the game so far
        self.current_trick = [] # Cards played in the current trick: list of {"player": int, "card": Card}
        self.trump = None       # "Hearts", "Spades", etc.
        self.lead_suit = None   # First suit played in the current trick
        self.teammate_index = 2 # Assuming 0 is me, 1 is RHO, 2 is teammate, 3 is LHO
        self.player_count = 4
        self.scores = {"team_me": 0, "team_opp": 0}
        self.player_voids = [set() for _ in range(4)] # Track suits each player is void in

    def to_dict(self):
        return {
            "my_hand": [str(c) for c in self.my_hand],
            "played_cards": [str(c) for c in self.played_cards],
            "current_trick": [{"player": t["player"], "card": str(t["card"])} for t in self.current_trick],
            "trump": self.trump,
            "lead_suit": self.lead_suit,
            "scores": self.scores
        }

    def __str__(self):
        return json.dumps(self.to_dict(), indent=2)

class Card:
    def __init__(self, rank, suit):
        self.rank = rank
        self.suit = suit

    def __repr__(self):
        return f"{self.rank}_{self.suit}"

    def __str__(self):
        return f"{self.rank} of {self.suit}"

    def __eq__(self, other):
        if not isinstance(other, Card):
            return False
        return self.rank == other.rank and self.suit == other.suit
