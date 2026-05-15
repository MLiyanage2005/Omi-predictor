"""
Neural Network Strategy Module for Omi Card Game.

Loads the trained Keras model and provides nn_predict_card() to replace
the hand-coded heuristics in strategy.py.

Feature vector layout (123 dimensions) — MUST match training notebook exactly:
  [0:36]   hand_vec         — binary: cards in player's hand
  [36:72]  table_vec        — binary: cards on table this trick
  [72:108] prev_vec         — binary: all previously played cards
  [108:112] trump_vec       — one-hot: trump suit
  [112:117] lead_vec        — one-hot: lead suit + no-lead flag
  [117:118] trick_vec       — float: trick_index / 8
  [118:122] pos_vec         — one-hot: position in trick (0-3)
  [122:123] partner_flag    — binary: is partner currently winning?
"""

import os
import numpy as np

# ---------- Card Encoding (mirrors training notebook §3) ----------

RANKS = ["6", "7", "8", "9", "10", "J", "Q", "K", "A"]
SUITS = ["Clubs", "Diamonds", "Hearts", "Spades"]
ALL_CARDS = [f"{r}_{s}" for r in RANKS for s in SUITS]

card_to_idx = {card: i for i, card in enumerate(ALL_CARDS)}
idx_to_card = {i: card for card, i in card_to_idx.items()}

RANK_ORDER = {rank: i for i, rank in enumerate(RANKS)}


def _encode_cards(card_strings):
    """Convert a list of card strings ('A_Spades') to a 36-dim binary vector."""
    vec = [0] * 36
    for card in card_strings:
        if card in card_to_idx:
            vec[card_to_idx[card]] = 1
    return vec


def _encode_trump(trump):
    """One-hot encode the trump suit (4 dims)."""
    return [1 if s == trump else 0 for s in SUITS]


def _encode_lead_suit(table_card_strings):
    """
    One-hot encode the lead suit (5 dims).
    If no cards on table, returns [0,0,0,0,1] (no-lead flag).
    """
    if not table_card_strings:
        return [0, 0, 0, 0, 1]
    suit = table_card_strings[0].split("_")[1]
    return [1 if s == suit else 0 for s in SUITS] + [0]


def _encode_position(pos):
    """One-hot encode position in trick (4 dims)."""
    vec = [0, 0, 0, 0]
    vec[pos] = 1
    return vec


def _encode_trick(trick_index):
    """Normalised trick number (1 dim)."""
    return [trick_index / 8]


def _get_card_value(card_str):
    """Get the numeric rank value from a card string like 'A_Spades'."""
    rank = card_str.split("_")[0]
    return RANK_ORDER.get(rank, 0)


def _get_current_winner_idx(table_card_strings, trump):
    """Return the index (within table_card_strings) of the currently winning card."""
    if not table_card_strings:
        return None
    lead_suit = table_card_strings[0].split("_")[1]
    winning_idx = 0
    winning_card = table_card_strings[0]
    for i in range(1, len(table_card_strings)):
        card = table_card_strings[i]
        suit = card.split("_")[1]
        if suit == trump:
            if winning_card.split("_")[1] != trump or _get_card_value(card) > _get_card_value(winning_card):
                winning_card = card
                winning_idx = i
        elif suit == lead_suit and winning_card.split("_")[1] == lead_suit:
            if _get_card_value(card) > _get_card_value(winning_card):
                winning_card = card
                winning_idx = i
    return winning_idx


def _is_partner_winning(player_id, table_card_strings, trump, players_on_table):
    """Check if the player's partner is currently winning the trick."""
    idx = _get_current_winner_idx(table_card_strings, trump)
    if idx is None:
        return 0
    winner_player = players_on_table[idx]
    partner_map = {0: 2, 1: 3, 2: 0, 3: 1}
    partner = partner_map.get(player_id)
    return 1 if winner_player == partner else 0


# ---------- Model Loading ----------

_model = None


def _load_model():
    """Lazy-load the Keras model once."""
    global _model
    if _model is not None:
        return _model

    try:
        import tensorflow as tf
        # Suppress TF info logs
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

        model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "omi_model.keras")
        if not os.path.exists(model_path):
            print(f"[NN Strategy] Model not found at {model_path}")
            return None

        _model = tf.keras.models.load_model(model_path)
        print(f"[NN Strategy] Model loaded successfully from {model_path}")
        return _model

    except Exception as e:
        print(f"[NN Strategy] Failed to load model: {e}")
        return None


# ---------- Public API ----------

def nn_predict_card(game_state, player_id=0):
    """
    Use the trained neural network to predict the best card to play.

    Args:
        game_state: GameState object from game/state.py
        player_id:  Which player we are (default 0 = "Me")

    Returns:
        A Card object representing the best move, or None if prediction fails.
    """
    from game.state import Card
    from game.rules import is_valid_move

    model = _load_model()
    if model is None:
        return None  # Caller should fall back to heuristic strategy

    # --- Convert game state to string-based card names ---
    hand_strings = [f"{c.rank}_{c.suit}" for c in game_state.my_hand]

    table_card_strings = []
    players_on_table = []
    for t in game_state.current_trick:
        card = t["card"]
        table_card_strings.append(f"{card.rank}_{card.suit}")
        players_on_table.append(t["player"])

    prev_strings = [f"{c.rank}_{c.suit}" for c in game_state.played_cards]

    trump = game_state.trump if game_state.trump else "Clubs"

    # Current position in the trick (0 = leading, 1 = second, etc.)
    position = len(game_state.current_trick)

    # Estimate the trick number from how many cards have been played
    # Each trick = 4 cards, so trick_index = len(played_cards) // 4
    trick_index = len(game_state.played_cards) // 4

    # Partner winning flag
    partner_flag = _is_partner_winning(player_id, table_card_strings, trump, players_on_table)

    # --- Build the 123-dim feature vector (exact same order as training) ---
    hand_vec = _encode_cards(hand_strings)
    table_vec = _encode_cards(table_card_strings)
    prev_vec = _encode_cards(prev_strings)
    trump_vec = _encode_trump(trump)
    lead_vec = _encode_lead_suit(table_card_strings)
    trick_vec = _encode_trick(trick_index)
    pos_vec = _encode_position(position)

    full_vec = hand_vec + table_vec + prev_vec + trump_vec + lead_vec + trick_vec + pos_vec + [partner_flag]

    # --- Run inference ---
    X = np.array([full_vec], dtype=np.float32)
    probs = model.predict(X, verbose=0)[0]  # Shape: (36,)

    # --- Legal move masking ---
    # Step 1: Zero out cards NOT in hand
    mask = np.zeros(36)
    for card_str in hand_strings:
        if card_str in card_to_idx:
            mask[card_to_idx[card_str]] = 1.0
    probs = probs * mask

    # Step 2: Enforce suit-following rules
    lead_suit = game_state.lead_suit
    if lead_suit:
        has_lead_suit = any(c.suit == lead_suit for c in game_state.my_hand)
        if has_lead_suit:
            # Must follow suit — zero out cards of other suits
            for card_str in hand_strings:
                suit = card_str.split("_")[1]
                if suit != lead_suit:
                    probs[card_to_idx[card_str]] = 0.0

    # --- Pick the best legal card ---
    if probs.sum() == 0:
        # Safety fallback: if all probabilities are zero, return None
        return None

    best_idx = np.argmax(probs)
    best_card_str = idx_to_card[best_idx]
    rank, suit = best_card_str.split("_")

    return Card(rank, suit)
