#!/usr/bin/env python
# coding: utf-8

# # Omi Card Game - Neural Network Predictor
# This notebook covers the preprocessing, training, and evaluation of the Omi dataset.

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import ast

# Machine Learning libraries
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix


# ## 1. Load the Dataset

# In[ ]:


DATASET_PATH = 'omi_dataset.csv'

if os.path.exists(DATASET_PATH):
    df = pd.read_csv(DATASET_PATH)
    print(f"Successfully loaded dataset with {len(df)} records.")
else:
    print(f"Dataset not found at {DATASET_PATH}.")


# ## 2. Parse Strings into Python Lists

# In[ ]:


columns_to_parse = ["p0_start", "p1_start", "p2_start", "p3_start", "history"]

for col in columns_to_parse:
    df[col] = df[col].apply(ast.literal_eval)

print("Parsing complete.")


# ## 3. Card Encoding Logic

# In[ ]:


RANKS = ["6", "7", "8", "9", "10", "J", "Q", "K", "A"]
SUITS = ["Clubs", "Diamonds", "Hearts", "Spades"]
ALL_CARDS = [f"{r}_{s}" for r in RANKS for s in SUITS]

card_to_idx = {card: i for i, card in enumerate(ALL_CARDS)}
idx_to_card = {i: card for card, i in card_to_idx.items()}

def encode_cards(cards):
    vec = [0] * 36
    for card in cards:
        if card in card_to_idx:
            vec[card_to_idx[card]] = 1
    return vec


# ## 4. Helper Functions for Strategic Logic

# In[ ]:


def parse_trick(trick_text):
    parts = trick_text.split(": ")[1]
    parts = parts.split(". Winner")[0]
    plays = parts.split(", ")

    result = []
    for play in plays:
        tokens = play.split(" ")
        player = tokens[0]
        card = tokens[2]
        result.append((player, card))
    return result

RANK_ORDER = ["6", "7", "8", "9", "10", "J", "Q", "K", "A"]

def get_card_value(card):
    rank = card.split("_")[0]
    return RANK_ORDER.index(rank)

def get_partner(player_id):
    mapping = {"P0": "P2", "P1": "P3", "P2": "P0", "P3": "P1"}
    return mapping.get(player_id)

def get_current_winner_idx(table_cards, trump):
    if not table_cards:
        return None
    lead_suit = table_cards[0].split("_")[1]
    winning_idx = 0
    winning_card = table_cards[0]
    for i in range(1, len(table_cards)):
        card = table_cards[i]
        suit = card.split("_")[1]
        if suit == trump:
            if winning_card.split("_")[1] != trump or get_card_value(card) > get_card_value(winning_card):
                winning_card = card
                winning_idx = i
        elif suit == lead_suit and winning_card.split("_")[1] == lead_suit:
            if get_card_value(card) > get_card_value(winning_card):
                winning_card = card
                winning_idx = i
    return winning_idx

def is_partner_winning(player_id, table_cards, trump, players_on_table):
    idx = get_current_winner_idx(table_cards, trump)
    if idx is None:
        return 0
    winner_player = players_on_table[idx]
    partner = get_partner(player_id)
    return 1 if winner_player == partner else 0


# ## 5. Feature Extraction Loop

# In[ ]:


def encode_trump(trump):
    suits = ["Clubs", "Diamonds", "Hearts", "Spades"]
    return [1 if s == trump else 0 for s in suits]

def encode_lead_suit(table_cards):
    suits = ["Clubs", "Diamonds", "Hearts", "Spades"]
    if not table_cards:
        return [0, 0, 0, 0, 1]
    suit = table_cards[0].split("_")[1]
    return [1 if s == suit else 0 for s in suits] + [0]

def encode_position(pos):
    vec = [0, 0, 0, 0]
    vec[pos] = 1
    return vec

def encode_trick(trick_index):
    return [trick_index / 8]

X = []
y = []

print(f"Processing {len(df)} games...")

for i, row in df.iterrows():
    hands = {"P0": row["p0_start"].copy(), "P1": row["p1_start"].copy(), "P2": row["p2_start"].copy(), "P3": row["p3_start"].copy()}
    previous_cards = []
    trump = row["trump"]
    trump_vec = encode_trump(trump)

    for trick_index, trick_text in enumerate(row["history"]):
        table_cards = []
        players_on_table = []
        plays = parse_trick(trick_text)

        for pos, (player, card) in enumerate(plays):
            partner_flag = is_partner_winning(player, table_cards, trump, players_on_table)
            hand_vec = encode_cards(hands[player])
            table_vec = encode_cards(table_cards)
            prev_vec = encode_cards(previous_cards)
            lead_vec = encode_lead_suit(table_cards)
            pos_vec = encode_position(pos)
            trick_vec = encode_trick(trick_index)

            full_vec = hand_vec + table_vec + prev_vec + trump_vec + lead_vec + trick_vec + pos_vec + [partner_flag]

            X.append(full_vec)
            y.append(card_to_idx[card])

            hands[player].remove(card)
            table_cards.append(card)
            players_on_table.append(player)
            previous_cards.append(card)

X = np.array(X)
y = np.array(y)

print(f"Created {len(X)} training samples.")


# ## 6. Training and Evaluation

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = keras.Sequential([
    keras.layers.Dense(128, activation='relu', input_shape=(123,)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(36, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

history = model.fit(X_train, y_train, epochs=5, batch_size=32, validation_split=0.1)

# --- EVALUATION --- #

print("\nEvaluating on Test Set...")
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy:.4f}")

# 1. Plot Training History
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Accuracy over Epochs')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Loss over Epochs')
plt.legend()
plt.savefig('training_metrics.png')
print("Training metrics plot saved as training_metrics.png")

# 2. Detailed Classification Report
print("\nGenerating predictions for classification report...")
y_pred = np.argmax(model.predict(X_test), axis=1)
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=ALL_CARDS, digits=4))

# 3. Save the model
model.save('omi_model.keras')
print("\nModel saved as omi_model.keras")

