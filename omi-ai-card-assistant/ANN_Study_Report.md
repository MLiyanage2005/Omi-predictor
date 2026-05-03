# 🧠 ANN Study Report — Omi Card Game Next-Move Predictor
## Step-by-Step Learning Guide with Tool Guidance

---

## 🎯 What We Are Building

> **At every turn in the game, the player can see:**
> - Their own hand (cards currently held)
> - Cards already on the table this trick (what others played this round)
> - All cards played in every previous trick (full game memory)
> - The trump suit and lead suit
>
> **Goal: predict the best card to play right now to help the team win.**

This is a **card-play decision model** — a 36-class classification problem.
The ANN outputs a probability for each of the 36 cards, and the highest valid one is played.

---

## 📦 Tools You Will Use

| Tool | Purpose | Install |
|---|---|---|
| **Python 3.10+** | Language | Already installed |
| **pandas** | Load & manipulate CSV data | `pip install pandas` |
| **numpy** | Build feature vectors & arrays | `pip install numpy` |
| **TensorFlow / Keras** | Build and train the ANN | `pip install tensorflow` |
| **scikit-learn** | Data splits, metrics, baselines | `pip install scikit-learn` |
| **matplotlib** | Plot training curves | `pip install matplotlib` |
| **seaborn** | Prettier plots (confusion matrix) | `pip install seaborn` |
| **Jupyter** | Interactive notebook coding | `pip install jupyter` |

### Install everything in one command:
```bash
pip install tensorflow scikit-learn pandas numpy matplotlib seaborn jupyter
```

### Check TensorFlow installed correctly:
```bash
python -c "import tensorflow as tf; print(tf.__version__)"
```

---

## 📊 Part 1 — Understanding the Dataset

### File: `omi_dataset.csv`
- **10,000 rows** — each row = one complete Omi game
- **8 columns** per game

| Column | Type | Example |
|---|---|---|
| `game_id` | int | 0 |
| `trump` | string | `"Clubs"` |
| `p0_start` | JSON list | `["10_Clubs", "6_Clubs", ...]` |
| `p1_start` | JSON list | `["K_Spades", "7_Clubs", ...]` |
| `p2_start` | JSON list | `["A_Clubs", "A_Hearts", ...]` |
| `p3_start` | JSON list | `["9_Hearts", "A_Spades", ...]` |
| `history` | JSON list | `["Trick 1: P0 played Q_Diamonds, P1 played K_Diamonds, ... Winner: P2", ...]` |
| `result` | string | `"Team 1 Wins"` |

### Key facts:
- Each game has exactly **9 tricks** × 4 players = **36 decision points**
- Total decision points: `10,000 × 36 = 360,000` — this is your real training dataset
- Labels come from the **ProBot** (heuristic expert) — the ANN learns to imitate smart moves
- Teams: **Team 0 = P0 & P2** | **Team 1 = P1 & P3**

### Run `step1_explore.py` to see this for yourself.

---

## 🔧 Part 2 — Feature Engineering

A neural network only understands numbers. Here is how to convert game state to numbers.

### Card Index Map (36 cards → 0–35)

```
"6_Clubs"=0,  "7_Clubs"=1,  ..., "A_Clubs"=8
"6_Diamonds"=9, ..., "A_Diamonds"=17
"6_Hearts"=18, ..., "A_Hearts"=26
"6_Spades"=27, ..., "A_Spades"=35
```

### Feature Vector — 123 numbers per decision point

```
Block                         Size   What it encodes
────────────────────────────────────────────────────────────────
my_hand_vector                 36    Binary: which cards am I holding?
cards_on_table_vector          36    Binary: cards played this trick so far
previously_played_vector       36    Binary: all cards from past tricks
trump_suit                      4    One-hot: [Clubs, Diamonds, Hearts, Spades]
lead_suit                       5    One-hot: [Clubs, Diamonds, Hearts, Spades, NoLead]
trick_number                    1    Float 0.0–1.0 (trick 0→0.0, trick 8→1.0)
my_position_in_trick            4    One-hot: [1st, 2nd, 3rd, 4th to play]
teammate_winning_flag           1    Binary: is my teammate currently winning?
────────────────────────────────────────────────────────────────
TOTAL                         123
```

### Label — 36-class integer

```
y = card_to_idx[card_played]   →  integer from 0 to 35
```

### Run `step2_features.py` to build and verify X and y.

---

## 🧠 Part 3 — ANN Architecture

### How a Dense (Fully Connected) Layer Works

```
output = activation( W · input + bias )

W       = weight matrix (learned during training)
bias    = learnable offset
·       = dot product (matrix multiply)
activation = non-linear function (ReLU, Sigmoid, Softmax)
```

### Activation Functions Used

| Function | Formula | Where Used |
|---|---|---|
| **ReLU** | `max(0, x)` | All hidden layers |
| **BatchNorm** | Normalises layer inputs | After hidden layers |
| **Dropout** | Randomly zeros neurons | After hidden layers (prevents overfitting) |
| **Softmax** | `eˣ / Σeˣ` | Final layer → 36 probabilities summing to 1 |

### ANN Architecture Diagram

```
Input (123)
    │
    ▼
Dense(256, ReLU) ──► BatchNorm ──► Dropout(0.3)
    │
    ▼
Dense(128, ReLU) ──► BatchNorm ──► Dropout(0.2)
    │
    ▼
Dense(64, ReLU) ──► Dropout(0.1)
    │
    ▼
Dense(36, Softmax)   ←── 36 probabilities, one per card
    │
    ▼
argmax + masking     ←── Pick highest-prob VALID card
    │
    ▼
Card to Play 🃏
```

### Loss Function: `sparse_categorical_crossentropy`

Used because labels are integers (0–35), not one-hot arrays.

```
Loss = -log(probability assigned to correct card)
```

The optimizer (Adam) nudges all weights to reduce this loss over time.

### Run `step3_build_model.py` to see the model summary.

---

## 📚 Part 4 — Training Process Explained

### What happens in one epoch:

1. **Forward Pass**: Feature vector goes through all layers → 36 probabilities output
2. **Loss Calc**: Compare predicted probabilities to true card label
3. **Backward Pass (Backprop)**: Calculate gradient — how much each weight contributed to error
4. **Weight Update**: Adam optimizer shifts weights slightly to reduce loss
5. Repeat for every batch in the dataset

### Key Training Parameters

| Parameter | What it does | Value Used |
|---|---|---|
| `epochs` | How many full passes through data | 100 (with early stopping) |
| `batch_size` | Samples per weight update | 256 |
| `learning_rate` | Step size for weight updates | 0.001 (Adam default) |
| `validation_split` | % of training data held back to monitor overfitting | 0.1 (10%) |

### Early Stopping

```python
EarlyStopping(monitor='val_accuracy', patience=10)
```
If validation accuracy doesn't improve for 10 epochs → stop training, restore best weights.
This prevents overfitting automatically.

### Run `step4_train.py` to train the model and save it.

---

## 📈 Part 5 — Evaluation

### Metrics

| Metric | Meaning | Target |
|---|---|---|
| **Top-1 Accuracy** | ANN picks exact card the ProBot played | 40–65% |
| **Top-3 Accuracy** | Correct card is in top 3 predictions | 65–85% |
| **Legal Move Rate** | ANN never plays illegal card | 100% (guaranteed by masking) |

> **Why is 50% considered good?**
> Multiple cards are strategically correct at many moments. If your teammate is winning, ANY low card is right. Top-3 accuracy is the more meaningful metric.

### Confusion Matrix (optional, for top cards)

Shows which cards the model confuses with each other. Expected: cards of the same suit/rank will be confused occasionally.

### Run `step5_evaluate.py` to see all metrics and plots.

---

## 🎮 Part 6 — Using the Model in the Real Game

```python
def predict_next_card(model, my_hand, table_cards, prev_cards,
                      trump, lead_suit, trick_num, my_position):
    # 1. Build feature vector (same 123 features as training)
    feature = build_feature(...)
    
    # 2. Get 36 raw probabilities from ANN
    probs = model.predict(feature)   # e.g. [0.02, 0.01, 0.15, ...]
    
    # 3. MASK: zero out cards not in my hand
    probs = mask_invalid_cards(probs, my_hand)
    
    # 4. MASK: if I must follow suit, zero off-suit cards
    probs = apply_suit_rules(probs, my_hand, lead_suit)
    
    # 5. Pick the card with highest remaining probability
    best_card = ALL_CARDS[argmax(probs)]
    return best_card
```

### Run `step6_inference.py` for a working demo.

---

## 🗺️ Part 7 — Algorithm Comparison

| Algorithm | Suitability for This Task | Strength | Weakness |
|---|---|---|---|
| **Logistic Regression** | 🟡 Weak baseline | Fast, simple | Linear only — can't learn card interactions |
| **Random Forest** | 🟢 Good baseline | No scaling needed, robust | Slow predict with 36 classes |
| **XGBoost / GBM** | 🟢 Strong baseline | Often beats ANN on tabular | Needs tuning |
| **MLP / ANN** ⭐ | 🟢 Primary model | Learns complex patterns | Needs tuning |
| **LSTM** | 🔵 Phase 2 | Treats tricks as sequence | More complex to build |
| **Transformer** | 🔵 Advanced | Attention over all cards | Overkill for now |

**Your progression path:**
```
Random Forest baseline → ANN (MLP) → LSTM (sequence-aware) → Transformer
```

---

## ✅ Study Checklist

Complete in order. Don't skip ahead.

- [ ] **Install tools**: Run `pip install tensorflow scikit-learn pandas numpy matplotlib seaborn jupyter`
- [ ] **Step 1**: Run `step1_explore.py` — understand the dataset, parse history strings
- [ ] **Step 2**: Run `step2_features.py` — build feature matrix, confirm X=(360000,123)
- [ ] **Step 3**: Run `step3_build_model.py` — see model architecture, count parameters
- [ ] **Step 4**: Run `step4_train.py` — train the ANN, watch loss decrease each epoch
- [ ] **Step 5**: Run `step5_evaluate.py` — check accuracy, plot training curves
- [ ] **Step 6**: Run `step6_inference.py` — test the model on a made-up game state
- [ ] **Experiment 1**: Change hidden layer size (256→512), retrain, compare accuracy
- [ ] **Experiment 2**: Add/remove a Dropout layer, observe overfitting change
- [ ] **Experiment 3**: Compare ANN accuracy vs Random Forest on same data
- [ ] **Next phase**: Add LSTM to process trick sequence as time series

---

## 💡 Key Concepts Cheat Sheet

| Term | Plain English |
|---|---|
| **Decision point** | One moment where a player chooses which card to play |
| **Feature vector** | List of 123 numbers describing the game state at that moment |
| **36-class classification** | Output: probability distribution over 36 possible cards |
| **Softmax** | Makes 36 outputs sum to 1.0 (proper probabilities) |
| **Masking** | Setting illegal card probabilities to 0 before picking best card |
| **Sparse categorical cross-entropy** | Loss when labels are integers (not one-hot) |
| **Top-3 accuracy** | Correct card is in the 3 highest-probability predictions |
| **ProBot imitation** | We train ANN to copy the expert heuristic bot's decisions |
| **Backpropagation** | Algorithm to compute how each weight affects the loss |
| **Adam optimizer** | Smart gradient descent that auto-adjusts step sizes per weight |
| **Dropout** | Randomly disables neurons during training → prevents memorisation |
| **Early stopping** | Halt training when val accuracy stops improving → saves time |
| **Overfitting** | Model memorises training data but fails on new games |
| **Epoch** | One full pass through all 360,000 training samples |
| **Batch** | Small group of samples (256) processed before each weight update |
