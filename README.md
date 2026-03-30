# DQN Assignment 3 — Deep Q-Network on Atari Pong

**Course:** CSCN 8020 — Reinforcement Learning Programming  
**Assignment:** Assignment 3  


---

## Author

| Field | Value |
|---|---|
| **Name** | Aiswarya Thekkuveettil Thazhath |
| **Student ID** | 8993970 |


---

## Assignment Summary

This project implements the **Deep Q-Network (DQN)** algorithm to train a reinforcement
learning agent to play the Atari game **Pong** (`ALE/Pong-v5`) from raw pixel observations.

The agent uses a **Convolutional Neural Network (CNN)** to approximate Q-values,
taking the last **4 stacked grayscale frames** as input instead of a single frame,
giving the network temporal context to infer ball velocity and direction.

### Key Features

- **GPU accelerated** — automatically uses CUDA if available
- **4-frame stacking** as input channels (not blending)
- **Experience Replay Buffer** — uniform random sampling
- **Target Network** — updated every N episodes for stable learning
- **ε-greedy exploration** with exponential decay applied per step (as per assignment spec)
- **Gradient clipping** (max norm = 10) for training stability
- Uses **professor-provided** `assignment3_utils.py` for preprocessing

---

## Network Architecture

| Layer | Type | Details |
|---|---|---|
| Input | — | (batch, 4, 84, 80) — 4 stacked grayscale frames |
| Conv1 | Conv2d | 32 filters, 8×8 kernel, stride 4 → ReLU |
| Conv2 | Conv2d | 64 filters, 4×4 kernel, stride 2 → ReLU |
| Conv3 | Conv2d | 64 filters, 3×3 kernel, stride 1 → ReLU |
| Flatten | — | Flattens 3D conv output to 1D vector |
| FC1 | Linear | 512 units → ReLU |
| FC2 | Linear | 6 units (one Q-value per action, linear output) |

---

## Preprocessing Pipeline

Each raw frame `(210, 160, 3)` is processed through the following pipeline
using the professor-provided `assignment3_utils.py`:

| Step | Function | Input Shape | Output Shape | Description |
|---|---|---|---|---|
| 1. Crop | `img_crop` | (210, 160, 3) | (168, 160, 3) | Removes score bar (top 30px) and padding (bottom 12px) |
| 2. Downsample | `downsample` | (168, 160, 3) | (84, 80, 3) | Takes every other pixel — halves resolution |
| 3. Grayscale | `to_grayscale` | (84, 80, 3) | (84, 80) | Averages RGB channels to single channel |
| 4. Normalize | `normalize_grayscale` | (84, 80) | (84, 80) | Scales pixel values from [0,255] to [-1, 1] |
| 5. Stack | `FrameStack` | (84, 80) × 4 | (4, 84, 80) | Last 4 frames stacked as channels for CNN input |

> **Why 84×80?** Applying `img_crop` changes the height from 210 to 168,
> then downsampling halves it to **84**. Without crop it would be 105×80.

---

## Hyperparameters

| Parameter | Value |
|---|---|
| Mini-batch size | 8 (default) |
| Target network update | every 10 episodes (default) |
| Discount factor γ | 0.95 |
| ε initial | 1.0 |
| ε decay δ | 0.995 (per step) |
| ε minimum | 0.05 |
| Replay buffer size | 10,000 |
| Learning rate | 1e-4 (Adam) |
| Training episodes | 400 |

---

## Experiments

| Experiment | Batch Size | Target Update | Description |
|---|---|---|---|
| 1 — Default | 8 | 10 episodes | Baseline run as per assignment spec |
| 2 — Batch comparison | 16 | 10 episodes | Effect of larger mini-batch size |
| 3 — Target update comparison | 8 | 3 episodes | Effect of more frequent target sync |

---

## Project Structure

```
DQN_Assignment3/
│
├── DQN_Assignment3.ipynb     # Main Jupyter Notebook (OOP implementation + all experiments)
├── assignment3_utils.py      # Professor-provided preprocessing utilities
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

---

## How to Run

### 1. Clone the repository

```bash
git clone " https://github.com/AiswaryaSukumar/DQN_Assignment3_8993970.git "
cd DQN_Assignment3
```

### 2. Create and activate a virtual environment

```bash
# Windows
python -m venv .venv
.\.venv\Scripts\activate

# Mac / Linux
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Install Atari ROMs

```bash
pip install autorom
autorom --accept-license
```

### 5. Install GPU-enabled PyTorch (recommended)

Visit https://pytorch.org/get-started/locally/ and pick your CUDA version:

```bash
# CUDA 12.4 example:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

### 6. Launch the notebook

```bash
jupyter notebook DQN_Assignment3.ipynb
```

Run all cells in order from top to bottom. Each experiment section is clearly labelled.

---

## Requirements

- Python 3.10+
- PyTorch 2.0+ (CUDA build recommended)
- Gymnasium 0.29.1
- ale-py 0.9.1
- NumPy 1.24+
- Matplotlib 3.7+

See `requirements.txt` for the complete list.

---

## Results Summary

| Experiment | Config | Final Score | Final Avg-5 |
|---|---|---|---|
| Experiment 1 | Batch=8, Target=10 (default) | -18.0 | -15.40 |
| Experiment 2 | Batch=16, Target=10 | -17.0 | -16.20 |
| Experiment 3 | Batch=8, Target=3 | -13.0 | -15.00 |

**Best configuration:** Batch=16, Target update=10 episodes  
*(Batch=16 achieved faster early improvement and higher peak scores.
Target=10 produced the most stable learning curve with least oscillation.)*

---

## References

- Mnih, V. et al. (2015). *Human-level control through deep reinforcement learning.* Nature, 518, 529–533.
- Farama Foundation — Gymnasium: https://gymnasium.farama.org
- Professor utility file: `assignment3_utils.py`
