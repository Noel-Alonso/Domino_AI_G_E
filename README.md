# Partnership Domino AI: Adversarial Reasoning Under Partial Observability

> **Course:** ICOM5015 — Artificial Intelligence (University of Puerto Rico at Mayagüez)  
> **Domain:** 4-Player Team-Based Domino (Double-Six Set, Target Score: 100 Points)  
> **AI Paradigm:** Averaging Over Clairvoyance + UCB1 Monte Carlo Simulation  
> **Multi-Agent Framework:** PettingZoo (`AECEnv`)

---

## 1. Executive Overview

In 4-player partnership Domino, two cross-table teams (**Team 1: Players 0 & 2** vs. **Team 2: Players 1 & 3**) compete under **imperfect information** and **coupled team dynamics**. At the start of a round, each player observes only their own 7 tiles while 21 tiles remain hidden across three other hands—yielding $\binom{21}{7}\binom{14}{7} = 399{,}072{,}960$ possible hidden deal configurations on Turn 1 alone.

This repository implements a modular, tournament-ready Domino engine integrated with **PettingZoo's Agent-Environment-Cycle (`AECEnv`)** API and an adversarial AI agent (**`ClairvoyanceMCTSAgent`**) that combines **belief-state determinization (Averaging Over Clairvoyance)** with **Upper Confidence Bound (UCB1) Monte Carlo rollouts** to optimize long-term partnership utility.

---

## 2. AI Methodology & Library Justification

### Chosen Algorithmic Approach
* **Techniques Used:** **Monte Carlo Simulation** & **Averaging Over Clairvoyance** (Syllabus Approaches #3 and #4).
* **Why This Approach Fits Domino:**
  1. **Intractability of Exact Search:** Pure Minimax and Expectiminimax fail due to branching factor explosion across hidden opponent and partner hands.
  2. **Logical Belief Filtering:** Rather than sampling hidden hands uniformly at random, our agent tracks every open board end on which a player has previously **passed**, eliminating impossible tiles from sampled worlds via rejection sampling.
  3. **Team-Utility Rollouts:** Within each sampled clairvoyant deal, UCB1 allocates simulations across legal moves and evaluates complete round outcomes (both *Domino-Out* and *Blocked Game / Tranca* scenarios) from the partnership's cumulative score perspective.

### Library Integration
* **PettingZoo (`pettingzoo.AECEnv`):** Wraps our Domino state machine into a standardized multi-agent cycle (`DominoPettingZooEnv`), enforcing strict partial observability (`observe(agent)`), legal action masking, and zero-sum team rewards while keeping baseline, heuristic, human, and MCTS agents interchangeable.
* **Matplotlib & Python Standard Library (`random`, `math`, `time`):** Used for deterministic pseudorandom seeding (`seed=42`), UCB1 bandit selection, high-precision benchmarking, and publication-grade tournament visualizations.

---

## 3. System Architecture & Team Responsibilities

```text
Domino_AI_G_E/
├── domino_rules.py          # [Member 1] 28-tile deck, shuffling, dealing, legal moves, tile orientation 
├── game_state.py            # [Member 2] Cross-table teams, turn rotation, state cloning, pass tracking 
├── scoring.py               # [Member 3] Team pip sums, Domino-Out vs. Tranca resolution, 100-pt match end 
├── ai_agents.py             # [AI Core]  PettingZoo AECEnv, ClairvoyanceMCTSAgent, Baselines, Tournament 
├── game_interface.py        # [Member 4] CLI table renderer, Human interactive input, match controller 
├── Domino_AI_Report.ipynb   # [All]      Interactive evaluation notebook, metrics, and figures 
└── README.md                # [All]      Installation, architecture, and reproducibility documentation 