import math
import random
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

from domino_rules import (
    Board,
    Move,
    Tile,
    apply_move,
    get_legal_moves,
    get_open_ends,
    tile_pip_sum,
)
from game_state import DominoGameState
from scoring import (
    TARGET_SCORE,
    calculate_round_points,
    check_game_winner,
    determine_round_winner,
    update_score,
)

try:
    from gymnasium import spaces
    from pettingzoo import AECEnv

    PETTINGZOO_AVAILABLE = True
except ImportError:
    PETTINGZOO_AVAILABLE = False
    AECEnv = object


# ===========================================================================
# 1. PettingZoo Multi-Agent Environment Wrapper (Section 2: Library Integration)
# ===========================================================================

class DominoPettingZooEnv(AECEnv):
    """
    Wraps `DominoGameState` (Member 2), `domino_rules` (Member 1), and `scoring`
    (Member 3) into a standard PettingZoo Agent-Environment-Cycle (AEC) API.
    """

    metadata = {"render_modes": ["human"], "name": "domino_partnership_v1"}

    def __init__(self, scoring_mode: str = "opponents_only"):
        super().__init__()
        self.possible_agents = ["player_0", "player_1", "player_2", "player_3"]
        self.scoring_mode = scoring_mode
        self.game_state = DominoGameState()

        if PETTINGZOO_AVAILABLE:
            self.action_spaces = {a: spaces.Discrete(57) for a in self.possible_agents}
            self.observation_spaces = {
                a: spaces.Dict({"action_mask": spaces.Box(0, 1, shape=(57,), dtype=int)})
                for a in self.possible_agents
            }

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        self.agents = list(self.possible_agents)
        starting_player = options.get("starting_player") if options else None
        self.game_state.reset_round(seed=seed, starting_player=starting_player)
        self.agent_selection = f"player_{self.game_state.current_turn}"
        self.rewards = {a: 0.0 for a in self.agents}
        self._cumulative_rewards = {a: 0.0 for a in self.agents}
        self.terminations = {a: False for a in self.agents}
        self.truncations = {a: False for a in self.agents}
        self.infos = {a: {} for a in self.agents}

    def observe(self, agent: str) -> Dict[str, Any]:
        player_idx = int(agent.split("_")[1])
        return self.game_state.get_observation(player_idx)

    def step_move(self, move: Optional[Move]) -> Dict[str, Any]:
        """Advances the environment by one legal (tile, side) move or None (PASS)."""
        event = self.game_state.step(move)
        if self.game_state.is_round_over():
            winner_team = determine_round_winner(
                self.game_state.hands, self.game_state.domino_out_player
            )
            pts = calculate_round_points(
                self.game_state.hands, winner_team, scoring_mode=self.scoring_mode
            )
            for idx, agent in enumerate(self.agents):
                self.terminations[agent] = True
                if winner_team is None:
                    self.rewards[agent] = 0.0
                elif self.game_state.get_player_team(idx) == winner_team:
                    self.rewards[agent] = float(pts)
                else:
                    self.rewards[agent] = float(-pts)
        else:
            self.agent_selection = f"player_{self.game_state.current_turn}"
        return event


# ===========================================================================
# 2. Baseline & Human Agents (Section 4)
# ===========================================================================

class BaseAgent:
    name: str = "BaseAgent"

    def select_move(self, observation: Dict[str, Any]) -> Optional[Move]:
        raise NotImplementedError


class RandomAgent(BaseAgent):
    """Baseline 1: Selects uniformly at random from legal moves."""

    def __init__(self, seed: Optional[int] = None, name: str = "Random"):
        self.name = name
        self.rng = random.Random(seed)

    def select_move(self, observation: Dict[str, Any]) -> Optional[Move]:
        legal = observation["legal_moves"]
        return self.rng.choice(legal) if legal else None


class HeuristicAgent(BaseAgent):
    """
    Baseline 2: Tactical rule-based heuristic agent.
    Scores candidate moves by:
    1. Shedding high-pip tiles (reduces penalty risk if opponents domino out).
    2. Shedding doubles early (doubles only match one pip value and are harder to play).
    3. Preserving suit diversity in the remaining hand to avoid future passes.
    """

    def __init__(self, name: str = "Heuristic"):
        self.name = name

    def select_move(self, observation: Dict[str, Any]) -> Optional[Move]:
        legal = observation["legal_moves"]
        if not legal:
            return None

        hand: List[Tile] = observation["hand"]

        def evaluate_move(move: Move) -> float:
            tile, _ = move
            pips = tile_pip_sum(tile)
            double_bonus = 4.0 if tile[0] == tile[1] else 0.0

            remaining = list(hand)
            remaining.remove(tile)
            distinct_suits = len({pip for t in remaining for pip in t})
            return pips + double_bonus + (1.5 * distinct_suits)

        return max(legal, key=evaluate_move)


class HumanAgent(BaseAgent):
    """Interactive CLI agent allowing a human player to compete alongside/against AI."""

    def __init__(self, name: str = "Human"):
        self.name = name

    def select_move(self, observation: Dict[str, Any]) -> Optional[Move]:
        legal = observation["legal_moves"]
        print(f"\n--- {self.name} (Player {observation['player_idx']}, Team {observation['team']}) ---")
        print(f"Board: {observation['board']}")
        print(f"Open Ends: {observation['open_ends']}")
        print(f"Your Hand: {observation['hand']}")

        if not legal:
            print("No legal moves available. Passing turn...")
            return None

        for idx, (tile, side) in enumerate(legal):
            print(f"  [{idx}] Play {tile} on {side}")

        while True:
            raw = input(f"Select move index (0-{len(legal) - 1}): ").strip()
            if raw.isdigit() and 0 <= int(raw) < len(legal):
                return legal[int(raw)]
            print("Invalid selection. Try again.")


# ===========================================================================
# 3. Advanced AI: Averaging Over Clairvoyance + UCB1 Monte Carlo Simulation
# ===========================================================================

class ClairvoyanceMCTSAgent(BaseAgent):
    """
    Adversarial AI combining Approaches #3 (Monte Carlo Simulation) and
    #4 (Averaging Over Clairvoyance):
    1. Infers belief distribution over unseen tiles using known board tiles,
       own hand, and observed opponent/partner pass constraints.
    2. Samples `num_worlds` determinized (clairvoyant) deals.
    3. Allocates `rollouts_per_world` simulations across legal candidate moves
       using Upper Confidence Bound (UCB1) selection and fast tactical rollouts,
       maximizing expected partnership score differential.
    """

    def __init__(
        self,
        num_worlds: int = 35,
        rollouts_per_world: int = 12,
        exploration_c: float = 15.0,
        use_pass_inference: bool = True,
        scoring_mode: str = "opponents_only",
        seed: Optional[int] = None,
        name: str = "ClairvoyanceMCTS",
    ):
        self.name = name
        self.num_worlds = num_worlds
        self.rollouts_per_world = rollouts_per_world
        self.exploration_c = exploration_c
        self.use_pass_inference = use_pass_inference
        self.scoring_mode = scoring_mode
        self.rng = random.Random(seed)

    def _sample_clairvoyant_hands(self, observation: Dict[str, Any]) -> List[List[Tile]]:
        """Samples a consistent deal of `unseen_tiles` to the 3 other players."""
        me = observation["player_idx"]
        my_hand = list(observation["hand"])
        unseen = list(observation["unseen_tiles"])
        hand_sizes = observation["hand_sizes"]
        passed_pips = observation["passed_pips"]
        others = [p for p in range(4) if p != me]

        max_attempts = 25 if self.use_pass_inference else 1
        for _ in range(max_attempts):
            self.rng.shuffle(unseen)
            hands: List[List[Tile]] = [[] for _ in range(4)]
            hands[me] = list(my_hand)

            cursor = 0
            valid = True
            for p in others:
                k = hand_sizes[p]
                deal = unseen[cursor : cursor + k]
                cursor += k

                if self.use_pass_inference:
                    forbidden = passed_pips.get(p, set())
                    if any(t[0] in forbidden or t[1] in forbidden for t in deal):
                        valid = False
                        break
                hands[p] = deal

            if valid:
                return hands

        self.rng.shuffle(unseen)
        hands = [[] for _ in range(4)]
        hands[me] = list(my_hand)
        cursor = 0
        for p in others:
            k = hand_sizes[p]
            hands[p] = unseen[cursor : cursor + k]
            cursor += k
        return hands

    def _simulate_rollout(
        self,
        hands: List[List[Tile]],
        board: Board,
        next_player: int,
        my_team: int,
    ) -> float:
        """
        Simulates a full round from the post-move state using an epsilon-greedy
        tactical rollout policy and returns net utility for `my_team`.
        """
        sim_hands = [list(h) for h in hands]
        sim_board = list(board)
        turn = next_player
        consecutive_passes = 0
        domino_out_player: Optional[int] = None

        while consecutive_passes < 4:
            hand = sim_hands[turn]
            legal = get_legal_moves(hand, sim_board)

            if not legal:
                consecutive_passes += 1
            else:
                consecutive_passes = 0
                if len(legal) == 1 or self.rng.random() >= 0.20:
                    move = max(legal, key=lambda m: tile_pip_sum(m[0]))
                else:
                    move = self.rng.choice(legal)

                sim_hands[turn], sim_board = apply_move(hand, sim_board, move[0], move[1])
                if len(sim_hands[turn]) == 0:
                    domino_out_player = turn
                    break

            turn = (turn + 1) % 4

        winner_team = determine_round_winner(sim_hands, domino_out_player)
        if winner_team is None:
            return 0.0

        pts = calculate_round_points(
            sim_hands, winner_team, scoring_mode=self.scoring_mode
        )
        reward = float(pts + 10)
        return reward if winner_team == my_team else -reward

    def select_move(self, observation: Dict[str, Any]) -> Optional[Move]:
        legal_moves: List[Move] = observation["legal_moves"]
        if not legal_moves:
            return None
        if len(legal_moves) == 1:
            return legal_moves[0]

        me = observation["player_idx"]
        my_team = observation["team"]
        my_hand = observation["hand"]
        board = observation["board"]

        for move in legal_moves:
            if len(my_hand) == 1:
                return move

        total_values: Dict[Move, float] = {m: 0.0 for m in legal_moves}
        visit_counts: Dict[Move, int] = {m: 0 for m in legal_moves}

        for _ in range(self.num_worlds):
            world_hands = self._sample_clairvoyant_hands(observation)

            for step_idx in range(max(self.rollouts_per_world, len(legal_moves))):
                unvisited = [m for m in legal_moves if visit_counts[m] == 0]
                if unvisited:
                    chosen_move = unvisited[0]
                else:
                    total_visits = sum(visit_counts.values())
                    chosen_move = max(
                        legal_moves,
                        key=lambda m: (total_values[m] / visit_counts[m])
                        + self.exploration_c * math.sqrt(math.log(total_visits) / visit_counts[m]),
                    )

                new_hand, new_board = apply_move(
                    my_hand, board, chosen_move[0], chosen_move[1]
                )
                sim_hands = [list(h) for h in world_hands]
                sim_hands[me] = new_hand

                utility = self._simulate_rollout(
                    sim_hands, new_board, next_player=(me + 1) % 4, my_team=my_team
                )
                total_values[chosen_move] += utility
                visit_counts[chosen_move] += 1

        return max(legal_moves, key=lambda m: total_values[m] / max(1, visit_counts[m]))


# ===========================================================================
# 4. Automated Partnership Tournament & Evaluation Suite (Sections 4 & 5)
# ===========================================================================

def play_full_match(
    agents: Sequence[BaseAgent],
    target_score: int = TARGET_SCORE,
    scoring_mode: str = "opponents_only",
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Plays a complete multi-round partnership Domino match up to `target_score`.
    Seats: Team 1 = (agents[0], agents[2]), Team 2 = (agents[1], agents[3]).
    """
    rng = random.Random(seed)
    scores = {1: 0, 2: 0}
    rounds_played = 0
    starting_player: Optional[int] = None

    while check_game_winner(scores, target_score=target_score) is None:
        round_seed = rng.randint(0, 10**9)
        state = DominoGameState(seed=round_seed, starting_player=starting_player)

        while not state.is_round_over():
            active = state.current_turn
            obs = state.get_observation(active)
            move = agents[active].select_move(obs)
            state.step(move)

        winner_team = determine_round_winner(state.hands, state.domino_out_player)
        pts = calculate_round_points(state.hands, winner_team, scoring_mode=scoring_mode)
        update_score(scores, winner_team, pts)

        rounds_played += 1
        starting_player = (state.move_history[0]["player"] + 1) % 4 if state.move_history else 0

    match_winner = check_game_winner(scores, target_score=target_score)
    return {
        "winner_team": match_winner,
        "team1_score": scores[1],
        "team2_score": scores[2],
        "score_diff": scores[1] - scores[2],
        "rounds": rounds_played,
    }


def evaluate_matchup(
    label: str,
    team1_agents: Tuple[BaseAgent, BaseAgent],
    team2_agents: Tuple[BaseAgent, BaseAgent],
    num_matches: int = 100,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Evaluates a team matchup over `num_matches` full games (to 100 points)
    and computes win rate, average score differential, and runtime metrics.
    """
    lineup = [team1_agents[0], team2_agents[0], team1_agents[1], team2_agents[1]]
    t1_wins = 0
    score_diffs: List[int] = []
    round_counts: List[int] = []

    t0 = time.perf_counter()
    for match_idx in range(num_matches):
        res = play_full_match(lineup, seed=seed + match_idx)
        if res["winner_team"] == 1:
            t1_wins += 1
        score_diffs.append(res["score_diff"])
        round_counts.append(res["rounds"])
    elapsed = time.perf_counter() - t0

    return {
        "matchup": label,
        "matches": num_matches,
        "t1_win_rate": round(100.0 * t1_wins / num_matches, 2),
        "mean_score_diff": round(sum(score_diffs) / num_matches, 2),
        "mean_rounds": round(sum(round_counts) / num_matches, 2),
        "elapsed_sec": round(elapsed, 2),
    }


def run_full_tournament(num_matches: int = 100, seed: int = 42) -> List[Dict[str, Any]]:
    """
    Executes all required Section 4 & Section 5 partnership experiments:
    - Baseline comparisons (vs Random, vs Heuristic)
    - Partner sensitivity (AI + Random, AI + Heuristic)
    - Ablation study (With vs Without Pass-Constraint Inference)
    - Self-play symmetry check (AI + AI vs AI + AI)
    """
    ai_strong = ClairvoyanceMCTSAgent(num_worlds=30, rollouts_per_world=10, use_pass_inference=True, seed=seed)
    ai_partner = ClairvoyanceMCTSAgent(num_worlds=30, rollouts_per_world=10, use_pass_inference=True, seed=seed + 1)
    ai_no_infer = ClairvoyanceMCTSAgent(num_worlds=30, rollouts_per_world=10, use_pass_inference=False, seed=seed + 2)
    ai_opp1 = ClairvoyanceMCTSAgent(num_worlds=30, rollouts_per_world=10, use_pass_inference=True, seed=seed + 3)
    ai_opp2 = ClairvoyanceMCTSAgent(num_worlds=30, rollouts_per_world=10, use_pass_inference=True, seed=seed + 4)

    heur_1, heur_2 = HeuristicAgent("Heuristic-1"), HeuristicAgent("Heuristic-2")
    rand_1, rand_2 = RandomAgent(seed=seed + 10, name="Random-1"), RandomAgent(seed=seed + 11, name="Random-2")

    scenarios = [
        ("(AI + AI) vs (Random + Random)", (ai_strong, ai_partner), (rand_1, rand_2)),
        ("(AI + AI) vs (Heuristic + Heuristic)", (ai_strong, ai_partner), (heur_1, heur_2)),
        ("(AI + Random) vs (Random + Random) [Weak Partner]", (ai_strong, rand_1), (rand_2, RandomAgent(seed + 12))),
        ("(AI + Random) vs (Heuristic + Heuristic) [Weak Partner]", (ai_strong, rand_1), (heur_1, heur_2)),
        ("(AI + Heuristic) vs (Heuristic + Heuristic) [Med Partner]", (ai_strong, heur_1), (heur_2, HeuristicAgent("Heuristic-3"))),
        ("(AI No-Pass-Infer + AI) vs (Heuristic + Heuristic) [Ablation]", (ai_no_infer, ai_partner), (heur_1, heur_2)),
        ("(AI + AI) vs (AI + AI) [Self-Play Balance]", (ai_strong, ai_partner), (ai_opp1, ai_opp2)),
    ]

    results = []
    print(f"{'Matchup':<62} | {'T1 Win %':>8} | {'Avg Diff':>8} | {'Rounds':>6} | {'Time(s)':>7}")
    print("-" * 102)
    for label, t1, t2 in scenarios:
        summary = evaluate_matchup(label, t1, t2, num_matches=num_matches, seed=seed)
        results.append(summary)
        print(
            f"{summary['matchup']:<62} | "
            f"{summary['t1_win_rate']:7.2f}% | "
            f"{summary['mean_score_diff']:+8.2f} | "
            f"{summary['mean_rounds']:6.2f} | "
            f"{summary['elapsed_sec']:7.2f}"
        )
    return results


if __name__ == "__main__":
    run_full_tournament(num_matches=100, seed=42)