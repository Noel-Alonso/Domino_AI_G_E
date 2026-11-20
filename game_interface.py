import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ai_agents import (
    BaseAgent,
    ClairvoyanceMCTSAgent,
    HeuristicAgent,
    RandomAgent,
)
from domino_rules import Board, Move, Tile, get_open_ends, hand_pip_sum
from game_state import DominoGameState
from scoring import (
    TARGET_SCORE,
    calculate_round_points,
    calculate_team_points,
    check_game_winner,
    determine_round_winner,
    get_player_team,
    update_score,
)


# ===========================================================================
# 1. Visual Formatting & Display Helpers
# ===========================================================================

def format_tile(tile: Tile) -> str:
    """Formats a domino tuple (a, b) as [a|b]."""
    return f"[{tile[0]}|{tile[1]}]"


def format_hand(hand: List[Tile]) -> str:
    """Formats a player's list of dominoes cleanly."""
    if not hand:
        return "(Empty Hand)"
    return "  ".join(f"{idx}:{format_tile(t)}" for idx, t in enumerate(hand))


def format_board(board: Board) -> str:
    """Formats the domino chain currently placed on the table."""
    if not board:
        return "<Table is empty — any tile may be played>"
    return "-".join(format_tile(t) for t in board)


def display_game_state(
    state: DominoGameState,
    scores: Dict[int, int],
    round_num: int,
    human_players: Optional[List[int]] = None,
    reveal_all_hands: bool = False,
) -> None:
    """
    Displays the complete table state:
    - Match scores for Team 1 (P0 & P2) and Team 2 (P1 & P3)
    - All 4 players, their tile counts, and visible hands
    - The domino chain on the table and its left/right open ends
    - Whose turn it currently is
    """
    human_players = human_players or []
    active = state.current_turn
    active_team = state.get_player_team(active)
    open_ends = get_open_ends(state.board)

    print("\n" + "=" * 78)
    print(
        f"ROUND {round_num}  |  MATCH SCORE -> "
        f"Team 1 (P0 & P2): {scores[1]} pts  |  "
        f"Team 2 (P1 & P3): {scores[2]} pts  (Target: {TARGET_SCORE})"
    )
    print("-" * 78)

    for p_idx in range(4):
        team_id = state.get_player_team(p_idx)
        hand = state.get_hand(p_idx)
        turn_marker = ">>> TURN >>>" if p_idx == active else "            "

        if reveal_all_hands or p_idx in human_players:
            hand_display = format_hand(hand)
        else:
            hidden_tiles = " ".join(["[?|?]"] * len(hand)) if hand else "(Empty)"
            hand_display = f"{hidden_tiles} ({len(hand)} tiles)"

        print(f"{turn_marker} {state.player_names[p_idx]:<18} [Team {team_id}]: {hand_display}")

    print("-" * 78)
    print(f"TABLE CHAIN : {format_board(state.board)}")
    if open_ends is not None:
        print(f"OPEN ENDS   : Left = [{open_ends[0]}]   |   Right = [{open_ends[1]}]")
    print(f"ACTIVE TURN : {state.player_names[active]} (Team {active_team})")
    print("=" * 78)


# ===========================================================================
# 2. Interactive Human Player Interface
# ===========================================================================

class InteractiveHumanAgent(BaseAgent):
    """
    Allows a human player to view their hand, inspect legal moves, and select
    a domino and board side ('LEFT' or 'RIGHT') via console input.
    """

    def __init__(self, name: str = "Human"):
        self.name = name

    def select_move(self, observation: Dict[str, Any]) -> Optional[Move]:
        legal: List[Move] = observation["legal_moves"]
        hand: List[Tile] = observation["hand"]

        print(f"\n[{self.name}] Your Hand: {format_hand(hand)}")

        if not legal:
            print(f"[{self.name}] You have NO legal moves matching open ends {observation['open_ends']}.")
            input("Press Enter to PASS your turn...")
            return None

        print("Legal Moves:")
        for idx, (tile, side) in enumerate(legal):
            print(f"  ({idx}) Play {format_tile(tile)} on the {side} end")

        while True:
            choice = input(f"Choose move number (0-{len(legal) - 1}): ").strip()
            if choice.isdigit():
                move_idx = int(choice)
                if 0 <= move_idx < len(legal):
                    selected = legal[move_idx]
                    print(f"-> You selected {format_tile(selected[0])} on {selected[1]}.")
                    return selected
            print(f"Invalid input. Please enter a number between 0 and {len(legal) - 1}.")


# ===========================================================================
# 3. Full Game & Match Controller (Integrates Members 1, 2, 3, & AI)
# ===========================================================================

class DominoGameController:
    """
    Orchestrates complete 4-player partnership Domino matches up to `target_score`,
    connecting rules, state management, scoring, display, and AI/Human agents.
    """

    def __init__(
        self,
        agents: Sequence[BaseAgent],
        player_names: Optional[List[str]] = None,
        target_score: int = TARGET_SCORE,
        scoring_mode: str = "opponents_only",
        reveal_all_hands: bool = False,
        turn_delay_sec: float = 0.4,
    ):
        if len(agents) != 4:
            raise ValueError("Domino requires exactly 4 agents (Team 1: 0 & 2, Team 2: 1 & 3).")

        self.agents = list(agents)
        self.player_names = player_names or [
            f"P0 ({self.agents[0].name})",
            f"P1 ({self.agents[1].name})",
            f"P2 ({self.agents[2].name})",
            f"P3 ({self.agents[3].name})",
        ]
        self.target_score = target_score
        self.scoring_mode = scoring_mode
        self.reveal_all_hands = reveal_all_hands
        self.turn_delay_sec = turn_delay_sec
        self.human_indices = [
            idx for idx, agent in enumerate(self.agents) if isinstance(agent, InteractiveHumanAgent)
        ]

    def play_match(self, seed: Optional[int] = None) -> Dict[str, Any]:
        """Runs rounds until a team reaches `self.target_score` and announces the winner."""
        scores = {1: 0, 2: 0}
        round_num = 1
        starting_player: Optional[int] = None

        print("\n" + "#" * 78)
        print("STARTING 4-PLAYER PARTNERSHIP DOMINO MATCH")
        print(f"Team 1: {self.player_names[0]} & {self.player_names[2]}")
        print(f"Team 2: {self.player_names[1]} & {self.player_names[3]}")
        print("#" * 78)

        while check_game_winner(scores, target_score=self.target_score) is None:
            round_seed = (seed + round_num) if seed is not None else None
            state = DominoGameState(
                player_names=self.player_names,
                seed=round_seed,
                starting_player=starting_player,
            )

            print(f"\n*** Beginning Round {round_num} (Lead: {self.player_names[state.current_turn]}) ***")

            while not state.is_round_over():
                active = state.current_turn
                display_game_state(
                    state=state,
                    scores=scores,
                    round_num=round_num,
                    human_players=self.human_indices,
                    reveal_all_hands=self.reveal_all_hands,
                )

                obs = state.get_observation(active)
                move = self.agents[active].select_move(obs)
                event = state.step(move)

                if event["action"] == "PASS":
                    print(f">> {self.player_names[active]} PASSED (no match for {event['open_ends']}).")
                else:
                    print(
                        f">> {self.player_names[active]} PLAYED "
                        f"{format_tile(event['tile'])} on {event['side']}."
                    )

                if self.turn_delay_sec > 0 and active not in self.human_indices:
                    time.sleep(self.turn_delay_sec)

            winner_team = determine_round_winner(state.hands, state.domino_out_player)
            pts_awarded = calculate_round_points(
                state.hands, winner_team, scoring_mode=self.scoring_mode
            )
            update_score(scores, winner_team, pts_awarded)
            t1_pips, t2_pips = calculate_team_points(state.hands)

            print("\n" + "*" * 78)
            print(f"ROUND {round_num} COMPLETE!")
            if state.domino_out_player is not None:
                print(f"Reason: {self.player_names[state.domino_out_player]} DOMINOED OUT!")
            else:
                print("Reason: BLOCKED GAME (TRANCA) — 4 consecutive passes.")

            for p_idx in range(4):
                print(
                    f"  {self.player_names[p_idx]:<20}: "
                    f"{format_hand(state.hands[p_idx])} (Pips: {hand_pip_sum(state.hands[p_idx])})"
                )
            print(f"Remaining Pips -> Team 1: {t1_pips} | Team 2: {t2_pips}")
            if winner_team is not None:
                print(f"Round Winner: TEAM {winner_team} (+{pts_awarded} points)")
            else:
                print("Round Result: TIE (0 points awarded)")
            print(f"Updated Match Score -> Team 1: {scores[1]} | Team 2: {scores[2]}")
            print("*" * 78)

            lead_player = state.move_history[0]["player"] if state.move_history else 0
            starting_player = (lead_player + 1) % 4
            round_num += 1

        champion = check_game_winner(scores, target_score=self.target_score)
        print("\n" + "=" * 78)
        print(f"MATCH OVER! CHAMPION: TEAM {champion}")
        print(f"Final Score -> Team 1: {scores[1]}  |  Team 2: {scores[2]}")
        print("=" * 78)

        return {
            "winner_team": champion,
            "final_scores": dict(scores),
            "rounds_played": round_num - 1,
        }


# ===========================================================================
# 4. Interactive Launcher Menu (CLI Entry Point)
# ===========================================================================

def launch_interactive_menu() -> None:
    """Provides a simple menu to play as a Human or watch an AI Showcase."""
    print("\n=== ICOM5015 Partnership Domino — Interactive Launcher ===")
    print("1. Play as Human (Team 1: You + ClairvoyanceMCTS AI vs. 2 ClairvoyanceMCTS AIs)")
    print("2. Play as Human (Team 1: You + ClairvoyanceMCTS AI vs. 2 Heuristic AIs)")
    print("3. Spectator Demo Mode (All 4 hands visible: AI+AI vs. Heuristic+Heuristic)")

    choice = input("Select mode (1, 2, or 3): ").strip()

    if choice == "1":
        agents = [
            InteractiveHumanAgent("You"),
            ClairvoyanceMCTSAgent(num_worlds=25, rollouts_per_world=8, seed=101, name="AI-Opp1"),
            ClairvoyanceMCTSAgent(num_worlds=25, rollouts_per_world=8, seed=102, name="AI-Partner"),
            ClairvoyanceMCTSAgent(num_worlds=25, rollouts_per_world=8, seed=103, name="AI-Opp2"),
        ]
        controller = DominoGameController(agents, reveal_all_hands=False, turn_delay_sec=0.5)
    elif choice == "2":
        agents = [
            InteractiveHumanAgent("You"),
            HeuristicAgent("Heuristic-1"),
            ClairvoyanceMCTSAgent(num_worlds=25, rollouts_per_world=8, seed=102, name="AI-Partner"),
            HeuristicAgent("Heuristic-2"),
        ]
        controller = DominoGameController(agents, reveal_all_hands=False, turn_delay_sec=0.5)
    else:
        agents = [
            ClairvoyanceMCTSAgent(num_worlds=20, rollouts_per_world=8, seed=42, name="AI-P0"),
            HeuristicAgent("Heuristic-P1"),
            ClairvoyanceMCTSAgent(num_worlds=20, rollouts_per_world=8, seed=43, name="AI-P2"),
            HeuristicAgent("Heuristic-P3"),
        ]
        controller = DominoGameController(agents, reveal_all_hands=True, turn_delay_sec=0.2)

    controller.play_match(seed=42)


if __name__ == "__main__":
    launch_interactive_menu()