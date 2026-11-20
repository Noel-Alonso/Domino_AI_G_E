from typing import Dict, List, Optional, Tuple
from domino_rules import Tile, hand_pip_sum

TARGET_SCORE = 100

TEAM_1_PLAYERS = (0, 2)
TEAM_2_PLAYERS = (1, 3)


def get_player_team(player_idx: int) -> int:
    """Returns 1 for Players 0 and 2; returns 2 for Players 1 and 3."""
    if player_idx in TEAM_1_PLAYERS:
        return 1
    elif player_idx in TEAM_2_PLAYERS:
        return 2
    raise ValueError(f"Invalid player index {player_idx}. Expected 0, 1, 2, or 3.")


def calculate_team_points(player_hands: List[List[Tile]]) -> Tuple[int, int]:
    """
    Calculates the total remaining pip points in each team's hands.
    Team 1: Players 0 and 2
    Team 2: Players 1 and 3
    """
    team1_points = hand_pip_sum(player_hands[0]) + hand_pip_sum(player_hands[2])
    team2_points = hand_pip_sum(player_hands[1]) + hand_pip_sum(player_hands[3])
    return team1_points, team2_points


def determine_round_winner(
    player_hands: List[List[Tile]],
    domino_out_player: Optional[int] = None,
) -> Optional[int]:
    """
    Determines which team wins the round:
    1. Domino Out: If a player played their last tile (hand is empty), that
       player's team wins immediately regardless of remaining partner pips.
    2. Blocked Game (Tranca): If nobody emptied their hand and all 4 players
       passed consecutively, the team with fewer total remaining pips wins.
       Returns None if both teams tie in a blocked game.
    """
    if domino_out_player is not None:
        return get_player_team(domino_out_player)

    for idx, hand in enumerate(player_hands):
        if len(hand) == 0:
            return get_player_team(idx)

    team1_points, team2_points = calculate_team_points(player_hands)

    if team1_points < team2_points:
        return 1
    elif team2_points < team1_points:
        return 2
    return None


def calculate_round_points(
    player_hands: List[List[Tile]],
    winning_team: Optional[int],
    scoring_mode: str = "opponents_only",
) -> int:
    """
    Calculates the points awarded to `winning_team`.

    Parameters:
    - scoring_mode="opponents_only": Winning team scores the losing team's
      remaining pips (standard international partnership scoring).
    - scoring_mode="all_remaining": Winning team scores all remaining pips
      from both teams (common Caribbean/Puerto Rican rule).
    """
    if winning_team is None:
        return 0

    team1_points, team2_points = calculate_team_points(player_hands)

    if scoring_mode == "all_remaining":
        return team1_points + team2_points

    if winning_team == 1:
        return team2_points
    elif winning_team == 2:
        return team1_points
    return 0


def update_score(
    scores: Dict[int, int], winning_team: Optional[int], points: int
) -> Dict[int, int]:
    """Updates and returns the cumulative match score dictionary."""
    if winning_team in (1, 2):
        scores[winning_team] += points
    return scores


def check_game_winner(
    scores: Dict[int, int], target_score: int = TARGET_SCORE
) -> Optional[int]:
    """
    Returns the winning team (1 or 2) once a team reaches or exceeds `target_score`,
    or None if the match is still in progress.
    """
    for team in (1, 2):
        if scores.get(team, 0) >= target_score:
            return team
    return None