
# scoring.py

from domino_rules import hand_pip_sum

TARGET_SCORE = 100

def calculate_team_points(player_hands):
    """
    Calculate the remaining points for each team.

    Team 1: Players 1 and 2 (indexes 0 and 1)
    Team 2: Players 3 and 4 (indexes 2 and 3)
    """

    team1_points = (
        hand_pip_sum(player_hands[0]) + 
        hand_pip_sum(player_hands[1])
    )

    team2_points = (
        hand_pip_sum(player_hands[2]) +
        hand_pip_sum(player_hands[3])
    )

    return team1_points, team2_points


def determine_round_winner(player_hands):
    """
    Determine the winner based on remaining points.

    The team with fewer remaining points wins.
    Returns 1, 2, or None if tied.
    """

    team1_points, team2_points = calculate_team_points(player_hands)

    if team1_points < team2_points:
        return 1

    elif team2_points < team1_points:
        return 2

    else:
        return None


def calculate_round_points(player_hands, winning_team):
    """
    Calculate the points awarded to the winning team.

    The winning team receives the total remaining
    points in the opposing team's hands.
    """

    team1_points, team2_points = calculate_team_points(player_hands)

    if winning_team == 1:
        return team2_points

    elif winning_team == 2:
        return team1_points

    return 0


def update_score(scores, winning_team, points):
    """
    Update the cumulative score.
    """

    if winning_team is not None:
        scores[winning_team] += points

    return scores


def check_game_winner(scores):
    for team, score in scores.items():
        if score >= TARGET_SCORE:
            return team

    return None
