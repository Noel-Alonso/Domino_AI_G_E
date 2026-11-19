
from scoring import (
    calculate_team_points,
    determine_round_winner,
    calculate_round_points,
    update_score,
    check_game_winner,
    TARGET_SCORE
)


def test_scoring():

    player_hands = [
        [(6, 4), (3, 2)],  # Player 1
        [(4, 3), (1, 1)],  # Player 2
        [(5, 0), (2, 1)],  # Player 3
        [(6, 2), (0, 0)]   # Player 4
    ]

    # Test team points
    assert calculate_team_points(player_hands) == (24, 16)

    # Test round winner
    assert determine_round_winner(player_hands) == 2

    # Test awarded points
    assert calculate_round_points(player_hands, 2) == 24

    # Test score update
    scores = {1: 0, 2: 0}
    update_score(scores, 2, 24)
    assert scores[2] == 24

    # Test game winner
    scores = {1: 100, 2: 24}
    assert check_game_winner(scores) == 1

    # Test no winner yet
    scores = {1: 50, 2: 24}
    assert check_game_winner(scores) is None

    print("All scoring tests passed!")


if __name__ == "__main__":
    test_scoring()
