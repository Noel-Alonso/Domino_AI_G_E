import random
from typing import List, Optional, Tuple

Tile = Tuple[int, int]
Move = Tuple[Tile, str]
Board = List[Tile]


def create_deck() -> List[Tile]:
    """
    Generates the complete standard double-six set of 28 unique domino pieces
    (0-0 through 6-6).
    """
    return [(i, j) for i in range(7) for j in range(i, 7)]


def shuffle_deck(deck: List[Tile], seed: Optional[int] = None) -> List[Tile]:
    """
    Returns a shuffled copy of the domino deck.
    Supports an optional random seed for reproducible AI experiments.
    """
    shuffled = list(deck)
    rng = random.Random(seed)
    rng.shuffle(shuffled)
    return shuffled


def deal_hands(deck: List[Tile], num_players: int = 4, hand_size: int = 7) -> List[List[Tile]]:
    """
    Deals `hand_size` dominoes to each of the `num_players` players.
    Default: 4 players receiving 7 tiles each (28 tiles total).
    """
    if len(deck) < num_players * hand_size:
        raise ValueError(
            f"Not enough tiles in deck ({len(deck)}) to deal "
            f"{hand_size} tiles to {num_players} players."
        )
    return [
        list(deck[i * hand_size : (i + 1) * hand_size])
        for i in range(num_players)
    ]


def get_open_ends(board: Board) -> Optional[Tuple[int, int]]:
    """
    Returns (left_end, right_end) of the domino chain on the table.
    Returns None if the board is currently empty.
    """
    if not board:
        return None
    left_end = board[0][0]
    right_end = board[-1][1]
    return (left_end, right_end)


def is_move_legal(tile: Tile, side: str, board: Board) -> bool:
    """
    Determines whether placing `tile` on `side` ('LEFT' or 'RIGHT') is legal
    given the current `board`.
    """
    side = side.upper()
    if side not in ("LEFT", "RIGHT"):
        return False

    if not board:
        return True

    left_end, right_end = get_open_ends(board)
    target_pip = left_end if side == "LEFT" else right_end
    return target_pip in tile


def get_legal_moves(hand: List[Tile], board: Board) -> List[Move]:
    """
    Returns all legal (tile, side) moves available for a player's `hand`
    given the current `board`.

    Notes:
    - If the board is empty, each tile in `hand` is returned once with side 'LEFT'
      to avoid duplicate symmetric actions for AI agents.
    - If left_end == right_end on a non-empty board, both 'LEFT' and 'RIGHT'
      are included so action-space indexing remains consistent.
    """
    if not hand:
        return []

    if not board:
        return [(tile, "LEFT") for tile in hand]

    left_end, right_end = get_open_ends(board)
    legal_moves: List[Move] = []

    for tile in hand:
        if left_end in tile:
            legal_moves.append((tile, "LEFT"))
        if right_end in tile:
            legal_moves.append((tile, "RIGHT"))

    return legal_moves


def must_pass(hand: List[Tile], board: Board) -> bool:
    """
    Returns True if the player has no legal moves and must pass their turn.
    """
    return len(get_legal_moves(hand, board)) == 0


def handle_pass(hand: List[Tile], board: Board) -> Tuple[bool, str]:
    """
    Validates and handles a player's pass action.
    Returns (is_valid_pass, status_message).
    """
    if not must_pass(hand, board):
        return False, "Illegal pass: player has at least one legal move available."
    return True, "Player has no legal moves and passes turn."


def orient_tile_for_side(tile: Tile, side: str, board: Board) -> Tile:
    """
    Orients (flips if needed) `tile` so its matching pip connects cleanly to
    the open end of `board` on `side`.
    Raises ValueError if the placement is illegal.
    """
    side = side.upper()
    if not is_move_legal(tile, side, board):
        raise ValueError(f"Illegal move: tile {tile} cannot be played on {side} of board {board}.")

    if not board:
        return tile

    left_end, right_end = get_open_ends(board)
    a, b = tile

    if side == "LEFT":
        return (a, b) if b == left_end else (b, a)
    else:
        return (a, b) if a == right_end else (b, a)


def apply_move(
    hand: List[Tile], board: Board, tile: Tile, side: str
) -> Tuple[List[Tile], Board]:
    """
    Pure state-transition helper: validates the move, removes `tile` from a copy
    of `hand`, orients `tile`, and places it on `side` of a copy of `board`.

    Returns (updated_hand, updated_board) without mutating inputs, making it safe
    for Monte Carlo Tree Search (MCTS) and lookahead simulations.
    """
    if tile not in hand:
        reversed_tile = (tile[1], tile[0])
        if reversed_tile in hand:
            tile = reversed_tile
        else:
            raise ValueError(f"Tile {tile} is not in the player's hand {hand}.")

    oriented_tile = orient_tile_for_side(tile, side, board)

    updated_hand = list(hand)
    updated_hand.remove(tile)

    updated_board = list(board)
    if side.upper() == "LEFT":
        updated_board.insert(0, oriented_tile)
    else:
        updated_board.append(oriented_tile)

    return updated_hand, updated_board


def tile_pip_sum(tile: Tile) -> int:
    """Returns the pip point value of a single domino tile."""
    return tile[0] + tile[1]


def hand_pip_sum(hand: List[Tile]) -> int:
    """Returns the total pip point value remaining in a player's hand."""
    return sum(tile_pip_sum(t) for t in hand)