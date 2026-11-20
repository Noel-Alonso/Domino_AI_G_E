from typing import Any, Dict, List, Optional, Set, Tuple
from domino_rules import (
    Board,
    Move,
    Tile,
    apply_move,
    create_deck,
    deal_hands,
    get_legal_moves,
    get_open_ends,
    handle_pass,
    must_pass,
    shuffle_deck,
)


def _canonical_tile(tile: Tile) -> Tile:
    """Normalizes (a, b) so (6, 4) and (4, 6) compare equal."""
    return (min(tile), max(tile))


class DominoGameState:
    """
    Complete state machine for a 4-player partnership Domino round.
    """

    def __init__(
        self,
        player_names: Optional[List[str]] = None,
        seed: Optional[int] = None,
        starting_player: Optional[int] = None,
    ):
        self.player_names: List[str] = player_names or [
            "Player 0 (T1)",
            "Player 1 (T2)",
            "Player 2 (T1)",
            "Player 3 (T2)",
        ]
        self.teams: Dict[int, Tuple[int, int]] = {1: (0, 2), 2: (1, 3)}

        self.hands: List[List[Tile]] = [[] for _ in range(4)]
        self.board: Board = []
        self.current_turn: int = 0
        self.consecutive_passes: int = 0
        self.domino_out_player: Optional[int] = None
        self.move_history: List[Dict[str, Any]] = []

        self.passed_pips: Dict[int, Set[int]] = {p: set() for p in range(4)}

        self.reset_round(seed=seed, starting_player=starting_player)

    def reset_round(
        self, seed: Optional[int] = None, starting_player: Optional[int] = None
    ) -> None:
        """
        Creates, shuffles, and deals a fresh 28-tile deck to the 4 players.
        If `starting_player` is None, the player holding (6, 6) leads first.
        """
        deck = shuffle_deck(create_deck(), seed=seed)
        self.hands = deal_hands(deck, num_players=4, hand_size=7)
        self.board = []
        self.consecutive_passes = 0
        self.domino_out_player = None
        self.move_history = []
        self.passed_pips = {p: set() for p in range(4)}

        if starting_player is not None:
            self.current_turn = starting_player % 4
        else:
            self.current_turn = 0
            for idx, hand in enumerate(self.hands):
                if (6, 6) in hand:
                    self.current_turn = idx
                    break

    def get_player_team(self, player_idx: int) -> int:
        """Returns 1 if player_idx in (0, 2), else 2."""
        return 1 if player_idx in self.teams[1] else 2

    def get_partner(self, player_idx: int) -> int:
        """Returns the teammate index for `player_idx`."""
        return (player_idx + 2) % 4

    def get_hand(self, player_idx: int) -> List[Tile]:
        """Returns a copy of `player_idx`'s current hand."""
        return list(self.hands[player_idx])

    def get_hand_sizes(self) -> List[int]:
        """Returns the number of tiles remaining in each player's hand."""
        return [len(h) for h in self.hands]

    def get_legal_moves_for_current_player(self) -> List[Move]:
        """Returns all legal (tile, side) moves for the active player."""
        return get_legal_moves(self.hands[self.current_turn], self.board)

    def get_remaining_unplayed_tiles(self) -> List[Tile]:
        """
        Returns all tiles from the 28-tile deck that have NOT yet been played
        on the table (i.e., tiles currently held across all 4 players' hands).
        """
        played = {_canonical_tile(t) for t in self.board}
        return [t for t in create_deck() if _canonical_tile(t) not in played]

    def get_unseen_tiles_for_player(self, player_idx: int) -> List[Tile]:
        """
        Returns all tiles hidden from `player_idx` (excludes tiles on the table
        and tiles in `player_idx`'s own hand). Used for AI belief sampling.
        """
        known = {_canonical_tile(t) for t in self.board} | {
            _canonical_tile(t) for t in self.hands[player_idx]
        }
        return [t for t in create_deck() if _canonical_tile(t) not in known]

    def is_round_over(self) -> bool:
        """
        Detects when the round ends:
        1. Any player has 0 tiles remaining (Domino Out).
        2. All 4 players pass consecutively (Blocked Game / Tranca).
        """
        if self.domino_out_player is not None or any(len(h) == 0 for h in self.hands):
            return True
        if self.consecutive_passes >= 4:
            return True
        return False

    def step(self, move: Optional[Move] = None) -> Dict[str, Any]:
        """
        Executes one turn for `self.current_turn`:
        - If `move` is None, verifies the player truly must pass, records the open
          board pips they lacked, increments `consecutive_passes`, and advances turn.
        - If `move` is `(tile, side)`, validates and places the tile on the board,
          removes it from the player's hand, resets `consecutive_passes`, checks
          for Domino Out, and advances turn.
        """
        if self.is_round_over():
            raise RuntimeError("Cannot step: the round has already ended.")

        player = self.current_turn
        hand = self.hands[player]

        if move is None:
            valid_pass, msg = handle_pass(hand, self.board)
            if not valid_pass:
                raise ValueError(f"Player {player} attempted illegal pass: {msg}")

            open_ends = get_open_ends(self.board)
            if open_ends is not None:
                self.passed_pips[player].add(open_ends[0])
                self.passed_pips[player].add(open_ends[1])

            self.consecutive_passes += 1
            event = {
                "player": player,
                "team": self.get_player_team(player),
                "action": "PASS",
                "open_ends": open_ends,
            }
            self.move_history.append(event)
            self.current_turn = (self.current_turn + 1) % 4
            return event

        tile, side = move
        updated_hand, updated_board = apply_move(hand, self.board, tile, side)
        self.hands[player] = updated_hand
        self.board = updated_board
        self.consecutive_passes = 0

        if len(updated_hand) == 0:
            self.domino_out_player = player

        event = {
            "player": player,
            "team": self.get_player_team(player),
            "action": "PLAY",
            "tile": tile,
            "side": side.upper(),
            "open_ends": get_open_ends(self.board),
        }
        self.move_history.append(event)
        self.current_turn = (self.current_turn + 1) % 4
        return event

    def get_observation(self, player_idx: int) -> Dict[str, Any]:
        """
        Returns the partially observable state visible to `player_idx`.
        Prevents cheating by hiding opponents' and partner's exact hands.
        """
        return {
            "player_idx": player_idx,
            "team": self.get_player_team(player_idx),
            "hand": list(self.hands[player_idx]),
            "board": list(self.board),
            "open_ends": get_open_ends(self.board),
            "hand_sizes": self.get_hand_sizes(),
            "passed_pips": {p: set(pips) for p, pips in self.passed_pips.items()},
            "legal_moves": get_legal_moves(self.hands[player_idx], self.board),
            "unseen_tiles": self.get_unseen_tiles_for_player(player_idx),
        }

    def clone(self) -> "DominoGameState":
        """Creates an independent copy of the current game state for AI rollouts."""
        cloned = DominoGameState.__new__(DominoGameState)
        cloned.player_names = list(self.player_names)
        cloned.teams = {1: (0, 2), 2: (1, 3)}
        cloned.hands = [list(h) for h in self.hands]
        cloned.board = list(self.board)
        cloned.current_turn = self.current_turn
        cloned.consecutive_passes = self.consecutive_passes
        cloned.domino_out_player = self.domino_out_player
        cloned.move_history = list(self.move_history)
        cloned.passed_pips = {p: set(pips) for p, pips in self.passed_pips.items()}
        return cloned