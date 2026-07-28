"""Deterministic Plump game environment."""

from __future__ import annotations

import random
from typing import Optional

from .cards import Card, Suit, make_deck, sort_cards
from .rules import (
    bidding_order,
    compute_voids,
    determine_first_leader_from_bids,
    determine_trick_winner,
    legal_bids,
    legal_cards,
    score_round,
)
from .state import (
    Action,
    Bid,
    BidAction,
    EventType,
    GameConfig,
    GameEvent,
    GameState,
    IllegalActionError,
    Observation,
    Phase,
    PlayCardAction,
    RoundState,
    StepResult,
    Trick,
    TrickPlay,
    TrumpPolicy,
)


class PlumpEnv:
    """Core Plump engine with explicit legal actions and deterministic stepping."""

    # Build the per-step Observation, i.e. StepResult.observation. Callers that
    # read the state directly can turn this off: the observation is rebuilt from
    # scratch on every step (voids over every player and suit, per-player played
    # cards, legal actions, a copy of the event log) and it is the single most
    # expensive part of stepping. get_observation() stays callable either way.
    emit_observations: bool = True

    def __init__(self, config: Optional[GameConfig] = None, seed: Optional[int] = None):
        self.config = config or GameConfig()
        self._validate_config()
        self.rng = random.Random(seed)
        self.state = GameState(
            config=self.config,
            cumulative_scores={player: 0 for player in range(self.config.num_players)},
        )

    def reset(
        self,
        seed: Optional[int] = None,
        *,
        deck_order: Optional[list[Card]] = None,
        manual_hands: Optional[dict[int, list[Card]]] = None,
        manual_trump_suit: Optional[Suit] = None,
    ) -> GameState:
        """Reset and start the first round.

        Optional deck, hands, and trump overrides are useful for deterministic
        tests and scripted scenarios.
        """

        if seed is not None:
            self.rng.seed(seed)
        self.state = GameState(
            config=self.config,
            cumulative_scores={player: 0 for player in range(self.config.num_players)},
        )
        self._deck_override = deck_order
        self._hands_override = manual_hands
        self._trump_override = manual_trump_suit
        self._start_round(0)
        return self.state

    def clone(self) -> "PlumpEnv":
        """Copy mutable game state without recursively copying configuration."""

        clone = object.__new__(PlumpEnv)
        clone.config = self.config
        # __new__ rather than Random(): the constructor draws 32 bytes from the
        # OS and runs the key schedule (~15us) only for setstate to throw it all
        # away (~1.5us). Branching rollouts clone thousands of envs per batch.
        clone.rng = random.Random.__new__(random.Random)
        clone.rng.setstate(self.rng.getstate())
        clone.state = GameState(
            config=self.config,
            phase=self.state.phase,
            round_index=self.state.round_index,
            current_player=self.state.current_player,
            cumulative_scores=dict(self.state.cumulative_scores),
            rounds=[
                RoundState(
                    round_index=round_state.round_index,
                    hand_size=round_state.hand_size,
                    trump_suit=round_state.trump_suit,
                    bidding_start_player=(
                        round_state.bidding_start_player
                    ),
                    bidding_order=round_state.bidding_order,
                    play_start_player=round_state.play_start_player,
                    initial_hands=round_state.initial_hands,
                    current_hands={
                        player: list(hand)
                        for player, hand in (
                            round_state.current_hands.items()
                        )
                    },
                    bids=list(round_state.bids),
                    tricks=[
                        Trick(
                            trick_index=trick.trick_index,
                            leader=trick.leader,
                            led_suit=trick.led_suit,
                            plays=list(trick.plays),
                            winner=trick.winner,
                        )
                        for trick in round_state.tricks
                    ],
                    tricks_won=dict(round_state.tricks_won),
                    round_scores=dict(round_state.round_scores),
                    cumulative_scores_after_round=dict(
                        round_state.cumulative_scores_after_round
                    ),
                )
                for round_state in self.state.rounds
            ],
            event_log=list(self.state.event_log),
        )
        for name in (
            "_deck_override",
            "_hands_override",
            "_trump_override",
            "emit_observations",
        ):
            if hasattr(self, name):
                setattr(clone, name, getattr(self, name))
        return clone

    def current_player(self) -> int:
        if self.state.current_player is None:
            raise RuntimeError("There is no current player.")
        return self.state.current_player

    def phase(self) -> Phase:
        return self.state.phase

    def is_done(self) -> bool:
        return self.state.phase == Phase.GAME_OVER

    def legal_actions(self) -> list[Action]:
        if self.state.phase == Phase.BIDDING:
            player = self.current_player()
            return [BidAction(player, bid) for bid in self._legal_bids_for_current_player()]
        if self.state.phase == Phase.PLAYING:
            player = self.current_player()
            return [PlayCardAction(player, card) for card in self._legal_cards_for_current_player()]
        return []

    def step(self, action: Action) -> StepResult:
        """Validate and apply one bid or card-play action."""

        if self.state.phase == Phase.GAME_OVER:
            raise IllegalActionError("Cannot act after the game is over.")
        if isinstance(action, BidAction):
            return self._step_bid(action)
        if isinstance(action, PlayCardAction):
            return self._step_play(action)
        raise IllegalActionError(f"Unsupported action: {action!r}")

    def get_observation(self, player_id: int) -> Observation:
        """Return information visible to ``player_id`` only."""

        self._validate_player(player_id)
        if not self.state.rounds:
            return Observation(
                player_id=player_id,
                phase=self.state.phase,
                round_index=self.state.round_index,
                total_rounds=len(self.config.hand_sizes),
                rounds_remaining=len(self.config.hand_sizes),
                hand_size=0,
                trump_suit=None,
                current_player=self.state.current_player,
                bidding_start_player=0,
                bidding_order=[],
                play_start_player=None,
                my_hand=[],
                bids=[],
                tricks_won={player: 0 for player in range(self.config.num_players)},
                current_trick=None,
                completed_tricks=[],
                played_cards_by_player={player: [] for player in range(self.config.num_players)},
                played_cards_total=[],
                voids={player: {suit: False for suit in Suit} for player in range(self.config.num_players)},
                legal_bids=[],
                legal_cards=[],
                scores=dict(self.state.cumulative_scores),
                event_log=list(self.state.event_log),
                hand_size_schedule=list(self.config.hand_sizes),
            )

        round_state = self.state.current_round
        current_trick = self._current_trick()
        completed_tricks = [trick for trick in round_state.tricks if trick.winner is not None]
        played_cards_by_player = {player: [] for player in range(self.config.num_players)}
        played_cards_total: list[Card] = []
        for trick in round_state.tricks:
            for play in trick.plays:
                played_cards_by_player[play.player].append(play.card)
                played_cards_total.append(play.card)

        legal_bid_values: list[int] = []
        legal_card_values: list[Card] = []
        if self.state.current_player == player_id:
            if self.state.phase == Phase.BIDDING:
                legal_bid_values = self._legal_bids_for_current_player()
            elif self.state.phase == Phase.PLAYING:
                legal_card_values = self._legal_cards_for_current_player()

        return Observation(
            player_id=player_id,
            phase=self.state.phase,
            round_index=round_state.round_index,
            total_rounds=len(self.config.hand_sizes),
            rounds_remaining=max(len(self.config.hand_sizes) - round_state.round_index - 1, 0),
            hand_size=round_state.hand_size,
            trump_suit=round_state.trump_suit,
            current_player=self.state.current_player,
            bidding_start_player=round_state.bidding_start_player,
            bidding_order=list(round_state.bidding_order),
            play_start_player=round_state.play_start_player,
            my_hand=sort_cards(round_state.current_hands.get(player_id, [])),
            bids=list(round_state.bids),
            tricks_won=dict(round_state.tricks_won),
            current_trick=current_trick,
            completed_tricks=list(completed_tricks),
            played_cards_by_player={player: list(cards) for player, cards in played_cards_by_player.items()},
            played_cards_total=list(played_cards_total),
            voids=compute_voids(round_state.tricks, self.config.num_players),
            legal_bids=legal_bid_values,
            legal_cards=legal_card_values,
            scores=dict(self.state.cumulative_scores),
            event_log=list(self.state.event_log),
            hand_size_schedule=list(self.config.hand_sizes),
        )

    def _step_bid(self, action: BidAction) -> StepResult:
        round_state = self.state.current_round
        if self.state.phase != Phase.BIDDING:
            raise IllegalActionError("Bids are only legal during the bidding phase.")
        if action.player != self.state.current_player:
            raise IllegalActionError(f"Player {action.player} cannot bid; player {self.state.current_player} is to act.")
        if action.bid not in self._legal_bids_for_current_player():
            raise IllegalActionError(f"Illegal bid {action.bid} for player {action.player}.")

        position = len(round_state.bids)
        round_state.bids.append(Bid(action.player, action.bid, position))
        self.state.event_log.append(
            GameEvent(
                type=EventType.BID,
                round_index=round_state.round_index,
                player=action.player,
                bid=action.bid,
            )
        )

        if len(round_state.bids) == self.config.num_players:
            leader = determine_first_leader_from_bids(round_state.bids, round_state.bidding_order)
            round_state.play_start_player = leader
            round_state.tricks.append(Trick(trick_index=0, leader=leader))
            self.state.phase = Phase.PLAYING
            self.state.current_player = leader
        else:
            self.state.current_player = round_state.bidding_order[position + 1]

        return self._result({player: 0 for player in range(self.config.num_players)})

    def _step_play(self, action: PlayCardAction) -> StepResult:
        round_state = self.state.current_round
        if self.state.phase != Phase.PLAYING:
            raise IllegalActionError("Cards are only legal during the playing phase.")
        if action.player != self.state.current_player:
            raise IllegalActionError(f"Player {action.player} cannot play; player {self.state.current_player} is to act.")

        hand = round_state.current_hands[action.player]
        if action.card not in hand:
            raise IllegalActionError(f"Player {action.player} does not hold {action.card}.")

        current_trick = self._current_trick()
        if current_trick is None:
            raise RuntimeError("Playing phase has no current trick.")
        legal = legal_cards(hand, current_trick)
        if action.card not in legal:
            raise IllegalActionError(f"Illegal play {action.card}; player must follow suit when able.")

        hand.remove(action.card)
        if not current_trick.plays:
            current_trick.led_suit = action.card.suit
        position = len(current_trick.plays)
        current_trick.plays.append(TrickPlay(action.player, action.card, position))
        self.state.event_log.append(
            GameEvent(
                type=EventType.PLAY,
                round_index=round_state.round_index,
                player=action.player,
                card=action.card,
                trick_index=current_trick.trick_index,
                position_in_trick=position,
            )
        )

        rewards = {player: 0 for player in range(self.config.num_players)}
        info: dict[str, object] = {}
        if len(current_trick.plays) == self.config.num_players:
            winner = determine_trick_winner(current_trick, round_state.trump_suit)
            current_trick.winner = winner
            round_state.tricks_won[winner] += 1
            self.state.event_log.append(
                GameEvent(
                    type=EventType.TRICK_WIN,
                    round_index=round_state.round_index,
                    player=winner,
                    trick_index=current_trick.trick_index,
                )
            )

            if len([trick for trick in round_state.tricks if trick.winner is not None]) == round_state.hand_size:
                rewards = self._finish_round()
                info["round_ended"] = True
                info["round_index"] = round_state.round_index
            else:
                next_index = current_trick.trick_index + 1
                round_state.tricks.append(Trick(trick_index=next_index, leader=winner))
                self.state.current_player = winner
        else:
            self.state.current_player = (current_trick.leader + len(current_trick.plays)) % self.config.num_players

        return self._result(rewards, info)

    def _finish_round(self) -> dict[int, int]:
        round_state = self.state.current_round
        bid_values = {bid.player: bid.value for bid in round_state.bids}
        rewards = score_round(bid_values, round_state.tricks_won, self.config.scoring)
        round_state.round_scores = dict(rewards)
        for player, points in rewards.items():
            self.state.cumulative_scores[player] += points
        round_state.cumulative_scores_after_round = dict(self.state.cumulative_scores)
        self.state.event_log.append(GameEvent(type=EventType.ROUND_END, round_index=round_state.round_index))

        next_round_index = round_state.round_index + 1
        if next_round_index >= len(self.config.hand_sizes):
            self.state.phase = Phase.GAME_OVER
            self.state.current_player = None
        elif self.config.auto_advance_rounds:
            self._start_round(next_round_index)
        else:
            self.state.phase = Phase.ROUND_OVER
            self.state.current_player = None
        return rewards

    def start_next_round(self) -> GameState:
        """Start the next round when ``auto_advance_rounds`` is disabled."""

        if self.state.phase != Phase.ROUND_OVER:
            raise RuntimeError("Next round can only be started from ROUND_OVER.")
        next_round_index = self.state.current_round.round_index + 1
        if next_round_index >= len(self.config.hand_sizes):
            self.state.phase = Phase.GAME_OVER
            return self.state
        self._start_round(next_round_index)
        return self.state

    def _start_round(self, round_index: int) -> None:
        hand_size = self.config.hand_sizes[round_index]
        if self.config.bidding_start_players and round_index < len(self.config.bidding_start_players):
            start_player = self.config.bidding_start_players[round_index]
        else:
            start_player = round_index % self.config.num_players
        order = bidding_order(start_player, self.config.num_players)
        initial_hands, trump_suit = self._deal_round(hand_size)
        round_state = RoundState(
            round_index=round_index,
            hand_size=hand_size,
            trump_suit=trump_suit,
            bidding_start_player=start_player,
            bidding_order=order,
            initial_hands={player: sort_cards(cards) for player, cards in initial_hands.items()},
            current_hands={player: sort_cards(cards) for player, cards in initial_hands.items()},
            tricks_won={player: 0 for player in range(self.config.num_players)},
        )
        self.state.round_index = round_index
        self.state.rounds.append(round_state)
        self.state.phase = Phase.BIDDING
        self.state.current_player = order[0]
        self.state.event_log.append(GameEvent(type=EventType.ROUND_START, round_index=round_index))

    def _deal_round(self, hand_size: int) -> tuple[dict[int, list[Card]], Optional[Suit]]:
        manual_hands = self._hands_override if hasattr(self, "_hands_override") else None
        deck_override = self._deck_override if hasattr(self, "_deck_override") else None
        trump_override = self._trump_override if hasattr(self, "_trump_override") else None
        manual_hands = manual_hands if manual_hands is not None else self.config.manual_hands
        deck_override = deck_override if deck_override is not None else self.config.deck_order
        trump_override = trump_override if trump_override is not None else self.config.manual_trump_suit

        deck = list(deck_override) if deck_override is not None else make_deck()
        self._validate_unique_cards(deck, "deck")
        if self.config.shuffle and deck_override is None:
            self.rng.shuffle(deck)

        if manual_hands is not None:
            hands = {player: list(manual_hands.get(player, [])) for player in range(self.config.num_players)}
            for player, hand in hands.items():
                if len(hand) != hand_size:
                    raise ValueError(f"Manual hand for player {player} must contain {hand_size} cards.")
            all_hand_cards = [card for hand in hands.values() for card in hand]
            self._validate_unique_cards(all_hand_cards, "manual hands")
            remaining = [card for card in deck if card not in set(all_hand_cards)]
        else:
            needed = hand_size * self.config.num_players
            if len(deck) < needed:
                raise ValueError("Deck does not contain enough cards for this round.")
            hands = {player: [] for player in range(self.config.num_players)}
            for _ in range(hand_size):
                for player in range(self.config.num_players):
                    hands[player].append(deck.pop(0))
            remaining = deck

        trump_suit = self._determine_trump(remaining, trump_override)
        return hands, trump_suit

    def _determine_trump(self, remaining_cards: list[Card], trump_override: Optional[Suit]) -> Optional[Suit]:
        if trump_override is not None:
            return trump_override
        if self.config.trump_policy == TrumpPolicy.NONE:
            return None
        if self.config.trump_policy == TrumpPolicy.REVEAL_NEXT_CARD:
            return remaining_cards[0].suit if remaining_cards else None
        raise ValueError(f"Unsupported trump policy {self.config.trump_policy}.")

    def _legal_bids_for_current_player(self) -> list[int]:
        round_state = self.state.current_round
        return legal_bids(
            round_state.hand_size,
            round_state.bids,
            self.config.num_players,
            self.config.forbid_total_bid_equals_hand_size,
        )

    def _legal_cards_for_current_player(self) -> list[Card]:
        round_state = self.state.current_round
        player = self.current_player()
        return legal_cards(round_state.current_hands[player], self._current_trick())

    def _current_trick(self) -> Optional[Trick]:
        if not self.state.rounds:
            return None
        round_state = self.state.current_round
        if not round_state.tricks:
            return None
        trick = round_state.tricks[-1]
        return trick if trick.winner is None else None

    def _result(self, rewards: dict[int, int], info: Optional[dict[str, object]] = None) -> StepResult:
        observation = (
            None
            if not self.emit_observations
            or self.is_done()
            or self.state.current_player is None
            else self.get_observation(self.current_player())
        )
        return StepResult(
            state=self.state,
            observation=observation,
            rewards=rewards,
            done=self.is_done(),
            info=info or {},
        )

    def _validate_config(self) -> None:
        if not 2 <= self.config.num_players <= 7:
            raise ValueError("PlumpEnv supports 2 to 7 players.")
        if not self.config.hand_sizes:
            raise ValueError("GameConfig.hand_sizes must not be empty.")
        max_hand_size = 52 // self.config.num_players
        for hand_size in self.config.hand_sizes:
            if hand_size < 1:
                raise ValueError("Hand sizes must be positive.")
            if hand_size > max_hand_size:
                raise ValueError(f"Hand size {hand_size} is too large for {self.config.num_players} players.")
        if self.config.bidding_start_players is not None:
            if len(self.config.bidding_start_players) > len(self.config.hand_sizes):
                raise ValueError("bidding_start_players cannot be longer than hand_sizes.")
            for player in self.config.bidding_start_players:
                self._validate_player(player)

    def _validate_player(self, player_id: int) -> None:
        if player_id < 0 or player_id >= self.config.num_players:
            raise ValueError(f"Invalid player id {player_id}.")

    @staticmethod
    def _validate_unique_cards(cards: list[Card], label: str) -> None:
        if len(cards) != len(set(cards)):
            raise ValueError(f"{label} contains duplicate cards.")
