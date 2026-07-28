import unittest
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from tempfile import TemporaryDirectory

import torch

from plump.cards import Card, Rank, Suit
from plump.gui.app import CheckpointModel, GuiController, sort_gui_hand
from plump.seq.config import SEQ_SCHEMA_VERSION, SeqModelConfig
from plump.seq.model import SeqPlumpModel
from plump.seq.tokens import card_id
from plump.state import Phase


class GuiControllerTest(unittest.TestCase):
    @staticmethod
    def _play_until_round_stops(controller):
        while controller.game.env.phase() in (Phase.BIDDING, Phase.PLAYING):
            env = controller.game.env
            if env.current_player() != 0:
                controller.advance_bot()
                continue
            action = env.legal_actions()[0]
            if env.phase() == Phase.BIDDING:
                controller.bid(action.bid)
            else:
                controller.play(action.card.suit.value, int(action.card.rank))

    def test_new_game_honors_human_bid_position(self):
        controller = GuiController()

        view = controller.new_game(opponents=3, hand_size=3, human_bid_position=3, seed=1)

        round_state = controller.game.env.state.current_round
        self.assertEqual(round_state.bidding_order, [2, 3, 0, 1])
        self.assertEqual(view["current_player"], 0)
        self.assertEqual(view["phase"], Phase.BIDDING.value)
        self.assertEqual(len(view["bids"]), 2)
        self.assertTrue(view["legal_bids"])

    def test_human_bid_advances_random_models_to_next_human_turn_or_done(self):
        controller = GuiController()
        controller.new_game(opponents=2, hand_size=3, human_bid_position=1, seed=2)

        view = controller.bid(0)

        self.assertEqual(view["phase"], Phase.PLAYING.value)
        self.assertEqual(len(view["players"]), 3)

    def test_invalid_setup_rejected(self):
        controller = GuiController()

        with self.assertRaises(ValueError):
            controller.new_game(opponents=6, hand_size=9, human_bid_position=1, seed=1)

        with self.assertRaises(ValueError):
            controller.new_game(opponents=5, hand_size=2, human_bid_position=1, seed=1)

        with self.assertRaises(ValueError):
            controller.new_game(opponents=2, hand_size=2, human_bid_position=1, seed=1)

    def test_full_game_schedule_and_rotating_bid_order(self):
        controller = GuiController()

        view = controller.new_game(
            opponents=2,
            hand_size=5,
            human_bid_position=3,
            seed=4,
            mode="game",
            min_hand_size=3,
            max_hand_size=5,
        )

        self.assertEqual(view["schedule"], [5, 4, 3, 4, 5])
        self.assertEqual(controller.game.env.config.bidding_start_players, [1, 2, 0, 1, 2])
        self.assertEqual(view["round_number"], 1)
        self.assertEqual(view["total_rounds"], 5)
        self.assertFalse(view["game_over"])

    def test_full_game_validates_selector_bounds(self):
        controller = GuiController()

        for minimum, maximum in ((2, 5), (5, 5), (3, 4), (3, 11)):
            with self.subTest(minimum=minimum, maximum=maximum), self.assertRaises(ValueError):
                controller.new_game(
                    opponents=3,
                    hand_size=5,
                    human_bid_position=1,
                    mode="game",
                    min_hand_size=minimum,
                    max_hand_size=maximum,
                )

    def test_full_game_pauses_and_carries_scores_to_next_round(self):
        controller = GuiController()
        controller.new_game(
            opponents=2,
            hand_size=5,
            human_bid_position=1,
            seed=8,
            mode="game",
            min_hand_size=3,
            max_hand_size=5,
        )

        self._play_until_round_stops(controller)
        paused = controller.view()

        self.assertEqual(paused["phase"], Phase.ROUND_OVER.value)
        self.assertTrue(paused["round_over"])
        self.assertFalse(paused["game_over"])
        self.assertEqual(len(paused["completed_rounds"]), 1)
        self.assertEqual(
            {player["id"]: player["score"] for player in paused["players"]},
            paused["round_scores"],
        )

        resumed = controller.next_round()

        self.assertEqual(resumed["round_number"], 2)
        self.assertEqual(resumed["hand_size"], 4)
        self.assertEqual(controller.game.env.state.current_round.bidding_start_player, 1)
        self.assertEqual(len(resumed["completed_rounds"]), 1)

    def test_full_game_finishes_with_complete_standings(self):
        controller = GuiController()
        controller.new_game(
            opponents=2,
            hand_size=5,
            human_bid_position=1,
            seed=12,
            mode="game",
            min_hand_size=3,
            max_hand_size=5,
        )

        while not controller.game.env.is_done():
            self._play_until_round_stops(controller)
            if controller.game.env.phase() == Phase.ROUND_OVER:
                controller.next_round()
        view = controller.view()

        self.assertTrue(view["game_over"])
        self.assertEqual(view["phase"], Phase.GAME_OVER.value)
        self.assertEqual(len(view["completed_rounds"]), len(view["schedule"]))
        high_score = max(player["score"] for player in view["players"])
        self.assertEqual(
            set(view["winner_ids"]),
            {player["id"] for player in view["players"] if player["score"] == high_score},
        )

    def test_batched_model_view_exposes_every_legal_bid_probability(self):
        controller = GuiController()
        controller.new_game(opponents=2, hand_size=3, human_bid_position=1, seed=2)
        env = controller.game.env

        class FakePolicy:
            def __init__(self):
                self.calls = 0

            def predict_many(self, envs, *, observers, aux_heads):
                self.calls += 1
                self.envs = envs
                self.observers = observers
                self.aux_heads = aux_heads
                batch = len(envs)
                bid_logits = torch.zeros(batch, 11)
                bid_logits[0] = torch.arange(11)
                return SimpleNamespace(
                    bid_logits=bid_logits,
                    card_logits=torch.zeros(batch, 52),
                    value=torch.zeros(batch),
                    trick_logits=None,
                    suit_logits=torch.zeros(batch, 5, 4),
                    bid_hit_logits=torch.zeros(batch, 5),
                )

        checkpoint = object.__new__(CheckpointModel)
        checkpoint.policy = FakePolicy()

        predictions, action_policy = checkpoint.view_predictions(env)

        self.assertEqual(checkpoint.policy.calls, 1)
        self.assertEqual(len(checkpoint.policy.envs), 3)
        self.assertEqual(checkpoint.policy.observers, [0, 1, 2])
        self.assertEqual(
            checkpoint.policy.aux_heads,
            ("suit", "bid_hit"),
        )
        self.assertEqual(set(predictions), {0, 1, 2})
        self.assertEqual(action_policy["phase"], Phase.BIDDING.value)
        self.assertEqual(
            {row["bid"] for row in action_policy["actions"]},
            set(env.get_observation(0).legal_bids),
        )
        self.assertAlmostEqual(
            sum(row["probability"] for row in action_policy["actions"]),
            1.0,
            places=6,
        )
        best = [row for row in action_policy["actions"] if row["is_best"]]
        self.assertEqual([row["bid"] for row in best], [max(env.get_observation(0).legal_bids)])

        _, hidden_action_policy = checkpoint.view_predictions(
            env,
            include_action_probabilities=False,
        )
        self.assertIsNone(hidden_action_policy)

    def test_play_probabilities_contain_only_legal_cards(self):
        controller = GuiController()
        controller.new_game(opponents=2, hand_size=3, human_bid_position=1, seed=3)
        controller.bid(0)
        while controller.game.env.current_player() != 0:
            controller.advance_bot()
        observation = controller.game.env.get_observation(0)
        card_logits = torch.arange(52, dtype=torch.float32).repeat(3, 1)
        output = SimpleNamespace(card_logits=card_logits)

        action_policy = CheckpointModel._action_probabilities(observation, output)

        legal_keys = {
            f"{card.suit.value}:{int(card.rank)}"
            for card in observation.legal_cards
        }
        self.assertEqual(action_policy["phase"], Phase.PLAYING.value)
        self.assertEqual(
            {row["card_key"] for row in action_policy["actions"]},
            legal_keys,
        )
        self.assertAlmostEqual(
            sum(row["probability"] for row in action_policy["actions"]),
            1.0,
            places=6,
        )
        expected_best = max(observation.legal_cards, key=card_id)
        self.assertEqual(
            [row["card_key"] for row in action_policy["actions"] if row["is_best"]],
            [f"{expected_best.suit.value}:{int(expected_best.rank)}"],
        )

    def test_real_schema_v6_checkpoint_produces_gui_predictions(self):
        config = SeqModelConfig(
            d_model=16,
            n_layers=1,
            n_heads=2,
            n_kv_heads=1,
            d_ff=32,
        )
        model = SeqPlumpModel(config)
        controller = GuiController()
        controller.new_game(
            opponents=2,
            hand_size=3,
            human_bid_position=1,
            seed=4,
        )
        with TemporaryDirectory() as directory:
            path = Path(directory) / "seq.pt"
            torch.save(
                {
                    "schema_version": SEQ_SCHEMA_VERSION,
                    "model_config": asdict(config),
                    "model_state_dict": model.state_dict(),
                },
                path,
            )
            checkpoint = CheckpointModel(path, device="cpu")
            predictions, action_policy = checkpoint.view_predictions(
                controller.game.env
            )

        self.assertEqual(set(predictions), {0, 1, 2})
        self.assertEqual(
            {row["player"] for row in predictions[0]["rows"]},
            {0, 1, 2},
        )
        self.assertEqual(action_policy["phase"], Phase.BIDDING.value)
        self.assertAlmostEqual(
            sum(row["probability"] for row in action_policy["actions"]),
            1.0,
            places=6,
        )

    def test_no_checkpoint_omits_model_action_probabilities(self):
        controller = GuiController()

        view = controller.new_game(opponents=2, hand_size=3, human_bid_position=1, seed=1)

        self.assertIsNone(view["model_action_probabilities"])

    def test_setup_probability_visibility_is_preserved_in_game_state(self):
        controller = GuiController()

        view = controller.new_game(
            opponents=2,
            hand_size=3,
            human_bid_position=1,
            seed=1,
            show_probabilities=False,
        )

        self.assertFalse(view["show_probabilities"])
        self.assertFalse(controller.game.show_probabilities)

        enabled = controller.set_probability_visibility(True)
        self.assertTrue(enabled["show_probabilities"])
        self.assertTrue(controller.game.show_probabilities)

        disabled = controller.set_probability_visibility(False)
        self.assertFalse(disabled["show_probabilities"])
        self.assertFalse(controller.game.show_probabilities)

    def test_gui_hand_order_is_hearts_spades_diamonds_clubs(self):
        cards = [
            Card(Suit.CLUBS, Rank.TWO),
            Card(Suit.SPADES, Rank.ACE),
            Card(Suit.HEARTS, Rank.KING),
            Card(Suit.DIAMONDS, Rank.THREE),
            Card(Suit.HEARTS, Rank.FOUR),
            Card(Suit.SPADES, Rank.FIVE),
        ]

        ordered = sort_gui_hand(cards)

        self.assertEqual(
            [(card.suit, card.rank) for card in ordered],
            [
                (Suit.HEARTS, Rank.FOUR),
                (Suit.HEARTS, Rank.KING),
                (Suit.SPADES, Rank.FIVE),
                (Suit.SPADES, Rank.ACE),
                (Suit.DIAMONDS, Rank.THREE),
                (Suit.CLUBS, Rank.TWO),
            ],
        )


if __name__ == "__main__":
    unittest.main()
