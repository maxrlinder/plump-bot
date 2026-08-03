import random
import unittest

from plump.cards import Card, Rank, Suit
from plump.env import PlumpEnv
from plump.state import BidAction, EventType, GameConfig, IllegalActionError, Phase, PlayCardAction, TrumpPolicy


class EnvTest(unittest.TestCase):
    def test_unchecked_step_matches_validated_step_for_legal_trajectory(self):
        env = PlumpEnv(
            GameConfig(num_players=4, hand_sizes=[5], trump_policy=TrumpPolicy.NONE),
            seed=9182,
        )
        env.reset(seed=9182)
        chooser = random.Random(71)

        while not env.is_done():
            action = chooser.choice(env.legal_actions())
            unchecked = env.clone()
            validated_result = env.step(action)
            unchecked_result = unchecked.step_unchecked(action)
            self.assertEqual(unchecked.state, env.state)
            self.assertEqual(unchecked_result, validated_result)

    def test_clone_is_independent_and_preserves_decision_state(self):
        env = PlumpEnv(GameConfig(num_players=3, hand_sizes=[3]), seed=41)
        env.reset()
        clone = env.clone()

        self.assertEqual(clone.phase(), env.phase())
        self.assertEqual(clone.current_player(), env.current_player())
        self.assertEqual(clone.legal_actions(), env.legal_actions())
        clone.step(clone.legal_actions()[0])

        self.assertEqual(len(env.state.current_round.bids), 0)
        self.assertEqual(len(clone.state.current_round.bids), 1)

    def test_illegal_play_rejected_when_player_can_follow_suit(self):
        hands = {
            0: [Card(Suit.SPADES, Rank.ACE), Card(Suit.CLUBS, Rank.TWO)],
            1: [Card(Suit.SPADES, Rank.TWO), Card(Suit.HEARTS, Rank.ACE)],
        }
        env = PlumpEnv(
            GameConfig(
                num_players=2,
                hand_sizes=[2],
                manual_hands=hands,
                manual_trump_suit=None,
                trump_policy=TrumpPolicy.NONE,
                forbid_total_bid_equals_hand_size=False,
            )
        )
        env.reset()
        env.step(BidAction(0, 1))
        env.step(BidAction(1, 0))
        env.step(PlayCardAction(0, Card(Suit.SPADES, Rank.ACE)))

        with self.assertRaises(IllegalActionError):
            env.step(PlayCardAction(1, Card(Suit.HEARTS, Rank.ACE)))

    def test_full_mini_round_scores_and_ends_game(self):
        hands = {
            0: [Card(Suit.SPADES, Rank.ACE)],
            1: [Card(Suit.SPADES, Rank.KING)],
            2: [Card(Suit.HEARTS, Rank.TWO)],
            3: [Card(Suit.SPADES, Rank.THREE)],
        }
        env = PlumpEnv(
            GameConfig(
                num_players=4,
                hand_sizes=[1],
                manual_hands=hands,
                trump_policy=TrumpPolicy.NONE,
                forbid_total_bid_equals_hand_size=False,
            )
        )
        state = env.reset()

        self.assertEqual(state.phase, Phase.BIDDING)
        self.assertEqual(env.current_player(), 0)

        env.step(BidAction(0, 1))
        env.step(BidAction(1, 0))
        env.step(BidAction(2, 0))
        env.step(BidAction(3, 0))

        self.assertEqual(env.phase(), Phase.PLAYING)
        self.assertEqual(env.current_player(), 0)

        env.step(PlayCardAction(0, Card(Suit.SPADES, Rank.ACE)))
        env.step(PlayCardAction(1, Card(Suit.SPADES, Rank.KING)))
        env.step(PlayCardAction(2, Card(Suit.HEARTS, Rank.TWO)))
        result = env.step(PlayCardAction(3, Card(Suit.SPADES, Rank.THREE)))

        first_round = result.state.rounds[0]
        self.assertTrue(result.done)
        self.assertEqual(result.state.phase, Phase.GAME_OVER)
        self.assertEqual(first_round.tricks[0].winner, 0)
        self.assertEqual(first_round.tricks_won, {0: 1, 1: 0, 2: 0, 3: 0})
        self.assertEqual(first_round.round_scores, {0: 11, 1: 5, 2: 5, 3: 5})
        self.assertEqual(result.rewards, {0: 11, 1: 5, 2: 5, 3: 5})
        self.assertIn(EventType.ROUND_END, [event.type for event in result.state.event_log])

    def test_observation_hides_opponent_hands_and_includes_legal_actions(self):
        hands = {
            0: [Card(Suit.SPADES, Rank.ACE)],
            1: [Card(Suit.HEARTS, Rank.KING)],
        }
        env = PlumpEnv(
            GameConfig(
                num_players=2,
                hand_sizes=[1],
                manual_hands=hands,
                trump_policy=TrumpPolicy.NONE,
                forbid_total_bid_equals_hand_size=False,
            )
        )
        env.reset()

        obs0 = env.get_observation(0)

        self.assertEqual(obs0.my_hand, [Card(Suit.SPADES, Rank.ACE)])
        self.assertEqual(obs0.legal_bids, [0, 1])
        self.assertEqual(obs0.played_cards_total, [])
        self.assertFalse(any(Card(Suit.HEARTS, Rank.KING) in cards for cards in obs0.played_cards_by_player.values()))

    def test_config_can_override_bidding_start_player(self):
        env = PlumpEnv(GameConfig(num_players=4, hand_sizes=[1], bidding_start_players=[2]))
        state = env.reset()

        self.assertEqual(state.current_round.bidding_start_player, 2)
        self.assertEqual(state.current_round.bidding_order, [2, 3, 0, 1])
        self.assertEqual(env.current_player(), 2)

    def test_default_game_has_no_trump(self):
        env = PlumpEnv(GameConfig(num_players=4, hand_sizes=[1]), seed=1)
        state = env.reset()

        self.assertIsNone(state.current_round.trump_suit)

    def test_default_no_trump_off_suit_card_cannot_win(self):
        hands = {
            0: [Card(Suit.CLUBS, Rank.SEVEN)],
            1: [Card(Suit.CLUBS, Rank.FOUR)],
            2: [Card(Suit.CLUBS, Rank.EIGHT)],
            3: [Card(Suit.DIAMONDS, Rank.FIVE)],
        }
        env = PlumpEnv(
            GameConfig(
                num_players=4,
                hand_sizes=[1],
                manual_hands=hands,
                bidding_start_players=[0],
                forbid_total_bid_equals_hand_size=False,
            )
        )
        env.reset()
        env.step(BidAction(0, 1))
        env.step(BidAction(1, 0))
        env.step(BidAction(2, 0))
        env.step(BidAction(3, 0))
        env.step(PlayCardAction(0, Card(Suit.CLUBS, Rank.SEVEN)))
        env.step(PlayCardAction(1, Card(Suit.CLUBS, Rank.FOUR)))
        env.step(PlayCardAction(2, Card(Suit.CLUBS, Rank.EIGHT)))
        result = env.step(PlayCardAction(3, Card(Suit.DIAMONDS, Rank.FIVE)))

        self.assertEqual(result.state.rounds[0].tricks[0].winner, 2)


if __name__ == "__main__":
    unittest.main()
