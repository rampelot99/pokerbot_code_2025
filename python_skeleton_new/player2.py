'''
Refined Poker Bot for No-Limit Bounty Hold’em
Focuses on strong bounty play, opponent profiling, and pot odds.
'''
from skeleton.actions import FoldAction, CallAction, CheckAction, RaiseAction
from skeleton.states import GameState, TerminalState, RoundState
from skeleton.states import STARTING_STACK, BIG_BLIND
from skeleton.bot import Bot
from skeleton.runner import parse_args, run_bot

import random
import eval7

class Player(Bot):
    def __init__(self):
        '''
        Initialize bot state and parameters.
        '''
        self.bounty_rank = None
        self.round_counter = 0
        self.opponent_profiles = {
            "aggressive": 0,
            "tight": 0,
            "neutral": 0
        }
        self.total_rounds = 0

    def handle_new_round(self, game_state, round_state, active):
        '''
        Called at the beginning of a new round.
        '''
        self.round_counter = game_state.round_num
        self.bounty_rank = round_state.bounties[active]

    def handle_round_over(self, game_state, terminal_state, active):
        '''
        Called at the end of a round.
        '''
        self.total_rounds += 1
        opponent_action = terminal_state.previous_state.pips[1 - active]

        # Update opponent profile based on observed behavior
        if opponent_action > BIG_BLIND * 2:
            self.opponent_profiles["aggressive"] += 1
        elif opponent_action == BIG_BLIND:
            self.opponent_profiles["tight"] += 1
        else:
            self.opponent_profiles["neutral"] += 1

    def calculate_strength(self, my_cards, board_cards):
        '''
        Monte Carlo simulation to evaluate hand strength.
        '''
        # MC_ITER = 100
        if len(board_cards) == 0:  # Pre-flop
            MC_ITER = 100
        elif len(board_cards) <= 3:  # Flop
            MC_ITER = 100
        elif len(board_cards) == 4:  # Turn
            MC_ITER = 150
        else:  # River
            MC_ITER = 200

        # Convert cards to eval7.Card
        my_cards = [eval7.Card(card) for card in my_cards]
        board_cards = [eval7.Card(card) for card in board_cards]

        deck = eval7.Deck()
        for card in my_cards + board_cards:
            if card in deck.cards:
                deck.cards.remove(card)

        score = 0
        for _ in range(MC_ITER):
            deck.shuffle()
            draw = deck.peek(7 - len(my_cards + board_cards))
            opp_hand = draw[:2]
            remaining_board = draw[2:]

            my_hand = my_cards + board_cards + remaining_board
            opp_full_hand = opp_hand + board_cards + remaining_board

            my_value = eval7.evaluate(my_hand)
            opp_value = eval7.evaluate(opp_full_hand)

            if my_value > opp_value:
                score += 1
            elif my_value < opp_value:
                score += 0
            else:
                score += 0.5

        win_rate = score / MC_ITER

        # # Adjust for bounty rank
        # bounty_in_hand = any(card.rank == self.bounty_rank for card in my_cards + board_cards)
        # if bounty_in_hand:
        #     win_rate *= 1.15  # Slight preference for bounty-related hands

        # bounty_prob = sum(1 for card in deck.cards if card.rank == self.bounty_rank) / len(deck.cards)
        # if bounty_prob > 0.1:
        #     win_rate *= (1 + 0.1 * bounty_prob)

        return win_rate

    def rank_to_int(self, rank: str) -> int:
        dict = {
            '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7,
            '8': 8, '9': 9, '10': 10, 'T': 10, 'J': 11, 'Q': 12, 'K': 13, 'A': 14
        }
        return dict[rank]


    def set_bounty_aggression(self, my_cards, board_cards):
        my_cards = [eval7.Card(card) for card in my_cards]
        board_cards = [eval7.Card(card) for card in board_cards]

        deck = eval7.Deck()
        for card in my_cards + board_cards:
            if card in deck.cards:
                deck.cards.remove(card)

        bounty_in_hand = any(card.rank == self.rank_to_int(self.bounty_rank) for card in my_cards)
        bounty_on_board = any(card.rank == self.rank_to_int(self.bounty_rank) for card in board_cards)

        print(f"my cards: {my_cards}, bounty_rank: {self.bounty_rank}, {bounty_in_hand=}, {bounty_on_board=}")

        # Calculate probability of hitting the bounty on remaining draws
        remaining_draws = 5 - len(board_cards)
        bounty_prob = sum(1 for card in deck.cards if card.rank == self.bounty_rank) / len(deck.cards)
        ev = bounty_prob * remaining_draws
        print(f"{bounty_in_hand=}, {bounty_on_board=}, {bounty_prob=}, {ev=}")

        if bounty_in_hand:
            return 1.3  # bounty is already present
        elif ev > 0.2:
            return 1.1  # bounty is likely to appear
        else:
            return 1.0 # default


    def get_action(self, game_state, round_state, active):
        '''
        Main decision function for the bot.
        '''
        legal_actions = round_state.legal_actions()
        street = round_state.street
        my_cards = [card for card in round_state.hands[active]]
        board_cards = [card for card in round_state.deck[:street]]
        my_pip = round_state.pips[active]
        opp_pip = round_state.pips[1 - active]
        my_stack = round_state.stacks[active]
        pot_total = sum(round_state.pips)
        continue_cost = opp_pip - my_pip
        pot_odds = continue_cost / (pot_total + 0.1) if pot_total > 0 else 1.0

        strength = self.calculate_strength(my_cards, board_cards)

        if RaiseAction in legal_actions:
            min_raise, max_raise = round_state.raise_bounds()

        bounty_aggression = self.set_bounty_aggression(my_cards, board_cards)
        # Adjust aggression based on opponent profile
        aggressive_tendency = self.opponent_profiles["aggressive"] / max(self.total_rounds, 1)
        tight_tendency = self.opponent_profiles["tight"] / max(self.total_rounds, 1)

        # Aggressive play for bounty and strong hands
        if strength > 0.7 or any(card[0] == self.bounty_rank for card in my_cards):
            if RaiseAction in legal_actions:
                new_raise = int(min_raise * bounty_aggression) if aggressive_tendency > 0.3 else max_raise
                return RaiseAction(new_raise)

        # Bluff more against tight opponents
        if tight_tendency > 0.5 and random.random() < 0.3:
            if RaiseAction in legal_actions:
                return RaiseAction(min_raise)

        # Conservative play for weak hands
        if strength < 0.4 and pot_odds > strength:
            if FoldAction in legal_actions:
                return FoldAction()

        # Default to call or check
        if CallAction in legal_actions and continue_cost <= my_stack * 0.1:
            return CallAction()
        if CheckAction in legal_actions:
            return CheckAction()

        return FoldAction()




if __name__ == '__main__':
    run_bot(Player(), parse_args())
