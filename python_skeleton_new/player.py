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
        self.strong_hole = False
        my_cards = round_state.hands[active]
        card1 = my_cards[0] # '9s', 'Ad', 'Th'
        card2 = my_cards[1]

        rank1 = card1[0]
        suit1 = card1[1]
        rank2 = card2[0]
        suit2 = card2[1]

        if rank1 == rank2 or ((rank1 in "AKQJ") and (rank2 in "AKQJ")):
            self.strong_hole = True

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
        return dict[rank] - 2


    def set_bounty_aggression(self, my_cards, board_cards, strength):
        '''
        Calculate dynamic aggression based on bounty relevance and likelihood.
        '''
        my_cards = [eval7.Card(card) for card in my_cards]
        board_cards = [eval7.Card(card) for card in board_cards]

        deck = eval7.Deck()
        for card in my_cards + board_cards:
            if card in deck.cards:
                deck.cards.remove(card)

        bounty_in_hand = any(card.rank == self.rank_to_int(self.bounty_rank) for card in my_cards)
        bounty_on_board = any(card.rank == self.rank_to_int(self.bounty_rank) for card in board_cards)
        # print(f"my cards: {my_cards}, board cards: {board_cards}")
        # print(f"bounty_rank: {self.bounty_rank}, {bounty_in_hand=}, {bounty_on_board=}")

        # Probability of hitting the bounty
        remaining_draws = 5 - len(board_cards)
        bounty_prob = sum(1 for card in deck.cards if card.rank == self.rank_to_int(self.bounty_rank)) / len(deck.cards)
        ev = bounty_prob * remaining_draws

        high = 1.3
        if bounty_in_hand or bounty_on_board:
            return high
        else:
            return min(1.0 + 0.1 * ev, high)



    # def get_action(self, game_state, round_state, active):
    #     '''
    #     Main decision function for the bot.
    #     '''
    #     legal_actions = round_state.legal_actions()
    #     street = round_state.street
    #     my_cards = [card for card in round_state.hands[active]]
    #     board_cards = [card for card in round_state.deck[:street]]
    #     my_pip = round_state.pips[active]
    #     opp_pip = round_state.pips[1 - active]
    #     my_stack = round_state.stacks[active]
    #     pot_total = sum(round_state.pips)
    #     continue_cost = opp_pip - my_pip
    #     pot_odds = continue_cost / (pot_total + 0.1) if pot_total > 0 else 1.0

    #     strength = self.calculate_strength(my_cards, board_cards)

    #     if RaiseAction in legal_actions:
    #         min_raise, max_raise = round_state.raise_bounds()

    #     bounty_aggression = self.set_bounty_aggression(my_cards, board_cards, strength)
    #     # Adjust aggression based on opponent profile
    #     aggressive_tendency = self.opponent_profiles["aggressive"] / max(self.total_rounds, 1)
    #     tight_tendency = self.opponent_profiles["tight"] / max(self.total_rounds, 1)

    #     # Aggressive play for bounty and strong hands
    #     if strength > 0.7 or any(card[0] == self.bounty_rank for card in my_cards + board_cards):
    #         if RaiseAction in legal_actions:
    #             option1 = max_raise
    #             option2 = int(min_raise * bounty_aggression)
    #             new_raise = option2 if aggressive_tendency > 0.2 else option1
    #             new_raise = min_raise if new_raise < min_raise else max_raise if new_raise > max_raise else new_raise

    #             print(f"     STRONG HAND, RAISE {new_raise}, {option1=} {option2=}" )
    #             return RaiseAction(new_raise)

    #     bluff_chance = 0.15 + (0.1 * tight_tendency)
    #     if tight_tendency > 0.5 and random.random() < bluff_chance:
    #         print("     BLUFF")
    #         if RaiseAction in legal_actions:
    #             return RaiseAction(min_raise)


    #     # Conservative play for weak hands
    #     if strength < 0.4 and pot_odds > strength:
    #         print("         WEAK HAND, FOLD")
    #         if FoldAction in legal_actions:
    #             return FoldAction()

    #     # Default to call or check
    #     if CallAction in legal_actions and continue_cost <= my_stack * 0.1:
    #         print("            DEFAULT CALL")
    #         return CallAction()
    #     if CheckAction in legal_actions:
    #         return CheckAction()

    #     print("                 DEFAULT FOLD")  # Default to fold if no other action is available.
    #     return FoldAction()
    def get_action(self, game_state, round_state, active):
        '''
        Where the magic happens - your code should implement this function.
        Called any time the engine needs an action from your bot.

        Arguments:
        game_state: the GameState object.
        round_state: the RoundState object.
        active: your player's index.

        Returns:
        Your action.
        '''
        legal_actions = round_state.legal_actions()  # the actions you are allowed to take
        street = round_state.street  # 0, 3, 4, or 5 representing pre-flop, flop, turn, or river respectively
        my_cards = round_state.hands[active]  # your cards
        board_cards = round_state.deck[:street]  # the board cards
        my_pip = round_state.pips[active]  # the number of chips you have contributed to the pot this round of betting
        opp_pip = round_state.pips[1-active]  # the number of chips your opponent has contributed to the pot this round of betting
        my_stack = round_state.stacks[active]  # the number of chips you have remaining
        opp_stack = round_state.stacks[1-active]  # the number of chips your opponent has remaining
        continue_cost = opp_pip - my_pip  # the number of chips needed to stay in the pot
        my_bounty = round_state.bounties[active]  # your current bounty rank
        my_contribution = STARTING_STACK - my_stack  # the number of chips you have contributed to the pot
        opp_contribution = STARTING_STACK - opp_stack  # the number of chips your opponent has contributed to the pot

        strength = self.calculate_strength(my_cards, board_cards)
        # print(self.set_bounty_aggression(my_cards, board_cards, strength))
        print(f"Strength: {strength}")
        strength *= self.set_bounty_aggression(my_cards, board_cards, strength)
        print(f"New Strength: {strength}")

        pot_odds = continue_cost / (my_pip + opp_pip + 0.1)

        if RaiseAction in legal_actions:
           min_raise, max_raise = round_state.raise_bounds()  # the smallest and largest numbers of chips for a legal bet/raise
           min_cost = min_raise - my_pip  # the cost of a minimum bet/raise
           max_cost = max_raise - my_pip  # the cost of a maximum bet/raise
           raise_amt = int(min_raise + (max_raise - min_raise) * 0.3)

        if RaiseAction in legal_actions and self.strong_hole is True:

            for card in my_cards + board_cards:
                if card[0] == my_bounty:
                    return RaiseAction(max_raise)

            raise_prob = 0.8
            raise_amt = int(min_raise + (max_raise - min_raise) * 0.3)

            if random.random() < raise_prob:
                return RaiseAction(raise_amt)

        if RaiseAction in legal_actions:
            if random.random() < 0.5:
                if strength > 2*pot_odds:
                    raise_amount = int(min_raise + 0.1 * (max_raise - min_raise))
                    return RaiseAction(raise_amount)
                return RaiseAction(min_raise)
        if CheckAction in legal_actions:  # check-call
            return CheckAction()
        # if random.random() < 0.25:
        #     return FoldAction()
        if strength < 0.3 and pot_odds < 0.25:
            return FoldAction()
        if strength > 0.8 and RaiseAction in legal_actions:
            return RaiseAction(raise_amt)

        return CallAction()

if __name__ == '__main__':
    run_bot(Player(), parse_args())
