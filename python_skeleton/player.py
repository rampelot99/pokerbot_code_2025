'''
Simple example pokerbot, written in Python.
'''
from skeleton.actions import FoldAction, CallAction, CheckAction, RaiseAction
from skeleton.states import GameState, TerminalState, RoundState
from skeleton.states import NUM_ROUNDS, STARTING_STACK, BIG_BLIND, SMALL_BLIND
from skeleton.bot import Bot
from skeleton.runner import parse_args, run_bot
from opponent_update import OpponentProfile

import random

import eval7
from deuces import Deck, Evaluator, Card


class Player(Bot):
    '''
    A pokerbot.
    '''

    def __init__(self):
        '''
        Called when a new game starts. Called exactly once.

        Arguments:
        Nothing.

        Returns:
        Nothing.
        '''
        self.opponent = OpponentProfile()
        self.my_bounty = None
        self.big_blind = False

    def handle_new_round(self, game_state, round_state, active):
        '''
        Called when a new round starts. Called NUM_ROUNDS times.

        Arguments:
        game_state: the GameState object.
        round_state: the RoundState object.
        active: your player's index.

        Returns:
        Nothing.
        '''
        #my_bankroll = game_state.bankroll  # the total number of chips you've gained or lost from the beginning of the game to the start of this round
        #game_clock = game_state.game_clock  # the total number of seconds your bot has left to play this game
        #round_num = game_state.round_num  # the round number from 1 to NUM_ROUNDS
        my_cards = round_state.hands[active]  # your cards
        self.big_blind = bool(active)  # True if you are the big blind
        self.my_bounty = round_state.bounties[active]  # your current bounty rank
        
        self.strong_hole = False
        self.my_bounty_hit = False
        card1 = my_cards[0] # '9s', 'Ad', 'Th'
        card2 = my_cards[1] 

        rank1 = card1[0]
        suit1 = card1[1]
        rank2 = card2[0]
        suit2 = card2[1]

        if rank1 == rank2 or ((rank1 in "AKQJ") and (rank2 in "AKQJ")):
            self.strong_hole = True

    def is_bounty_hit(self, my_cards, board_cards):
        self.my_bounty_hit = self.my_bounty in [r[0] for r in my_cards + board_cards]
        return self.my_bounty_hit

    def handle_round_over(self, game_state, terminal_state, active):
        '''
        Called when a round ends. Called NUM_ROUNDS times.

        Arguments:
        game_state: the GameState object.
        terminal_state: the TerminalState object.
        active: your player's index.

        Returns:
        Nothing.
        '''
        #my_delta = terminal_state.deltas[active]  # your bankroll change from this round
        previous_state = terminal_state.previous_state  # RoundState before payoffs
        street = previous_state.street  # 0, 3, 4, or 5 representing when this round ended
        #my_cards = previous_state.hands[active]  # your cards
        opp_cards = previous_state.hands[1-active]  # opponent's cards or [] if not revealed
        
        self.my_bounty_hit = terminal_state.bounty_hits[active]  # True if you hit bounty
        opponent_bounty_hit = terminal_state.bounty_hits[1-active] # True if opponent hit bounty
        bounty_rank = previous_state.bounties[active]  # your bounty rank

        # The following is a demonstration of accessing illegal information (will not work)
        opponent_bounty_rank = previous_state.bounties[1-active]  # attempting to grab opponent's bounty rank

        self.opponent.add_showdown_hand(eval7.evaluate([eval7.Card(card) for card in opp_cards+previous_state.deck[:street+1]]))

        if self.my_bounty_hit:
            print("I hit my bounty of " + bounty_rank + "!")
        if opponent_bounty_hit:
            print("Opponent hit their bounty of " + opponent_bounty_rank + "!")

    def calculate_strength(self, my_cards, board_cards):
        print(f"my cards: {my_cards}")
        print(f"board cards: {board_cards}")
        
        # define amount of times we want to do monte-carlo iterations
        MC_ITER = 100
        
        deck = Deck()
        evaluator = Evaluator()
        my_hand = [Card.new(card) for card in my_cards]
        community_cards = [Card.new(card) for card in board_cards]
            
        score = 0
        for _ in range(MC_ITER):
            
            deck.shuffle()
            for card in my_hand + community_cards:
                deck.cards.remove(card)

            opp_draw = deck.draw(2)
            
            my_value = evaluator.evaluate(my_hand, community_cards)
            opp_value = evaluator.evaluate(opp_draw, community_cards)
            
            score += (opp_value - my_value)*0.2
        
        win_rate = score / MC_ITER
        print(f"win rate: {win_rate}")
        
        return win_rate
    
    def calculate_pot_odds(self, my_cards, board_cards):
        '''
        Count the number of cards that complete our hand (outs)
        Multiply this number by 2 (1/52 cards gives ~2% chance of getting a specific
        card)
        If we have two cards left to see (turn and river), multiply by 2
        '''
        deck = Deck()
        evaluator = Evaluator()
        my_hand = [Card.new(card) for card in my_cards]
        community_cards = [Card.new(card) for card in board_cards]
        card_num = len(board_cards)
        # Remove known cards from the deck
        known_cards = my_hand + community_cards
        for card in known_cards:
            deck.cards.remove(card)

        # Simulate the turn and river (all combinations of unseen cards)
        outs_count = 0
        if card_num == 3:
            for turn_card in deck.cards:
                for river_card in [c for c in deck.cards if c != turn_card]:
                    full_community = community_cards + [turn_card, river_card]
                    rank = evaluator.evaluate(my_hand, full_community)
                    # Consider high-ranking thresholds (e.g., less than 5000 is usually strong)
                    if rank <= 5000:  # Adjust threshold for your definition of a "high" rank
                        outs_count += 1
            return 4 * outs_count
        # Simulate river
        elif card_num == 4:
            for river_card in deck.cards:
                full_community = community_cards + [river_card]
                rank = evaluator.evaluate(my_hand, full_community)
                # Consider high-ranking thresholds (e.g., less than 5000 is usually strong)
                if rank <= 5000:  # Adjust threshold for your definition of a "high" rank
                    outs_count += 1
            return 2 * outs_count
        # know all our cards
        return evaluator.evaluate(my_hand, community_cards)


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

        isRaise, amount = False, 0
        if street == 0:
            if opp_pip > 2:
                self.opponent.raised_preflop = True
                isRaise, amount = True, opp_pip
            self.opponent.update_aggression(isRaise, amount)
        else:
            # post-flop
            
            # Calculate opponent's last action
            if self.big_blind and opp_pip == 0: # was a check
                isRaise = False
            elif self.big_blind and opp_pip > 0:
                isRaise, amount = True, opp_pip
            elif opp_pip == 0 or opp_pip == my_pip:
                isRaise = False
            else:
                isRaise, amount = True, opp_pip

            self.opponent.update_aggression(isRaise)
            self.opponent.update_continuation_bet(self.opponent.raised_preflop, isRaise)

        print(street, self.opponent.raised_preflop, my_pip, opp_pip)

        strength = 0
        if len(board_cards) == 5:
            strength = self.calculate_strength(my_cards, board_cards)
        elif len(board_cards) > 2:
            strength = self.calculate_pot_odds(my_cards, board_cards)

        tightness = self.opponent.estimate_tightness()
        # if self.opponent.aggression_frequency < 0.3:  # Passive opponent
        #     return 'bluff' if strength > 0.5 else CheckAction()
        # elif self.opponent.aggression_frequency > 0.7:  # Aggressive opponent
        #     return 'slow-play' if tightness == 'loose' and strength > 0.8 else 'bet'
        # else:  # Balanced opponent
        #     return 'bet' if strength > 0.6 else CheckAction()

        pot_odds = 0
        if my_pip:
            pot_odds = continue_cost / (my_pip + opp_pip + continue_cost)

        if self.is_bounty_hit(my_cards, board_cards):
            pot_odds = continue_cost / (my_pip + opp_pip + opp_pip * 0.5 + 10 + continue_cost)
        
        if RaiseAction in legal_actions:
           min_raise, max_raise = round_state.raise_bounds()  # the smallest and largest numbers of chips for a legal bet/raise
           min_cost = min_raise - my_pip  # the cost of a minimum bet/raise
           max_cost = max_raise - my_pip  # the cost of a maximum bet/raise

        if RaiseAction in legal_actions and self.strong_hole is True:

            if self.is_bounty_hit:
                return RaiseAction(max_raise)

            raise_prob = 0.8
            raise_amt = int(min_raise + (max_raise - min_raise) * 0.2)

            if random.random() < raise_prob:
                return RaiseAction(raise_amt)
        
        if RaiseAction in legal_actions:
            if self.opponent.aggression_frequency < 0.3:  # Passive opponent
                if strength > 1.5*pot_odds:
                    if random.random() < 0.7:
                        raise_amount = int(min_raise + 0.1 * (max_raise - min_raise))
                        return RaiseAction(raise_amount)
                    return RaiseAction(min_raise)
            elif self.opponent.aggression_frequency > 0.7:  # Aggressive opponent
                return RaiseAction(int(min_raise + 0.1 * (max_raise - min_raise))) if tightness == 'loose' and strength > 0.8 else RaiseAction(min_raise) 

            else:  # Balanced opponent
                if random.random() < 0.5:
                    if strength > 1.5*pot_odds:
                        raise_amount = int(min_raise + 0.1 * (max_raise - min_raise))
                        return RaiseAction(raise_amount)
                    return RaiseAction(min_raise)
        if CheckAction in legal_actions:  # check-call
            return CheckAction()
        if random.random() < 0.25:
            return FoldAction()
        return CallAction()


if __name__ == '__main__':
    run_bot(Player(), parse_args())
