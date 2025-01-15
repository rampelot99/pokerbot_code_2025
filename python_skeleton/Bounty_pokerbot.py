import random
import eval7
from skeleton.actions import FoldAction, CallAction, CheckAction, RaiseAction
from skeleton.states import GameState, TerminalState, RoundState
from skeleton.states import STARTING_STACK, BIG_BLIND
from skeleton.bot import Bot

import numpy as np
from sklearn.ensemble import RandomForestClassifier

class Player(Bot):
    def __init__(self):
        """
        Initializes the bot and machine learning model.
        """
        self.bounty_rank = None
        self.rf_model = RandomForestClassifier()
        self.training_data = []

    def handle_new_round(self, game_state, round_state, active):
        """
        Called at the beginning of a new round.
        """
        self.bounty_rank = round_state.bounties[active]

    def calculate_strength(self, my_cards, board_cards):
        """
        Monte Carlo simulation to estimate hand strength.
        """
        MC_ITER = 100
        my_cards = [eval7.Card(card) for card in my_cards]
        board_cards = [eval7.Card(card) for card in board_cards]

        deck = eval7.Deck()
        for card in my_cards + board_cards:
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
            elif my_value == opp_value:
                score += 0.5

        return score / MC_ITER

    def should_play_for_bounty(self, my_cards, board_cards):
        """
        Determines if the bot should play aggressively based on bounty rank.
        """
        return any(card[0] == self.bounty_rank for card in my_cards + board_cards)

    def predict_action(self, game_state, round_state, active):
        """
        Predict the opponent's next action using the machine learning model.
        """
        features = self.extract_features(game_state, round_state, active)
        if self.rf_model and len(self.training_data) > 100:
            return self.rf_model.predict([features])[0]
        return random.choice(["fold", "call", "raise"])

    def extract_features(self, game_state, round_state, active):
        """
        Extract features for machine learning model.
        """
        my_cards = round_state.hands[active]
        board_cards = round_state.deck[:round_state.street]
        my_pip = round_state.pips[active]
        opp_pip = round_state.pips[1 - active]
        my_stack = round_state.stacks[active]
        opp_stack = round_state.stacks[1 - active]

        features = [
            len(board_cards),  # Number of community cards
            my_pip,            # My contribution to the pot
            opp_pip,           # Opponent's contribution to the pot
            my_stack,          # My remaining stack
            opp_stack,         # Opponent's remaining stack
            self.bounty_rank in [card[0] for card in my_cards],  # My bounty rank in my cards
        ]
        return features

    def update_training_data(self, game_state, round_state, active, action):
        """
        Collect training data for machine learning model.
        """
        features = self.extract_features(game_state, round_state, active)
        self.training_data.append((features, action))
        if len(self.training_data) > 1000:
            self.training_data.pop(0)

    def train_model(self):
        """
        Train the machine learning model with collected data.
        """
        if len(self.training_data) > 100:
            features, labels = zip(*self.training_data)
            self.rf_model.fit(features, labels)

    def get_action(self, game_state, round_state, active):
        """
        Main decision-making function for the bot.
        """
        legal_actions = round_state.legal_actions()
        street = round_state.street
        my_cards = round_state.hands[active]
        board_cards = round_state.deck[:street]
        my_pip = round_state.pips[active]
        opp_pip = round_state.pips[1 - active]
        my_stack = round_state.stacks[active]
        continue_cost = opp_pip - my_pip
        pot_total = STARTING_STACK - my_stack + STARTING_STACK - round_state.stacks[1 - active]

        strength = self.calculate_strength(my_cards, board_cards)
        bounty_play = self.should_play_for_bounty(my_cards, board_cards)

        predicted_action = self.predict_action(game_state, round_state, active)

        if RaiseAction in legal_actions:
            min_raise, max_raise = round_state.raise_bounds()

        # Aggressive play if bounty rank is involved or strong hand strength
        if bounty_play or strength > 0.7:
            if RaiseAction in legal_actions:
                return RaiseAction(min_raise if random.random() < 0.5 else max_raise)
            return CallAction() if CallAction in legal_actions else CheckAction()

        # Conservative play for weaker hands
        if strength < 0.4:
            return FoldAction() if FoldAction in legal_actions else CheckAction()

        # Use ML-based prediction if the strength is moderate
        if predicted_action == "raise" and RaiseAction in legal_actions:
            return RaiseAction(min_raise)
        elif predicted_action == "call" and CallAction in legal_actions:
            return CallAction()
        else:
            return CheckAction()

if __name__ == '__main__':
    from skeleton.runner import parse_args, run_bot
    run_bot(Player(), parse_args())
