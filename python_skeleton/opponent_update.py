from skeleton.actions import FoldAction, CallAction, CheckAction, RaiseAction

class OpponentProfile:
    def __init__(self):
        self.aggression_frequency = 0  # Ratio of aggressive actions (bet/raise) to passive actions (check/call)
        self.continuation_bet_tendency = 0  # Frequency of betting after raising preflop
        self.showdown_hands = []  # List of hand strengths shown at showdown
        self.total_hands = 0  # Total number of hands observed
        self.raised_preflop = False

    def update_aggression(self, isRaise, amount = 0):
        """Update aggression frequency based on opponent's action."""
        self.total_hands += 1
        if isRaise:
            self.aggression_frequency = ((self.aggression_frequency * (self.total_hands - 1)) + amount*0.1 + 1) / self.total_hands
        else:
            self.aggression_frequency = (self.aggression_frequency * (self.total_hands - 1)) / self.total_hands

    def update_continuation_bet(self, preIsRaise, postIsRaise):
        """Update continuation betting tendency based on preflop and postflop actions."""
        if preIsRaise:
            if postIsRaise:
                self.continuation_bet_tendency = ((self.continuation_bet_tendency * (self.total_hands - 1)) + 1) / self.total_hands
            else:
                self.continuation_bet_tendency = (self.continuation_bet_tendency * (self.total_hands - 1)) / self.total_hands

    def add_showdown_hand(self, hand_strength):
        """Track the hand strength revealed at showdown."""
        self.showdown_hands.append(hand_strength)

    def estimate_tightness(self):
        """Estimate how tight or loose the opponent plays based on showdown hands."""
        if not self.showdown_hands:
            return 'unknown'
        average_hand_strength = sum(self.showdown_hands) / len(self.showdown_hands)
        return 'tight' if average_hand_strength > 0.7 else 'loose'


class PokerBot:
    def __init__(self):
        # self.opponents = {}  # Dictionary to store opponent profiles
        self.opponent = OpponentProfile()

    def get_opponent_profile(self, opponent_id):
        """Retrieve or create a profile for a specific opponent."""
        # if opponent_id not in self.opponents:
        #     self.opponents[opponent_id] = OpponentProfile()
        # return self.opponents[opponent_id]
        return self.opponent

    def decide_action(self, opponent_id, raised_preflop, action_postflop, hand_strength):
        """Decide action based on opponent behavior and hand strength."""
        profile = self.get_opponent_profile(opponent_id)
        profile.update_continuation_bet(raised_preflop, action_postflop)
        profile.update_aggression(action_postflop)

        tightness = profile.estimate_tightness()
        if profile.aggression_frequency < 0.3:  # Passive opponent
            return 'bluff' if hand_strength > 0.5 else CheckAction()
        elif profile.aggression_frequency > 0.7:  # Aggressive opponent
            return 'slow-play' if tightness == 'loose' and hand_strength > 0.8 else 'bet'
        else:  # Balanced opponent
            return 'bet' if hand_strength > 0.6 else CheckAction()


# # Example usage
# bot = PokerBot()
# opponent_id = 'player123'
# bot.decide_action(opponent_id, raised_preflop=True, action_postflop='bet', hand_strength=0.4)
