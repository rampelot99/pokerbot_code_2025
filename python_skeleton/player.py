'''
Simple example pokerbot, written in Python.
'''
from skeleton.actions import FoldAction, CallAction, CheckAction, RaiseAction
from skeleton.states import GameState, TerminalState, RoundState
from skeleton.states import NUM_ROUNDS, STARTING_STACK, BIG_BLIND, SMALL_BLIND
from skeleton.bot import Bot
from skeleton.runner import parse_args, run_bot

import random
import numpy as np
import json
import os
from collections import defaultdict

import eval7
from deuces import Deck, Evaluator, Card


class Player(Bot):
    '''
    A pokerbot using Q-learning for decision making.
    '''

    def __init__(self):
        '''
        Initialize Q-learning parameters and state-action value table.
        '''
        self.q_table_file = "q_values.json"
        self.load_q_table()
        
        self.initial_epsilon = 0.1
        self.min_epsilon = 0.01
        self.epsilon_decay = 0.997
        self.current_round = 0
        
        self.alpha = 0.1    # learning rate
        self.gamma = 0.9    # discount factor
        self.evaluator = Evaluator()
        
        # Track opponent patterns
        self.episode_memory = []
        self.opponent_aggression = 0.0
        self.opponent_vpip = 0.0
        self.hands_played = 0

    def load_q_table(self):
        """Load Q-table from file if it exists, otherwise create new."""
        try:
            with open(self.q_table_file, 'r') as f:
                self.q_table = defaultdict(lambda: defaultdict(float))
                saved_table = json.load(f)
                for state in saved_table:
                    for action, value in saved_table[state].items():
                        self.q_table[state][action] = value
        except (FileNotFoundError, json.JSONDecodeError):
            self.q_table = defaultdict(lambda: defaultdict(float))

    def save_q_table(self):
        """Save Q-table to file."""
        with open(self.q_table_file, 'w') as f:
            # Convert defaultdict to regular dict for JSON serialization
            table_dict = {k: dict(v) for k, v in self.q_table.items()}
            json.dump(table_dict, f)

    def get_epsilon(self):
        """Get current epsilon value with decay."""
        return max(self.min_epsilon, 
                  self.initial_epsilon * (self.epsilon_decay ** self.current_round))

    def get_state_key(self, game_state, round_state, active):
        """Create a discrete state representation."""
        position = "BB" if bool(active) else "SB"
        street = round_state.street
        
        # Enhanced hand strength calculation
        hole_cards = round_state.hands[active]
        board_cards = round_state.deck[:5] if hasattr(round_state, 'deck') else []
        hand_strength = self.evaluate_hand(hole_cards, board_cards)
        
        # Pot and stack information
        pot_size = round_state.pips[0] + round_state.pips[1]
        stack_ratio = min(1.0, round_state.stacks[active] / STARTING_STACK)
        pot_odds = round_state.pips[1-active] / (pot_size + round_state.pips[1-active]) if pot_size > 0 else 0
        
        # Opponent patterns
        opp_pattern = "A" if self.opponent_aggression > 0.6 else "P" if self.opponent_aggression < 0.4 else "N"
        
        # Bounty information
        has_bounty = self.check_bounty_hit(round_state.bounties[active], hole_cards, board_cards)
        
        return f"{position}_{street}_{hand_strength}_{pot_size}_{stack_ratio:.1f}_{pot_odds:.2f}_{opp_pattern}_{has_bounty}"

    def evaluate_hand(self, hole_cards, board_cards):
        """Evaluate hand strength on a scale of 0-9."""
        if not board_cards:
            # Preflop hand evaluation
            ranks = sorted([Card.get_rank_int(Card.new(card)) for card in hole_cards])
            suited = hole_cards[0][1] == hole_cards[1][1]
            
            # Premium hands
            if ranks[0] == ranks[1]:  # Pairs
                if ranks[0] >= 10:  # TT+
                    return 9
                elif ranks[0] >= 7:  # 77-99
                    return 8
                return 7
            
            # High card hands
            if suited:
                if ranks[1] == 12:  # Ax suited
                    return 8
                if ranks[1] == 11 and ranks[0] >= 9:  # KJs+
                    return 7
                if abs(ranks[0] - ranks[1]) == 1 and ranks[0] >= 8:  # High suited connectors
                    return 7
                if ranks[1] >= 10:  # Other suited high cards
                    return 6
                return 5
            else:
                if ranks[1] == 12 and ranks[0] >= 9:  # AK-AT
                    return 7
                if ranks[1] == 11 and ranks[0] >= 9:  # KQ-KT
                    return 6
                if ranks[1] >= 10 and ranks[0] >= 9:  # High cards
                    return 5
            return 4
        
        # Postflop evaluation
        deuces_cards = [Card.new(card) for card in hole_cards + board_cards]
        score = self.evaluator.evaluate(deuces_cards[:2], deuces_cards[2:])
        percentile = 1 - (score / 7462)  # Convert score to percentile
        return min(9, int(percentile * 10))

    def check_bounty_hit(self, bounty, hole_cards, board_cards):
        """Check if current hand hits the bounty."""
        if not bounty:
            return False
        all_cards = hole_cards + board_cards
        return any(Card.get_rank_int(Card.new(card)) == bounty for card in all_cards)

    def select_action(self, state_key, legal_actions):
        """Choose action using epsilon-greedy policy."""
        if random.random() < self.get_epsilon():
            return random.choice(legal_actions)
        
        # Get Q-values for all legal actions
        q_values = {action: self.q_table[state_key][str(action)] for action in legal_actions}
        
        # Choose action with highest Q-value
        return max(q_values.items(), key=lambda x: x[1])[0]

    def update_q_value(self, state, action, next_state, reward):
        """Update Q-value using Q-learning update rule."""
        if state and action:  # Only update if we have a previous state-action pair
            old_q = self.q_table[state][str(action)]
            
            # For terminal states or states we haven't seen before, use 0 as the future value
            if next_state == "terminal" or not self.q_table[next_state]:
                next_max_q = 0
            else:
                # Get all Q-values in the next state
                next_q_values = [self.q_table[next_state][str(a)] for a in self.q_table[next_state]]
                next_max_q = max(next_q_values) if next_q_values else 0
            
            # Update Q-value using the Q-learning formula
            new_q = old_q + self.alpha * (reward + self.gamma * next_max_q - old_q)
            self.q_table[state][str(action)] = new_q

    def handle_new_round(self, game_state, round_state, active):
        '''
        Called when a new round starts.
        '''
        self.last_state = None
        self.last_action = None
        self.load_q_table()
        self.current_round += 1

    def get_action_wrapper(self, game_state, round_state, active):
        """Wrapper for getting actions with Q-learning updates."""
        # Create state representation
        current_state = self.get_state_key(game_state, round_state, active)
        
        # Get legal actions
        legal_actions = []
        my_pip = round_state.pips[active]
        continue_cost = round_state.pips[1-active] - my_pip
        
        if continue_cost == 0:
            legal_actions.append(CheckAction())
        else:
            legal_actions.append(FoldAction())
            legal_actions.append(CallAction())
            
        # Can only raise if we have chips left
        if round_state.stacks[active] > continue_cost:
            min_raise, max_raise = round_state.raise_bounds()
            possible_raises = range(min_raise, max_raise + 1)
            legal_actions += [RaiseAction(raise_amount) for raise_amount in possible_raises]
        
        # Choose action using epsilon-greedy with decay
        if random.random() < self.get_epsilon():
            action = random.choice(legal_actions)
        else:
            q_values = {action: self.q_table[current_state][str(action)] for action in legal_actions}
            action = max(q_values.items(), key=lambda x: x[1])[0]
        
        # Store transition for later updates
        if self.last_state is not None:
            self.episode_memory.append({
                'state': self.last_state,
                'action': self.last_action,
                'next_state': current_state,
                'reward': 0  # Will be updated in handle_round_over
            })
        
        # Store current state and action for next transition
        self.last_state = current_state
        self.last_action = action
        
        return action

    def handle_round_over(self, game_state, terminal_state, active):
        '''
        Called when a round ends. Update Q-values with final reward.
        '''
        if self.episode_memory:
            # Calculate final reward with pot odds consideration
            final_reward = terminal_state.deltas[active]
            
            # Update opponent statistics
            self.hands_played += 1
            if terminal_state.deltas[1-active] > 0:
                self.opponent_aggression = (self.opponent_aggression * (self.hands_played - 1) + 
                                        (1 if final_reward < 0 else 0)) / self.hands_played
            
            # Add bounty bonus if applicable
            # if terminal_state.previous_state and hasattr(terminal_state.previous_state, 'deck'):
            #     if self.check_bounty_hit(terminal_state.previous_state.bounties[active],
            #                         terminal_state.previous_state.hands[active],
            #                         terminal_state.previous_state.deck[:5]):
            #         final_reward = 1.5 * final_reward + 20  # Increased bounty bonus
            
            # Scale reward based on pot size
            pot_size = sum(terminal_state.previous_state.pips)
            if pot_size > 0:
                final_reward *= (1 + pot_size / STARTING_STACK)
            
            # Update all transitions with properly scaled rewards
            for transition in self.episode_memory:
                # Scale reward based on action type and context
                action = transition['action']
                pot_size = sum(terminal_state.previous_state.pips)
                position = not bool(active)  # True if button/SB
                round_state = transition["state"]
                hand_strength = int(round_state.split('_')[2])
                pot_odds = float(round_state.split('_')[5])
                # my_pip = int(round_state.split('_')[3])
                # continue_cost = opp_pip - my_pip 

                # pot_odds = continue_cost / (pot_size + continue_cost)
                
                if isinstance(action, FoldAction):
                    # Penalize folding strong hands or when pot odds are good
                    
                    if hand_strength >= 7 or pot_odds < 0.2:  # Strong hand or good pot odds
                        transition['reward'] = -5
                    else:
                        transition['reward'] = 0  # Small penalty for folding weak hands
                        
                elif isinstance(action, RaiseAction):
                    # Reward aggressive play based on position and hand strength
                    if position and hand_strength >= 6:  # Strong hand in position
                        transition['reward'] = final_reward * 1.5
                    elif hand_strength >= 8:  # Very strong hand
                        transition['reward'] = final_reward * 1.3
                    else:
                        transition['reward'] = final_reward * 1.1
                        
                elif isinstance(action, CallAction):
                    # Reward calls based on pot odds and hand strength
                    if hand_strength >= int(pot_odds * 10):  # Hand strength justifies calling
                        transition['reward'] = final_reward * 1.2
                    else:
                        transition['reward'] = final_reward * 0.8
                else:
                    transition['reward'] = final_reward
                
                self.update_q_value(
                    transition['state'],
                    transition['action'],
                    transition['next_state'],
                    transition['reward']
                )
            
            self.save_q_table()

            # Clear episode memory
            self.episode_memory = []

    def get_action(self, game_state, round_state, active):
        '''
        Where the magic happens - the bot should return one of CallAction, CheckAction, FoldAction, or RaiseAction.
        '''
        # Get valid actions
        legal_actions = round_state.legal_actions()
        
        # Get current state
        state = self.get_state_key(game_state, round_state, active)
        
        # Get current epsilon for exploration
        epsilon = self.get_epsilon()
        
        # Pot odds calculation
        pot_size = sum(round_state.pips)
        to_call = round_state.pips[1-active] - round_state.pips[active]
        pot_odds = to_call / (pot_size + to_call) if pot_size + to_call > 0 else 0
        
        # Position advantage
        is_button = not bool(active)
        
        # Calculate minimum and maximum raise
        min_raise, max_raise = 0, 0
        if RaiseAction in legal_actions:
            min_raise = round_state.raise_bounds()[0]
            max_raise = round_state.raise_bounds()[1]
        
        # Adjust strategy based on hand strength and position
        hand_strength = int(state.split('_')[2])  # Get hand strength from state
        
        # Default to fold if hand is very weak
        if hand_strength <= 3 and RaiseAction in legal_actions and not is_button:
            return FoldAction()
        
        # More aggressive with position
        if is_button and hand_strength >= 5:
            if RaiseAction in legal_actions:
                raise_amount = min(max_raise, min_raise + int(pot_size * 0.75))
                return RaiseAction(raise_amount)
            elif CallAction in legal_actions:
                return CallAction()
        
        # Standard play
        if random.random() < epsilon:
            # Exploration: randomly choose an action
            chosen_action = random.choice(list(legal_actions))
            if chosen_action == RaiseAction:
                raise_size = random.randint(min_raise, max_raise)
                chosen_action = RaiseAction(raise_size)
            else:
                chosen_action = chosen_action()
        else:
            # Exploitation: choose best action from Q-table
            chosen_action = max(legal_actions, 
                              key=lambda x: self.q_table[state][str(x)])
            if chosen_action == RaiseAction:
                # Size raise based on hand strength and pot
                raise_amount = min(max_raise, 
                                 min_raise + int(pot_size * (0.5 + hand_strength * 0.1)))
                chosen_action = RaiseAction(raise_amount)
            else:
                chosen_action = chosen_action()
        
        # Store state and action for learning
        self.episode_memory.append({
            'state': state,
            'action': chosen_action,
            'next_state': None,
            'reward': 0
        })
        return chosen_action


if __name__ == '__main__':
    run_bot(Player(), parse_args())
