def evaluate_board_texture(board_cards):
    """
    Evaluates the board texture to determine if it's dry or wet.
    
    Args:
        board_cards (list of str): List of cards on the board in standard notation (e.g., ['Ks', '7h', '2c']).
    
    Returns:
        str: 'dry' for dry board, 'wet' for wet board.
    """
    ranks = [card[0] for card in board_cards]
    suits = [card[1] for card in board_cards]
    
    # Count rank frequencies to detect connectedness
    rank_counts = {rank: ranks.count(rank) for rank in set(ranks)}
    suit_counts = {suit: suits.count(suit) for suit in set(suits)}

    # Define criteria for wet boards
    connected_ranks = {'4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K'}
    flush_draw = any(count >= 3 for count in suit_counts.values())
    straight_draw = any(
        rank in connected_ranks for rank in ranks
    ) and len(set(ranks)) < len(ranks)

    # Wet board if there are flush or straight draw possibilities
    if flush_draw or straight_draw:
        return 'wet'
    else:
        return 'dry'

def adjust_strategy_based_on_texture(board_texture, hand_strength):
    """
    Adjusts bluffing or betting strategy based on board texture and hand strength.
    
    Args:
        board_texture (str): 'dry' or 'wet' board.
        hand_strength (float): Estimated strength of the player's hand (0 to 1).
    
    Returns:
        str: Suggested action ('bet', 'check', 'bluff').
    """
    if board_texture == 'dry':
        # Favor bluffing more on dry boards with fewer draw possibilities
        if hand_strength < 0.5:
            return 'bluff'
        else:
            return 'bet'
    elif board_texture == 'wet':
        # Avoid bluffing on wet boards unless hand strength is moderate or strong
        if hand_strength > 0.7:
            return 'bet'
        else:
            return 'check'

# Example usage
board = ['Ks', '7h', '2c']  # Dry board example
board_texture = evaluate_board_texture(board)
hand_strength = 0.4  # Example hand strength
action = adjust_strategy_based_on_texture(board_texture, hand_strength)
print(f"Board texture: {board_texture}, Suggested action: {action}")
