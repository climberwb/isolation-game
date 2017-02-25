"""This file contains all the classes you must complete for this project.

You can use the test cases in agent_test.py to help during development, and
augment the test suite with your own test cases to further test your code.

You must test your agent's strength against a set of agents with known
relative strength using tournament.py and include the results in your report.
"""
import random

    

class Timeout(Exception):
    """Subclass base exception for code clarity."""
    pass

    
def opponent_chase_heuristic(game,player):
    """ Gives a higher priority to the moves that limit the opponents moves 
        while still accounting for maximizing the number of the players own 
        moves. this results in a chasing effect. 
    """
    opp_moves = len(game.get_legal_moves(game.get_opponent(player)))
    player_moves = len(game.get_legal_moves(player))
    import random
    return float(player_moves - 2*opp_moves)

def shy_player_heuristic(game,player):
    """ (Opposite of opponent_chase_heuristic) Gives a higher priority to the moves that increase the players moves 
        while still accounting for maximizing the number of the players own 
        moves. this results in a chasing effect. 
    """
    opp_moves = len(game.get_legal_moves(game.get_opponent(player)))
    player_moves = len(game.get_legal_moves(player))
    return float(2*player_moves - opp_moves)
    
def opponent_minimum_heuristic(game,player):
    """" Gives the highest score to the moves that 
         minimizes the opponents moves the most. The
         score is inversly proportional to the ammount of remaining moves
         the opponent has. 
    """
    opp_moves = len(game.get_legal_moves(game.get_opponent(player)))
    division_by_zero_factor =1e-4
    opp_moves_inverse  =  1/(opp_moves+division_by_zero_factor)
    return opp_moves_inverse

    
     
def custom_score(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    
    # TODO: finish this function!
    if game.is_loser(player):
        return float('-inf')

    if game.is_winner(player):
        return float('inf')

    return opponent_chase_heuristic(game,player)



class CustomPlayer:
    """Game-playing agent that chooses a move using your evaluation function
    and a depth-limited minimax algorithm with alpha-beta pruning. You must
    finish and test this player to make sure it properly uses minimax and
    alpha-beta to return a good move before the search time limit expires.

    Parameters
    ----------
    search_depth : int (optional)
        A strictly positive integer (i.e., 1, 2, 3,...) for the number of
        layers in the game tree to explore for fixed-depth search. (i.e., a
        depth of one (1) would only explore the immediate sucessors of the
        current state.)

    score_fn : callable (optional)
        A function to use for heuristic evaluation of game states.

    iterative : boolean (optional)
        Flag indicating whether to perform fixed-depth search (False) or
        iterative deepening search (True).

    method : {'minimax', 'alphabeta'} (optional)
        The name of the search method to use in get_move().

    timeout : float (optional)
        Time remaining (in milliseconds) when search is aborted. Should be a
        positive value large enough to allow the function to return before the
        timer expires.
    """

    def __init__(self, search_depth=3, score_fn=custom_score,
                 iterative=True, method='minimax', timeout=15.):
        self.search_depth = search_depth
        self.iterative = iterative
        self.score = score_fn
        self.method = method
        self.time_left = None
        self.TIMER_THRESHOLD = timeout

    def get_move(self, game, legal_moves, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        This function must perform iterative deepening if self.iterative=True,
        and it must use the search method (minimax or alphabeta) corresponding
        to the self.method value.

        **********************************************************************
        NOTE: If time_left < 0 when this function returns, the agent will
              forfeit the game due to timeout. You must return _before_ the
              timer reaches 0.
        **********************************************************************

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        legal_moves : list<(int, int)>
            A list containing legal moves. Moves are encoded as tuples of pairs
            of ints defining the next (row, col) for the agent to occupy.

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """

        self.time_left = time_left
        # TODO: finish this function!
        
        # Perform any required initializations, including selecting an initial
        # move from the game board (i.e., an opening book), or returning
        # immediately if there are no legal moves
        if legal_moves == [] or legal_moves==None:
            return (-1,-1)
        move=None       
        try:
            # The search method call (alpha beta or minimax) should happen in
            # here in order to avoid timeout. The try/except block will
            # automatically catch the exception raised by the search method
            # when the timer gets close to expiring
            # print(self.method)
            very_large_depth = 1000000000000
            if self.method == 'minimax':
                #TODO add minimax call here
                if self.iterative:
                    for d in range(1,very_large_depth):
                        if self.time_left() < self.TIMER_THRESHOLD:
                            raise Timeout()
                        score, move = self.minimax(game,d)
                        if game.get_legal_moves()==[]:
                            break
                else:
                    score, move = self.minimax(game,self.search_depth)
                # print("\nOld state:\n{}".format(game.forecast_move(move).to_string()))
                return move
                
            elif self.method == "alphabeta":
                 #TODO add minimax call here
                if self.iterative:
                    for d in range(1,very_large_depth):
                        if self.time_left() < self.TIMER_THRESHOLD:
                            raise Timeout()
                        score, move = self.alphabeta(game,d)
                        if game.get_legal_moves()==[]:
                            break
                else:
                    score, move = self.alphabeta(game,self.search_depth)
                # print("\nOld state:\n{}".format(game.forecast_move(move).to_string()))
                return move
                
                # minimax(game, 1 )
        
        except Timeout:
            # Handle any actions required at timeout, if necessary
            # print("\nOld state:\n{}".format(game.forecast_move(move).to_string()))
            return move

        # Return the best move from the last completed search iteration
        # raise NotImplementedError
        
    # def build_tree(games,depth,game=None):
    #     if game:
    #         games.append([game.forecast_move(move) for move in game.get_legal_moves()])
    #         return build_tree(games,depth-1)
    #     else
               
    def minimax(self, game, depth, maximizing_player=True):
        """Implement the minimax search algorithm as described in the lectures.

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        maximizing_player : bool
            Flag indicating whether the current search depth corresponds to a
            maximizing layer (True) or a minimizing layer (False)

        Returns
        -------
        float
            The score for the current search branch

        tuple(int, int)
            The best move for the current branch; (-1, -1) for no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project unit tests; you cannot call any other
                evaluation function directly.
        """
        
        if self.time_left() < self.TIMER_THRESHOLD:
            raise Timeout()
        score, best_move = 0, (-1, -1)
        if game.is_loser(self) or game.is_winner(self) or (depth == 0):
            return self.score(game, self), best_move
        if maximizing_player:
            states = [(self.minimax(game.forecast_move(move),depth-1,False),move) 
                            for move in game.get_legal_moves(game.active_player)]
            score, best_move = max(states, key=lambda x: x[0][0])
        else:
            states = [(self.minimax(game.forecast_move(move),depth-1,True),move) 
                            for move in game.get_legal_moves(game.active_player)]
            score, best_move  = min(states, key=lambda x: x[0][0])
        return score[0], best_move

        # TODO: finish this function!
        # raise NotImplementedError

    def alphabeta(self, game, depth, alpha=float("-inf"), beta=float("inf"), maximizing_player=True):
        """Implement minimax search with alpha-beta pruning as described in the
        lectures.

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        alpha : float
            Alpha limits the lower bound of search on minimizing layers

        beta : float
            Beta limits the upper bound of search on maximizing layers

        maximizing_player : bool
            Flag indicating whether the current search depth corresponds to a
            maximizing layer (True) or a minimizing layer (False)

        Returns
        -------
        float
            The score for the current search branch

        tuple(int, int)
            The best move for the current branch; (-1, -1) for no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project unit tests; you cannot call any other
                evaluation function directly.
        """
        # print(self.TIMER_THRESHOLD)
        if self.time_left() < self.TIMER_THRESHOLD:
            raise Timeout()
        score, best_move = 0, (-1, -1)
        if game.is_loser(self) or game.is_winner(self) or (depth == 0):
            return self.score(game, self), best_move
        if maximizing_player:   
            states = []
            score=float("inf")
            for i,move in enumerate(game.get_legal_moves(game.active_player)):
                if self.time_left() < self.TIMER_THRESHOLD:
                    raise Timeout()
                score,best_move = self.alphabeta(game.forecast_move(move),depth-1,alpha,beta,False)
                if score >= beta:
                    return score,move
                else:
                    alpha = max(score,alpha)
                    states.append((score,move))
            return  max(states, key=lambda x: x[0])
        else:   
            states = []
            score=float("-inf")
            for i,move in enumerate(game.get_legal_moves(game.active_player)):
                if self.time_left() < self.TIMER_THRESHOLD:
                    raise Timeout()
                score,best_move = self.alphabeta(game.forecast_move(move),depth-1,alpha,beta,True)
                if score <= alpha:
                    return score,move
                else:
                    beta = min(score,beta)
                    states.append((score,move))
            return min(states, key=lambda x: x[0])
        return score, best_move
        # TODO: finish this function!
        raise NotImplementedError

