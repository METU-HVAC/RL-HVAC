
class OnOffController():
    '''
    OnOffController class opens the fan when there is an occupation, else closes.
    '''
    # mapping = {
    #     0: [ 5, 50, 0.0, 0.0],
    #     1: [ 5, 50, 0.0, 0.25],
    #     2: [ 5, 50, 0.0, 0.5],
    #     3: [ 5, 50, 0.0, 0.75],
    #     4: [ 5, 50, 0.0, 1.0],
    #     5 : OFF_ACTION
    # }
    def __init__(self, on_action=4, off_action=5):
        self.is_open = False
        self.on_action = on_action
        self.off_action = off_action
    def select_action(self, state):
        '''
        Act method for the controller
        '''
        occupancy = state[0][11]
        if occupancy > 0:
            return self.on_action
        else:
            return self.off_action