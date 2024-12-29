class SetpointController():
    '''
    Setpoint controller for CO2 Fan is a hystreresis controller that turns on the fan on full speed when the CO2 concentration
    is above 700 ppm and turns off the fan when the CO2 concentration is below 600 ppm.
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
        co2 = state[0][-2]
        if co2 > 700:
            self.is_open = True
            return self.on_action
        elif co2 < 600:
            self.is_open = False
            return self.off_action
        else:
            if self.is_open:
                return self.on_action
            else:
                return self.off_action