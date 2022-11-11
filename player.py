class PatchworkPlayer:

    def __init__(self):
        self.buttons = 5
        self.space = 0
        self.income = 0

    def can_afford(self, piece:dict):
        if piece['cost'] <= self.buttons:
            return True
        else:
            return False
    
    def buy_piece(self, piece:dict):
        self.income += piece['income']
        self.buttons -= piece['cost']
        self.space += piece['time']
        del piece

    def receive_income(self):
        self.buttons += self.income

