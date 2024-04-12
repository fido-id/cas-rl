class Position:
    def __init__(self, x: int, y: int) -> None:
        self.x = x
        self.y = y

    def update(self, delta_x: int, delta_y: int) -> None:
        self.x = self.x + delta_x
        self.y = self.y + delta_y
