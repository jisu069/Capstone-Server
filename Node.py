class Node:
    def __init__(self, id, point=None, parent=None):
        self.id = id
        self.point = point
        self.parent = parent
        self.cost = 0
        self.hueristic = 0
        self.score = 0

    def __lt__(self, other):
        return self.score < other.score
