class Obj():
    def __init__(self, name, pos, length):
        self.name = name
        self.pos = pos
        self.length = length


class ObjData():
    def __init__(self, number_of_b_boxes=0, left=None, right=None, horiz_len=None, up=None, down=None, vert_len=None, size=None):
        if size is None:
            size = []
        if vert_len is None:
            vert_len = []
        if down is None:
            down = []
        if up is None:
            up = []
        if horiz_len is None:
            horiz_len = []
        if right is None:
            right = []
        if left is None:
            left = []
        self.number_of_b_boxes = number_of_b_boxes
        self.left = left
        self.right = right
        self.horiz_len = horiz_len
        self.up = up
        self.down = down
        self.vert_len = vert_len
        self.size = size
