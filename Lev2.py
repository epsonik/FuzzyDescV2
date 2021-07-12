from Lev1 import Type


class Lev2():
    def __init__(self, name, type, variant):
        if name is None:
            name = []
        if type is None:
            type = Type.F
        if variant is None:
            variant = []
        self.name = name
        self.type = type
        self.variant = variant
