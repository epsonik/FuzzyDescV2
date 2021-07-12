from enum import Enum


class Type(Enum):
    F = 'Far'
    N = 'Near'
    C = "Close"
    E = "Edge"
    I = "Inside"
    L = "Left"
    A = "Above"


class Lev1():
    def __init__(self, name, type, variant, thr):
        if name is None:
            name = []
        if type is None:
            type = Type.F
        if variant is None:
            variant = []
        if thr is None:
            thr = ""
        self.name = name
        self.type = type
        self.variant = variant
        self.thr = thr
