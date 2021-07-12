# obiekt fuzyfikacji dla obiektu 2d
class Lev3():
    def __init__(self, maxt, type, tsal, tname, orientation, osal, oname):
        if maxt is None:
            maxt = 0
        if type is None:
            type = {}
        if tsal is None:
            tsal = []
        if tname is None:
            tname = {}
        if orientation is None:
            orientation = {}
        if osal is None:
            osal = []
        if oname is None:
            oname = {}
        self.maxt = maxt
        self.type = type
        self.tsal = tsal
        self.tname = tname
        self.orientation = orientation
        self.osal = osal
        self.oname = oname
