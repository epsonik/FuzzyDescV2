# obiekt fuzyfikacji dla obiektu 2d
class Lev3():
    def __init__(self, maxt, location, location_sal, location_names, orientation, orientation_sal, orientation_name):
        if maxt is None:
            maxt = 0
        if location is None:
            location = {}
        if location_sal is None:
            location_sal = []
        if location_names is None:
            location_names = {}
        if orientation is None:
            orientation = {}
        if orientation_sal is None:
            orientation_sal = []
        if orientation_name is None:
            orientation_name = {}
        self.maxt = maxt
        self.type = location
        self.location_sal = location_sal
        self.location_names = location_names
        self.orientation = orientation
        self.orientation_sal = orientation_sal
        self.orientation_name = orientation_name
        self.location_names = location_names
