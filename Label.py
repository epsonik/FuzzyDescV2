class Label():
    def __init__(self, name, type, labelColor, group, description, labelData):
        if labelData is None:
            labelData = []
        self.name = name
        self.type = type
        self.labelColor = labelColor
        self.group = group
        self.description = description
        self.labelData = labelData
