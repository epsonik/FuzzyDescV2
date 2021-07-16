class Scene():
    def __init__(self, fname,
                 im,
                 size,
                 onames,
                 ocols,
                 obj,
                 obj_num,
                 obj_org,
                 background,
                 background2):
        if fname is None:
            self.fname = ""
        self.fname = fname
        if im is None:
            self.im = None
        self.im = im
        self.size = size
        self.onames = onames
        self.ocols = ocols
        self.obj = obj
        self.obj_num = obj_num
        self.obj_org = obj_org
        self.background = background
        self.background2 = background2
