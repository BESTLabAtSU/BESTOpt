from modnn.utils import Mod

class modnn:
    def __init__(self):
        self.step = None

    def learn(self, args):
        mdl = Mod(args=args)
        mdl.data_ready()
        mdl.train()
        mdl.load()
        mdl.test()
        mdl.prediction_show()
        self.step = mdl.step_mdl()








