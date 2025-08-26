class mod:
    def __init__(self):
        self.modules = {}
        self.disturbance = {}
        self.action = {"signal": {}}
        self.state = {"thermal": {}, "electric": {}}

    def add_module(self, module):
        """Add a module to the framework"""
        self.modules[module.module_id] = module

    def remove_module(self, module_id: str):
        """Remove a module from the framework"""
        if module_id in self.modules:
            del self.modules[module_id]
        else:
            print("Module {} was not found in framework".format(module_id))

    def get_observation(self):
        state = {}
        state["signal_loop"] = self.signal_loop
        state["electric_loop"] = self.electric_loop
        state["thermal_loop"] = self.thermal_loop
        self.state = state

    def get_reference(self):
        pass

    # def get_action(self):
    #     action = controller(self.state, self.reference)

