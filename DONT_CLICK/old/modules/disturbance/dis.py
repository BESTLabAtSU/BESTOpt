import numpy as np

class ModDisturbance:
    """
    Handles external environmental and economic factors.
    Expected input: numpy array with shape (T, N)
    - Column 0: Ambient air temperature (Tamb)
    - Column 1: Solar radiation (Solar)
    - Column 2: Real-time utility prices (Price)
    ...
    - Column N: Add more if needed
    """

    def __init__(self, data_source: np.ndarray):
        if not isinstance(data_source, np.ndarray):
            raise ValueError("data_source must be a numpy array.")
        if data_source.shape[1] != 3:
            raise ValueError("data_source must have 3 columns: Tamb, Solar, Price.")
        self.data = data_source

    def get_values(self, timestep: int):
        """Returns disturbances for the current time step as a dict."""
        if timestep >= self.data.shape[0]:
            raise IndexError("Time step index is out of bounds for disturbances data.")

        Tamb = self.data[timestep, 0]
        Solar = self.data[timestep, 1]
        Price = self.data[timestep, 2]

        return {"Tamb": Tamb, "Solar": Solar, "Price": Price}