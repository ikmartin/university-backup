import numpy as np
import spinspace as spinspace
from clustering import Model as ClustModel

class TopDown(ClustModel):
    def __init__(self, data):
        super().__init__(data=data)
        model()

    def model(self):
        pass

space1 = spinspace.Spinspace(shape=(2,2))

spin1 = (np.array([1,-1]), np.array([1,-1]))
spin2 = (np.array([1,1]), np.array([1,-1]))

print(space1.dist(spin1,spin2))
