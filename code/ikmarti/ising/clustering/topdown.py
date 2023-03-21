import numpy as np
import spinspace as spinspace
from clustering import Model as ClustModel


class TopDown(ClustModel):
    def __init__(self, data):
        super().__init__(data=data)
        self.model()

    def model(self):
        pass


if __name__ == "__main__":
    space1 = spinspace.Spinspace(shape=(2, 2))
    space = spinspace.Spinspace(shape=(2, 2, 4))
    spin1 = (3, 1)
    spin2 = (2, 1)

    print(space1.dist(spin1, spin2))
