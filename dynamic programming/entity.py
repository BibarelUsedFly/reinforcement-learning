import numpy as np

class Entity:

    def __init__(self, img, x, y, tablero, gridsize):
        self.image = img
        self.x = x
        self.y = y
        self.tablero = tablero
        self.gridsize = gridsize

    @property
    def height(self):
        return self.image.get_height()

    @property
    def width(self):
        return self.image.get_width()

    @property
    def center(self):
        return np.array((self.width/2, self.height/2))

    @property
    def correction(self):
        return np.array((self.gridsize/2, self.gridsize/2))

    @property
    def position(self):
        return np.array((self.x, self.y))

    def draw(self): # Dibujo al robot en pantalla
        self.tablero.blit(self.image,
                          (self.position*self.gridsize - \
                           self.center + self.correction))