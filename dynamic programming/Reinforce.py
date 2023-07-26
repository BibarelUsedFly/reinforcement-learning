import numpy as np

## Toma un vector de los valores de las acciones y retorna un vector de
## probabilidades con la probabilidad distribuida equitativamente sobre las
## acciones óptimas
def optimalize(value_array):
    best = np.flatnonzero(value_array == value_array.max())
    ret = np.array([1/len(best) if (x == value_array.max()) \
                    else 0.0 for x in value_array])
    return ret
