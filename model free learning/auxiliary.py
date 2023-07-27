import numpy as np

## Toma un vector de los valores de las acciones y retorna un vector de
## probabilidades con la probabilidad distribuida equitativamente sobre las
## acciones óptimas
## O 1 para una única acción óptima en el caso determinista
def optimalize(value_array, deterministic=False):
    if deterministic:
        ret = np.zeros_like(value_array)
        ret[value_array.argmax()] = 1.0
    else:
        best = np.flatnonzero(value_array == value_array.max())
        ret = np.array([1/len(best) if (x == value_array.max()) \
                        else 0.0 for x in value_array])
    return ret

## Lo mismo de arriba pero con un epsilon de exploración
def epsilonize(value_array, e=0.1, deterministic=False):
    if deterministic:
        ret = np.full(len(value_array), e/len(value_array))
        ret[value_array.argmax()] += 1.0 - e
    else:
        best = np.flatnonzero(value_array == value_array.max())
        ret = np.array([(1-e)/len(best) + e/len(value_array) \
                         if (x == value_array.max()) \
                         else e/len(value_array) for x in value_array])
    return ret

## Hardcodea la función excepto por n elementos
def hardcode(func, *args, n=0):
    hard = args[n:]
    def hard_func(*argumentos):
        return func(*argumentos, *hard)
    return hard_func

def manhattan_distance(dot1, dot2):
    return abs(dot1[0] - dot2[0]) + abs(dot1[1] - dot2[1])

if __name__ == "__main__":
    arr = np.array([1, 2, 2, 0.5])
    print(epsilonize(arr))
