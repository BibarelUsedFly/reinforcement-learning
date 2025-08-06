import numpy as np

def epsilonize(value_array: np.array, e=0.1, deterministic=False) -> np.array:
    '''Takes action value array and returns action choice probability array'''
    if deterministic:
        ret = np.full(len(value_array), e/len(value_array))
        ret[value_array.argmax()] += 1.0 - e
    else:
        best = np.flatnonzero(value_array == value_array.max())
        ret = np.array([(1-e)/len(best) + e/len(value_array) \
                         if (x == value_array.max()) \
                         else e/len(value_array) for x in value_array])
    return ret