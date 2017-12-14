import time
from implementations import all_implementations
import numpy as np
import pandas as pd

def main():
    n = 10000
    data = []
    start = time.time()
    while True:
        random_array = np.random.rand(n)
        if time.time() - start > 55:
            break
        for sort in all_implementations:
            st = time.time()
            sort(random_array)
            en = time.time()
            data.append((sort.__name__, en-st))
    df = pd.DataFrame(data)
    df.to_csv('data.csv', index=False)


if __name__ == '__main__':
    main()
