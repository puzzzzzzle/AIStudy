from tqdm import tqdm
from time import sleep

with tqdm(total=100, desc='Iteration') as pbar:
    for i in range(100):
        pbar.update(1)
        pbar.set_postfix({
            'iteration': i,
        })
        sleep(0.1)