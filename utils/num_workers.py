import multiprocessing as mp
from time import time

from utils_data import get_dataloader

if __name__ == '__main__':
    print(f"number of CPU: {mp.cpu_count()}")
    for num_workers in range(2, mp.cpu_count(), 2):  
        train_loader, val_loader = get_dataloader(num_workers=num_workers, batch_size=64, pin_memory=True)
        start = time()
        for epoch in range(0, 2):
            for i, data in enumerate(train_loader, 0):
                if i > 100:
                    break
        end = time()
        print("Finish with:{} second, num_workers={}".format(end - start, num_workers))
