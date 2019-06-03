from multiprocessing import Process
import time

def foo(i):
    for a in range(10000000):
        a += 1
    print('say hi', i)


if __name__ == '__main__':
    start = time.time()
    for i in range(10):
        p = Process(target=foo, args=(i,))
        p.start()
    print(time.time() - start)

    start = time.time()
    for i in range(10):
        foo(i)
    print(time.time() - start)
