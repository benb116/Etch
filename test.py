import time
from threading import Thread

class worker(Thread):
    def run(self):
        for x in range(0,11):
            print(x)
            time.sleep(1)

class waiter(Thread):
    def run(self):
        for x in range(100,105):
            print(x)
            time.sleep(2)

def run():
    print('222')
    worker().start()
    waiter().start()

if __name__ == "__main__":
    run()