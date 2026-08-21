import threading
import time

def worker(name, delay):
    for i in range(3):
        print(f"{name}: count {i}")
        time.sleep(delay)  # pause this thread only

# Create threads
t1 = threading.Thread(target=worker, args=("Thread-A", 1))
t2 = threading.Thread(target=worker, args=("Thread-B", 0.5))

# Start them (they now run concurrently)
t1.start()
t2.start()

# Wait for both to finish before continuing
t1.join()
t2.join()

print("Done!")