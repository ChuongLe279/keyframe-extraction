import threading
import time

def print_items(label, items):
    for item in items:
        print(f"{label}: {item}")
        time.sleep(1)

# Define however many tasks you want here
tasks = [
    ("Numbers", [1, 2, 3, 4, 5]),
    ("Letters", ['A', 'B', 'C', 'D', 'E']),
    ("Colors", ['Red', 'Green', 'Blue', 'Yellow', 'Purple']),
    ("Animals", ['Cat', 'Dog', 'Bird', 'Fish', 'Fox']),
]

def run_sequential():
    print("\n--- Running Sequentially (no threading) ---")
    start = time.time()

    for label, items in tasks:
        print_items(label, items)

    elapsed = time.time() - start
    print(f"Sequential time: {elapsed:.2f} seconds")
    return elapsed


def run_multithreaded():
    print("\n--- Running with Multithreading ---")
    start = time.time()

    threads = []
    for label, items in tasks:
        t = threading.Thread(target=print_items, args=(label, items))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    elapsed = time.time() - start
    print(f"Multithreaded time: {elapsed:.2f} seconds")
    return elapsed


if __name__ == "__main__":
    seq_time = run_sequential()
    thread_time = run_multithreaded()

    print("\n--- Summary ---")
    print(f"Sequential:     {seq_time:.2f}s")
    print(f"Multithreaded:  {thread_time:.2f}s")
    print(f"Speedup:        {seq_time / thread_time:.2f}x faster")