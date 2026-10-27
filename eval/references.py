"""Reference implementations: one correct, one carrying the misconception.

Ground truth for two things.

`scripts/validate_test_cases.py` runs every case in data/test_cases.json
against both. A case that fails the correct implementation is a broken
case; a concept whose buggy implementation passes every case has a suite
that proves nothing. Both happened in the first draft, which is why this
exists rather than the cases being trusted on sight.

`scripts/run_eval.py` uses the same pairs to measure how often the critic's
reading of code agrees with what running the code shows. Each buggy
implementation carries exactly the misconception named for its concept in
misconceptions.csv, so a disagreement is informative rather than ambiguous.

These are deliberately plain. They are not meant to be good code; they are
meant to be unambiguous about whether the bug is present.
"""

from __future__ import annotations

CORRECT = {}
BUGGY = {}

CORRECT["Linear Search"] = '''
def linear_search(arr, target):
    for item in arr:
        if item == target:
            return True
    return False
'''
BUGGY["Linear Search"] = '''
def linear_search(arr, target):
    for item in arr:
        if item == target:
            return True
        else:
            return False
'''

_BS_CORRECT = '''
def binary_search(arr, target):
    low, high = 0, len(arr) - 1
    while low <= high:
        mid = (low + high) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] > target:
            high = mid - 1
        else:
            low = mid + 1
    return -1
'''
CORRECT["Binary Search (Off-by-One)"] = _BS_CORRECT
BUGGY["Binary Search (Off-by-One)"] = _BS_CORRECT.replace(
    "high = mid - 1", "high = mid")
CORRECT["Binary Search (Loop Condition)"] = _BS_CORRECT
BUGGY["Binary Search (Loop Condition)"] = _BS_CORRECT.replace(
    "while low <= high", "while low < high")

CORRECT["Binary Search (Prerequisite)"] = '''
def search_any_order(arr, target):
    for i, value in enumerate(arr):
        if value == target:
            return i
    return -1
'''
BUGGY["Binary Search (Prerequisite)"] = '''
def search_any_order(arr, target):
    ordered = sorted(arr)
    low, high = 0, len(ordered) - 1
    while low <= high:
        mid = (low + high) // 2
        if ordered[mid] == target:
            return mid
        elif ordered[mid] > target:
            high = mid - 1
        else:
            low = mid + 1
    return -1
'''

CORRECT["Bubble Sort (Early Exit)"] = '''
def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        swapped = False
        for j in range(n - 1 - i):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
                swapped = True
        if not swapped:
            break
    return arr
'''
BUGGY["Bubble Sort (Early Exit)"] = '''
def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(n - 1 - i):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    return arr
'''

CORRECT["Selection Sort (Min Index Swap)"] = '''
def selection_sort(arr):
    n = len(arr)
    for i in range(n):
        min_idx = i
        for j in range(i + 1, n):
            if arr[j] < arr[min_idx]:
                min_idx = j
        if min_idx != i:
            arr[i], arr[min_idx] = arr[min_idx], arr[i]
    return arr
'''
BUGGY["Selection Sort (Min Index Swap)"] = '''
def selection_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(i + 1, n):
            if arr[j] < arr[i]:
                arr[i], arr[j] = arr[j], arr[i]
    return arr
'''

CORRECT["Insertion Sort (Shifting Bug)"] = '''
def insertion_sort(arr):
    for i in range(1, len(arr)):
        key = arr[i]
        j = i - 1
        while j >= 0 and arr[j] > key:
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = key
    return arr
'''
BUGGY["Insertion Sort (Shifting Bug)"] = '''
def insertion_sort(arr):
    for i in range(1, len(arr)):
        j = i - 1
        while j >= 0 and arr[j] > arr[j + 1]:
            arr[j + 1] = arr[j]
            j -= 1
    return arr
'''

_MS_CORRECT = '''
def merge_sort(arr, low=0, high=None):
    if high is None:
        high = len(arr) - 1
    if low >= high:
        return arr
    mid = (low + high) // 2
    merge_sort(arr, low, mid)
    merge_sort(arr, mid + 1, high)
    left = arr[low:mid + 1]
    right = arr[mid + 1:high + 1]
    i = j = 0
    k = low
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            arr[k] = left[i]
            i += 1
        else:
            arr[k] = right[j]
            j += 1
        k += 1
    while i < len(left):
        arr[k] = left[i]
        i += 1
        k += 1
    while j < len(right):
        arr[k] = right[j]
        j += 1
        k += 1
    return arr
'''
CORRECT["Merge Sort (Slice Overhead)"] = _MS_CORRECT
BUGGY["Merge Sort (Slice Overhead)"] = '''
def merge_sort(arr):
    if len(arr) <= 1:
        return arr
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    out = []
    i = j = 0
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            out.append(left[i])
            i += 1
        else:
            out.append(right[j])
            j += 1
    out.extend(left[i:])
    out.extend(right[j:])
    return out
'''
CORRECT["Merge Sort (Two-Pointer Comparison)"] = _MS_CORRECT
BUGGY["Merge Sort (Two-Pointer Comparison)"] = _MS_CORRECT.replace(
    """        if left[i] <= right[j]:
            arr[k] = left[i]
            i += 1
        else:
            arr[k] = right[j]
            j += 1
        k += 1""",
    """        if left[i] <= right[j]:
            arr[k] = left[i]
        else:
            arr[k] = right[j]
        k += 1""")

_QS_CORRECT = '''
def _partition(arr, low, high):
    pivot = arr[high]
    i = low - 1
    for j in range(low, high):
        if arr[j] <= pivot:
            i += 1
            arr[i], arr[j] = arr[j], arr[i]
    arr[i + 1], arr[high] = arr[high], arr[i + 1]
    return i + 1


def quick_sort(arr, low=0, high=None):
    if high is None:
        high = len(arr) - 1
    if low < high:
        p = _partition(arr, low, high)
        quick_sort(arr, low, p - 1)
        quick_sort(arr, p + 1, high)
    return arr
'''
CORRECT["Quick Sort (Partition Off-By-One)"] = _QS_CORRECT
BUGGY["Quick Sort (Partition Off-By-One)"] = _QS_CORRECT.replace(
    "quick_sort(arr, low, p - 1)", "quick_sort(arr, low, p)")
CORRECT["Quick Sort (Lomuto Partition)"] = _QS_CORRECT
BUGGY["Quick Sort (Lomuto Partition)"] = _QS_CORRECT.replace(
    """            i += 1
            arr[i], arr[j] = arr[j], arr[i]""",
    """            arr[i], arr[j] = arr[j], arr[i]
            i += 1""")

_HS = '''
def _sift(arr, n, i, left, right):
    largest = i
    l = left(i)
    r = right(i)
    if l < n and arr[l] > arr[largest]:
        largest = l
    if r < n and arr[r] > arr[largest]:
        largest = r
    if largest != i:
        arr[i], arr[largest] = arr[largest], arr[i]
        _sift(arr, n, largest, left, right)


def heap_sort(arr):
    left = lambda i: {LEFT}
    right = lambda i: {RIGHT}
    n = len(arr)
    for i in range(n // 2 - 1, -1, -1):
        _sift(arr, n, i, left, right)
    for i in range(n - 1, 0, -1):
        arr[0], arr[i] = arr[i], arr[0]
        _sift(arr, i, 0, left, right)
    return arr
'''
CORRECT["Heap Sort (0-Indexed Parent/Child)"] = _HS.replace(
    "{LEFT}", "2 * i + 1").replace("{RIGHT}", "2 * i + 2")
BUGGY["Heap Sort (0-Indexed Parent/Child)"] = _HS.replace(
    "{LEFT}", "2 * i").replace("{RIGHT}", "2 * i + 1")

CORRECT["Graph Traversal (BFS Visited Tracking)"] = '''
from collections import deque


def bfs_order(graph, start):
    visited = {start}
    queue = deque([start])
    order = []
    while queue:
        v = queue.popleft()
        order.append(v)
        for n in graph.get(v, []):
            if n not in visited:
                visited.add(n)
                queue.append(n)
    return order
'''
BUGGY["Graph Traversal (BFS Visited Tracking)"] = '''
from collections import deque


def bfs_order(graph, start):
    visited = set()
    queue = deque([start])
    order = []
    while queue:
        v = queue.popleft()
        visited.add(v)
        order.append(v)
        for n in graph.get(v, []):
            if n not in visited:
                queue.append(n)
    return order
'''

CORRECT["Graph Traversal (DFS Cycle Detection)"] = '''
def has_cycle(graph):
    WHITE, GRAY, BLACK = 0, 1, 2
    color = {v: WHITE for v in graph}

    def dfs(v):
        color[v] = GRAY
        for n in graph.get(v, []):
            if color.get(n, WHITE) == GRAY:
                return True
            if color.get(n, WHITE) == WHITE and dfs(n):
                return True
        color[v] = BLACK
        return False

    for v in list(graph):
        if color[v] == WHITE and dfs(v):
            return True
    return False
'''
BUGGY["Graph Traversal (DFS Cycle Detection)"] = '''
def has_cycle(graph):
    visited = set()

    def dfs(v):
        if v in visited:
            return True
        visited.add(v)
        for n in graph.get(v, []):
            if dfs(n):
                return True
        return False

    for v in list(graph):
        if dfs(v):
            return True
    return False
'''

CORRECT["Recursion"] = '''
def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n - 1)
'''
BUGGY["Recursion"] = '''
def factorial(n):
    if n == 1:
        return 1
    return n * factorial(n - 1)
'''

CORRECT["Python Lists"] = '''
def remove_all(items, value):
    return [x for x in items if x != value]
'''
BUGGY["Python Lists"] = '''
def remove_all(items, value):
    out = list(items)
    for x in out:
        if x == value:
            out.remove(x)
    return out
'''

CORRECT["Python Default Arguments"] = '''
def collect(item, bucket=None):
    if bucket is None:
        bucket = []
    bucket.append(item)
    return bucket
'''
BUGGY["Python Default Arguments"] = '''
def collect(item, bucket=[]):
    bucket.append(item)
    return bucket
'''

CORRECT["Python Object Mutability"] = '''
def append_without_mutating(items, value):
    return list(items) + [value]
'''
BUGGY["Python Object Mutability"] = '''
def append_without_mutating(items, value):
    out = items
    out.append(value)
    return out
'''

CORRECT["Hash Tables"] = '''
def build_index(pairs, buckets):
    table = [[] for _ in range(buckets)]
    for key, value in pairs:
        table[hash(key) % buckets].append((key, value))
    return table
'''
BUGGY["Hash Tables"] = '''
def build_index(pairs, buckets):
    table = [[] for _ in range(buckets)]
    for key, value in pairs:
        table[hash(key) % buckets] = [(key, value)]
    return table
'''

CORRECT["SQL Left Join"] = '''
def customers_with_recent_orders(conn):
    return conn.execute("""
        SELECT c.name, COALESCE(SUM(o.amount), 0)
        FROM customers c
        LEFT JOIN orders o
               ON o.customer_id = c.id AND o.amount > 100
        GROUP BY c.id, c.name
        ORDER BY c.name
    """).fetchall()
'''
BUGGY["SQL Left Join"] = '''
def customers_with_recent_orders(conn):
    return conn.execute("""
        SELECT c.name, COALESCE(SUM(o.amount), 0)
        FROM customers c
        LEFT JOIN orders o ON o.customer_id = c.id
        WHERE o.amount > 100
        GROUP BY c.id, c.name
        ORDER BY c.name
    """).fetchall()
'''

CORRECT["SQL Group By"] = '''
def big_spenders(conn):
    return conn.execute("""
        SELECT c.name, SUM(o.amount)
        FROM customers c
        JOIN orders o ON o.customer_id = c.id
        GROUP BY c.id, c.name
        HAVING SUM(o.amount) > 100
        ORDER BY c.name
    """).fetchall()
'''
BUGGY["SQL Group By"] = '''
def big_spenders(conn):
    return conn.execute("""
        SELECT c.name, SUM(o.amount)
        FROM customers c
        JOIN orders o ON o.customer_id = c.id
        WHERE o.amount > 100
        GROUP BY c.id, c.name
        ORDER BY c.name
    """).fetchall()
'''

CORRECT["Object-Oriented Programming"] = '''
class Basket:
    def __init__(self):
        self.items = []

    def add(self, item):
        self.items.append(item)
'''
BUGGY["Object-Oriented Programming"] = '''
class Basket:
    items = []

    def add(self, item):
        self.items.append(item)
'''
