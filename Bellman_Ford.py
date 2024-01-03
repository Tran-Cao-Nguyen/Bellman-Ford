from networkx import *
import matplotlib.pyplot as plt
import random
from collections import deque
from time import time


INF = 10**9  # biến đại diện cho vô cùng

# đinh nghĩa một cạnh đồ thị
class Edge:
    def __init__(self, from_, to, capacity, cost):
        self.from_ = from_
        self.to = to
        self.capacity = capacity
        self.cost = cost

# Thuật toán tìm đường ngắn nhất sử dụng Bellman-Ford
"""
 n: số đỉnh
V0: đỉnh bắt đầu 
adj: ma trận liền kề
cost: ma trận chứa chi phí
capacity: ma trận sức chứa flow
"""


def Bellman_Ford(n, v0, adj, cost, capacity):
    dist = [INF] * n  # mảng chứa cost từ V0 tới đỉnh bất kỳ
    dist[v0] = 0
    isInQueue = [False] * n
    queue = deque([v0])  # Hàng đợi
    parent = [-1] * n  # Mảng chưa đỉnh cha của một đỉnh
    count = [0] * n

    while queue:
        u = queue.popleft()
        isInQueue[u] = False
        for v in adj[u]:
            if capacity[u][v] > 0 and dist[v] > dist[u] + cost[u][v]:
                dist[v] = dist[u] + cost[u][v]
                parent[v] = u
                if not isInQueue[v]:
                    queue.append(v)
                    isInQueue[v] = True
                count[v] += 1
                # Nếu một đỉnh được thêm vào hàng đợi hơn n lần thì có chu trình âm
                if count[v] > n:
                    return None
    return dist, parent


"""
Thuật toán min cost flow:
N: số đỉnh
edges: số cạnh
K: tổng luồng từ source
s: node source
t: node sink

"""


def min_cost_flow(N, edges, K, s, t):
    adj = [[] for _ in range(N)]
    cost = [[0] * N for _ in range(N)]
    capacity = [[0] * N for _ in range(N)]
    #Khởi tạo mảng liền kề, cost và capacity
    for e in edges:
        adj[e.from_].append(e.to)
        adj[e.to].append(e.from_)
        cost[e.from_][e.to] = e.cost
        cost[e.to][e.from_] = -e.cost
        capacity[e.from_][e.to] = e.capacity
    # flow đã được gửi đi và cost
    flow = 0
    cost_ = 0

    while flow < K:
        result = Bellman_Ford(N, s, adj, cost, capacity)
        if result is None:
            raise ValueError("Graph contains negative weight cycle")
        distance, parent = result

        if distance[t] == INF:
            break
            # thêm node vào mảng path
        subflow = K - flow
        cur = t
        path = []
        while cur != s:
            subflow = min(subflow, capacity[parent[cur]][cur])
            path.append(cur)
            cur = parent[cur]
        path.append(s)
        path = path[::-1]

        print("\n", "*" * 100)
        print("Path:", " -> ".join(map(str, path)))
        print("Total flow on this path", subflow)
        print("Cost per flow:", distance[t])
        print("*" * 100, "\n")
        # cập nhật giá trị flow và cost
        flow += subflow
        cost_ += subflow * distance[t]
        cur = t
        # cập nhật lại giá trị flow cần truyền đi
        while cur != s:
            capacity[parent[cur]][cur] -= subflow
            capacity[cur][parent[cur]] += subflow
            cur = parent[cur]

    if flow < K:
        return -1
    else:
        return cost_


def main():
    start_time = time()
    # thời gian bắt đầu chương trình
    G = grid_2d_graph(5, 10)  # tạo đồ thị 2D

    # thêm các giá trị random vào cost và capacity của cạnh từ u -> v
    for u, v in G.edges():
        G.edges[u, v]["cost"] = random.randint(1, 100)
        G.edges[u, v]["capacity"] = random.randint(1000, 2000)

    mapping = {node: i for i, node in enumerate(G.nodes())}
    G = relabel_nodes(G, mapping)

    pos = {i: (i % 10, 4 - i // 10) for i in range(50)}

    source_node = 0
    sink_node = 49
    node_colors = [
        "red" if node == source_node else "green" if node == sink_node else "skyblue"
        for node in G.nodes()
    ]

    # Vẽ đồ thị
    draw(G, pos, with_labels=True, node_size=500, node_color=node_colors, font_size=8)

    edge_labels = {
        (u, v): f"{G.edges[u, v]['cost']} | {G.edges[u, v]['capacity']}"
        for u, v in G.edges()
    }
    draw_networkx_edge_labels(
        G, pos, edge_labels=edge_labels, font_color="green", rotate=False
    )

    edges = []
    for u, v, attr in G.edges(data=True):
        edges.append(Edge(u, v, attr["capacity"], attr["cost"]))
        edges.append(Edge(v, u, 0, -attr["cost"]))
    # Xác định các node source, sink và tổng flow từ source
    s = 0
    t = 49
    K = 1500
    N = len(G.nodes())
    minCost = min_cost_flow(N, edges, K, s, t)
    print("Minimum cost:", minCost)

    end_time = time()
    elapsed_time = (end_time - start_time) * 1000
    print(f"elapsed time: {elapsed_time} ms")

    plt.show()


if __name__ == "__main__":
    main()
