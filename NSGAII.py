import concurrent.futures
import copy
import json
import logging
import math
import os
import random
from itertools import islice
import heapq
import matplotlib.pyplot as plt
import numpy as np

global network
global SFCs
global paths
Pt = []
d_max = 0

logging.basicConfig(filename='app.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def chunk_list(arr_range, arr_size):
    arr_range = iter(arr_range)
    return iter(lambda: tuple(islice(arr_range, arr_size)), ())


def set_value_network(nw):
    global network
    network = nw


def set_SFCs(sfcs):
    global SFCs
    SFCs = sfcs


def f1(servers, population):  # hàm mục tiêu cost
    f1_re = []

    for sol in population:
        cost_server = 0
        cost_vnf = 0
        cnt = 1
        cnt_empty_slot = 0
        for i in range(len(sol)):
            if sol[i] != -1:
                idx = servers[i // d_max].vnf_possible.index(sol[i])
                cost_vnf += servers[i // d_max].vnf_cost[idx]
            else:
                cnt_empty_slot += 1

            if cnt % d_max == 0:
                if cnt_empty_slot > 0:
                    cost_server += servers[i // d_max].cost
                cnt_empty_slot = 0

            cnt += 1
        re = ((cost_server / network.sum_cost_servers) + (cost_vnf / np.sum(network.cost_vnfs))) / 2
        f1_re.append(re)

    return f1_re


def get_weight(source, destination):
    s = int(source.split("_")[0])
    d = int(destination.split("_")[0])
    if s == d:
        return network.N[d].delay

    try:
        id_link = f"{s}-{d}"
        edge_weight = network.L[id_link].delay
        try:
            server_weight = network.N_server[d].delay
        except:
            server_weight = 0
        return edge_weight + server_weight
    except:
        id_link = f"{d}-{s}"
        edge_weight = network.L[id_link].delay
        try:
            server_weight = network.N_server[d].delay
        except:
            server_weight = 0
        return edge_weight + server_weight


def get_weight_path(path):
    re = 0
    for i in range(len(path) - 1):
        s = path[i]
        d = path[i + 1]
        if s == d:
            re += network.N[d].delay
            continue
        server_weight = 0

        try:
            try:
                id_link = f"{s}-{d}"
                edge_weight = network.L[id_link].delay
                if i < len(path) - 2:
                    try:
                        server_weight = network.N_server[d].delay
                    except:
                        server_weight = 0
                re += edge_weight + server_weight
            except:
                id_link = f"{d}-{s}"
                edge_weight = network.L[id_link].delay
                if i < len(path) - 2:
                    try:
                        server_weight = network.N_server[d].delay
                    except:
                        server_weight = 0
                re += edge_weight + server_weight
        except Exception as e:
            logging.error(f'Path: {path} - Network: {network.name} - SFC: {SFCs.name} - Exception: {e}')
    return re


def dijkstra(start, sfc, clone_network, des=None):
    clone_network1 = copy.deepcopy(clone_network)
    dist = {node: float('inf') for node in clone_network1.nx_network.nodes}
    dist[start] = 0

    pq = [(0, start)]

    visited = {node: False for node in clone_network1.nx_network.nodes}

    # Dùng prev_node để lưu trữ đỉnh trước của mỗi đỉnh
    prev_node = {}
    same_node = 0
    while pq:
        # Lấy đỉnh có khoảng cách ngắn nhất từ start
        current_dist, current_node = heapq.heappop(pq)

        # Kiểm tra xem có phải đỉnh đích không
        visited[current_node] = True

        network_node = clone_network1.N[current_node]

        to_end = False
        if network_node.type == 1 and sfc.vnf_demand:
            to_end = sfc.vnf_demand[0] in network_node.vnf_used
        elif des is not None:
            to_end = current_node == des

        if to_end:
            path = []
            node = current_node
            while node != start:
                path.append(node)
                node = prev_node.get(node)
            path.append(start)
            path.reverse()
            if same_node == 0:  # nếu server đó có luôn vnf demand tiếp theo thì sẽ lấy node đó
                return path
            else:  # không chứa vnf demand tiếp theo thì bỏ qua
                return path[1:]

        same_node += 1

        # Duyệt qua các đỉnh kề của đỉnh hiện tại
        for adj_node in clone_network1.nx_network.adj[current_node]:
            if not visited[adj_node]:
                distance = current_dist + get_weight(str(current_node), str(adj_node))
                # Nếu khoảng cách mới tốt hơn khoảng cách hiện tại
                if distance < dist[adj_node]:
                    prev_node[adj_node] = current_node
                    network_node = clone_network1.N[adj_node]
                    if network_node.cpu_available == -1 and network_node.mem_available == -1:
                        dist[adj_node] = distance
                        heapq.heappush(pq, (distance, adj_node))

    return []


def find_shortest_path(sfc, clone_network):
    clone_network1 = copy.deepcopy(clone_network)
    clone_sfc = copy.deepcopy(sfc)
    path = [sfc.source]
    source = sfc.source
    valid_path = True

    while True:
        if clone_sfc.vnf_demand:
            sub_path = dijkstra(source, clone_sfc, clone_network1)

            if sub_path:
                path.extend(sub_path)
                source = sub_path[len(sub_path) - 1]
                clone_sfc.vnf_demand.pop(0)
            else:
                valid_path = False
                break
        else:
            sub_path = dijkstra(source, clone_sfc, clone_network1, clone_sfc.destination)

            if sub_path:
                path.extend(sub_path)
            else:
                valid_path = False
            break
    if valid_path:
        return path, get_weight_path(path)
    else:
        return [], 10000000


def find_path_sfc(sfc, clone_network):
    path, delay = find_shortest_path(sfc, clone_network)
    return path, delay


def find_path_and_delay(population, SFCs):
    f2_re = []
    clone_network = copy.deepcopy(network)
    valid = False
    for sol in population:
        valid = False
        re = 0
        for sfc in SFCs.sfc_set:  # check sol có thoả mãn vnf yêu cầu của từng sfc
            valid = all(item in sol for item in sfc.vnf_demand)

            if not valid:
                break

        if not valid:
            continue

        for i in range(len(sol)):
            if sol[i] != -1:
                clone_network.N_server[i // d_max].vnf_used.append(sol[i])

        with concurrent.futures.ThreadPoolExecutor(max_workers=len(SFCs.sfc_set)) as executor:
            futures = []
            for sfc in SFCs.sfc_set:
                future = executor.submit(find_path_sfc, sfc, clone_network)
                futures.append(future)

            if not all(future.result()[0] for future in futures):
                valid = False
            for future in futures:
                re += future.result()[1]

        [server.vnf_used.clear() for server in clone_network.N_server]

        re /= (network.total_delay_link + network.total_delay_server) * SFCs.num_sfc
        f2_re.append(re)

    return valid, f2_re


def f2(population, SFCs, last_loop=False):  # hàm mục tiêu delay
    f2_re = []
    global paths
    paths = []

    idx = 0

    l_chunk = list(chunk_list(population, 1))
    num_threads = len(l_chunk)

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = []
        for chunk in l_chunk:
            print(f"idx:{idx}")
            future = executor.submit(find_path_and_delay, chunk, SFCs)
            futures.append(future)
            idx += len(chunk)

        for future in futures:
            f2_re.extend(future.result()[1])
    return f2_re


def generate_population(servers, p_size):  # sinh ra quần thể Pt lúc ban đầu
    temp_set = set()
    d_s_max = max(servers, key=lambda server: server.num_vnfs_limit)
    global d_max
    d_max = d_s_max.num_vnfs_limit
    f2_re = []
    cnt = 1
    while len(temp_set) < p_size:
        invid = []  # cách đặt vnf mỗi server
        l_vnf = []
        for sfc in SFCs.sfc_set:
            l_vnf.extend(sfc.vnf_demand)
        for i_server in range(len(servers)):
            tmp = []
            j = 0
            while j < servers[i_server].num_vnfs_limit:
                if l_vnf:
                    if random.randint(1, 4) > 1:
                        vnf_choice = random.choice(l_vnf)
                        tmp.append(vnf_choice)
                        l_vnf.remove(vnf_choice)
                    else:
                        tmp.append(-1)
                else:
                    for sfc in SFCs.sfc_set:
                        l_vnf.extend(sfc.vnf_demand)
                j += 1

            while len(tmp) < d_max:
                tmp.append(-1)

            invid.extend(tmp)
        valid_path, delay = find_path_and_delay([invid], SFCs)
        # valid_path, delay = find_path_and_delay([[4, 1, 1, 9, 3, 8, 3, 3, 1, 0, 3, 6, 8, 3, 8, 4, 3, 0, 9, 8]], SFCs)
        if valid_path:
            print(cnt)
            cnt += 1
            f2_re.extend(delay)
            temp_set.add(tuple(invid))

    return f2_re, list(temp_set)


def mutation(solution):
    mutation_prob = random.random()
    if mutation_prob < 0.1:
        idx1 = random.randint(0, len(solution) - 1)
        idx2 = random.randint(0, len(solution) - 1)
        solution[idx1], solution[idx2] = solution[idx2], solution[idx1]
    return solution


def crossover(solution1, solution2):
    l_vnf = set()
    for sfc in SFCs.sfc_set:
        l_vnf.update(sfc.vnf_demand)

    check = [False] * len(solution1)

    for i in l_vnf:
        idx = solution1.index(i)
        check[idx] = True

    child = copy.deepcopy(solution1)

    for i in range(len(solution1)):
        if not check[i]:
            child[i] = solution2[i]

    return child


def fast_non_dominated_sort(values1, values2):
    c = [0 for _ in range(len(values1))]  # số cá thể trội hơn cá thể p
    F1 = []  # gen 1
    front_perato = []  # các đường perato
    S = [[] for _ in range(len(values1))]  # tập các cá thể bị trội

    for i in range(len(values1)):
        for j in range(len(values1)):
            if j == i:
                continue

            if (values1[i] < values1[j] and values2[i] < values2[j]) or (
                    values1[i] <= values1[j] and values2[i] < values2[j]) or (
                    values1[i] < values1[j] and values2[i] <= values2[j]):  # so sánh xem i dominate j không
                if j not in S[i]:
                    S[i].append(j)
            elif (values1[i] > values1[j] and values2[i] > values2[j]) or (
                    values1[i] >= values1[j] and values2[i] > values2[j]) or (
                    values1[i] > values1[j] and values2[i] >= values2[j]):  # so sánh xem i bị dominate bởi j không
                c[i] += 1

        if c[i] == 0:
            if i not in F1:
                F1.append(i)
    front_perato.append(F1)

    i = 0
    while front_perato[i]:
        f_gen = []
        for invid in front_perato[i]:  # duyệt các cá thể trong cùng 1 rank
            for dominate_invid in S[invid]:  # duyệt tập bị bị p trội
                c[int(dominate_invid)] -= 1

                if c[int(dominate_invid)] == 0:
                    if dominate_invid not in f_gen:
                        f_gen.append(dominate_invid)

        front_perato.append(f_gen)
        i += 1
    del front_perato[len(front_perato) - 1]

    return front_perato


def sort_by_values(front, value1, value2):
    points = [(value1[i], value2[i], i) for i in front]

    sorted_list = sorted(points, key=lambda point: (point[0], point[1]))

    return sorted_list


def crowding_distance(value1, value2, front_perato):
    distance = [0 for _ in range(len(front_perato))]

    sorted_point = sort_by_values(front_perato, value1, value2)

    distance[front_perato.index(sorted_point[0][2])] = math.inf
    distance[front_perato.index(sorted_point[len(distance) - 1][2])] = math.inf

    min_point = sorted_point[0]
    max_point = sorted_point[-1]

    for i, point in enumerate(sorted_point[1:len(sorted_point) - 1]):
        i += 1  # để cập nhật đúng vị trí từ 1 theo danh sách duyệt từ vị trí 1
        if point[:2] == min_point[:2] or point[:2] == max_point[:2]:
            distance[front_perato.index(sorted_point[i][2])] = math.inf

    sorted_idx = [idx[2] for idx in sorted_point]

    for node in sorted_idx[1:len(sorted_idx) - 1]:
        idx = front_perato.index(node)  # vị trí trong distance
        i_sorted_idx = sorted_idx.index(node)  # vị trí trong sorted_idx
        distance[idx] = abs(value1[sorted_idx[i_sorted_idx + 1]] - value1[sorted_idx[i_sorted_idx - 1]] / (
                max(value1) - min(value1)))
    for node in sorted_idx[1:len(sorted_idx) - 1]:
        idx = front_perato.index(node)
        i_sorted_idx = sorted_idx.index(node)
        distance[idx] += abs(value2[sorted_idx[i_sorted_idx + 1]] - value2[sorted_idx[i_sorted_idx - 1]] / (
                max(value2) - min(value2)))

    return distance


def get_better_solution(num_of_solution, front, crowding_distance_values, population):
    solutions = []
    i = 0
    while len(solutions) <= num_of_solution and i < len(front[0]):
        solutions.append(population[front[0][i]])
        i += 1

    rank = 1
    while len(solutions) < num_of_solution:
        for _ in front[rank]:
            if len(solutions) == num_of_solution:
                break
            idx = crowding_distance_values[rank].index(max(crowding_distance_values[rank]))

            solutions.append(population[front[rank][idx]])

            crowding_distance_values[rank][idx] = -math.inf
        rank += 1
    return solutions


def visualize_perato(fronts, f1, f2):
    plt.clf()
    colors = plt.cm.tab10(range(len(fronts)))

    for i, front in enumerate(fronts):
        front_f1 = [f1[idx] for idx in front]
        front_f2 = [f2[idx] for idx in front]
        plt.scatter(front_f1, front_f2, color=colors[i])

    plt.xlabel('Cost')
    plt.ylabel('Delay')
    plt.savefig(f"./experiments/images/{network.name}_{SFCs.name}.png")


def main(p_size, num_loop, birth_rate):
    global Pt
    servers = network.N_server
    f2_re, Pt = generate_population(servers, p_size)  # sinh ra cách phương pháp đi
    print(Pt)
    print(f2_re)
    f1_re = f1(servers, Pt)
    result = {"network": network.name, "request": SFCs.name, "p_size": p_size}

    front_perato = fast_non_dominated_sort(f1_re, f2_re)

    # create offspring
    gen_max = num_loop

    gen_num = 0
    while gen_num <= gen_max:
        num_offspring = int(birth_rate * len(Pt) / 100)
        Qt = set()
        for i in range(num_offspring // 2):
            random_front1 = random.randint(0, len(front_perato) - 1)
            idx_sol1 = random.randint(0, len(front_perato[random_front1]) - 1)
            random_sol1 = Pt[front_perato[random_front1][idx_sol1]]

            if len(front_perato) > 1:
                random_front2 = random.choice([i for i in range(len(front_perato)) if i != random_front1])
                random_sol2 = Pt[front_perato[random_front2][random.randint(0, len(front_perato[random_front2]) - 1)]]
            else:
                random_front2 = random_front1
                idx_sol2 = random.choice([i for i in range(len(front_perato[random_front2])) if i != idx_sol1])
                random_sol2 = Pt[front_perato[random_front2][idx_sol2]]
            # random_front2 = random.randint(0, len(front_perato) - 1)

            child = crossover(list(random_sol1), list(random_sol2))
            child_mutate1 = mutation(child)

            Qt.add(tuple(child_mutate1))

        Rt = []  # combine Pt and Qt
        Rt.extend(Pt)
        Rt.extend(list(Qt))

        f1_re = f1(servers, Rt)
        f2_re = f2(Rt, SFCs)

        front_perato = fast_non_dominated_sort(f1_re, f2_re)
        print(f"front1: {front_perato}")

        crowding_distance_values = []

        for i in range(0, len(front_perato)):
            crowding_distance_values.append(
                crowding_distance(copy.deepcopy(f1_re), copy.deepcopy(f2_re), front_perato[i]))

        num_of_solution = p_size
        Pt = get_better_solution(num_of_solution, copy.deepcopy(front_perato),
                                 copy.deepcopy(crowding_distance_values), Rt)

        f1_re = f1(servers, Pt)
        f2_re = f2(Pt, SFCs)
        print(f"f2:{f2_re}")
        front_perato = fast_non_dominated_sort(f1_re, f2_re)
        print(f"front2: {front_perato}")
        print(f"Done gen {gen_num}")
        gen_num += 1

    result.update({"front_perato": front_perato[0], "cost": f1_re, "delay": f2_re, "front": front_perato})
    visualize_perato(front_perato, f1_re, f2_re)
    out_path = "./experiments"
    os.makedirs(out_path, exist_ok=True)
    file_name = f"{result['network']}_{result['request']}"
    with open(os.path.join(out_path, file_name), 'a') as f:
        json.dump(result, f)
    print(front_perato[0])
