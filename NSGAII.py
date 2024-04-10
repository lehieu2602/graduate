import concurrent.futures
import copy
import math
import random
from itertools import islice

global network
global SFCs
global paths
Pt = []
d_max = 0


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
        re = 0
        for i in range(len(sol)):
            cnt = 1
            if sol[i] != -1:
                idx = servers[i // d_max].vnf_possible.index(sol[i])
                re += servers[i // d_max].vnf_cost[idx]

            if cnt % d_max == 0:
                re += servers[i // d_max].cost
            cnt += 1
        f1_re.append(re)
    return f1_re


def sub_f2(population, SFCs, last_loop):
    f2_re = []
    clone_network = copy.deepcopy(network)
    for sol in population:
        print(sol)
        re = 0
        tmp = []
        invalid_sol = False

        for i in range(len(sol)):
            if sol[i] != -1:
                clone_network.N_server[i // d_max].vnf_used.append(sol[i])
        clone_network.create_networkx_expand()

        for sfc in SFCs.sfc_set:
            path, delay = clone_network.find_most_efficient_path(sfc)
            if path is None:
                invalid_sol = True
            re += delay
            if last_loop:
                tmp.append(path)
        if not invalid_sol:
            if last_loop:
                paths.append(tmp)
        f2_re.append(re)

        for server in clone_network.N_server:
            server.vnf_used = []
    return f2_re


def f2(servers, population, SFCs, last_loop=False):  # hàm mục tiêu delay
    f2_re = []
    global paths
    paths = []

    idx = 0

    l_chunk = list(chunk_list(population, 25))
    num_threads = len(l_chunk)

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = []
        for chunk in l_chunk:
            print(f"idx:{idx}")
            future = executor.submit(sub_f2, chunk, SFCs, last_loop)
            futures.append(future)
            idx += len(chunk)

        for future in futures:
            f2_re.extend(future.result())
    return f2_re


def generate_population(servers, p_size):  # sinh ra quần thể Pt lúc ban đầu
    temp_set = set()
    d_s_max = max(servers, key=lambda server: server.num_vnfs_limit)
    global d_max
    d_max = d_s_max.num_vnfs_limit

    for _ in range(p_size):
        invid = []  # cách đặt vnf mỗi server
        for i_server in range(len(servers)):
            tmp = []
            j = 0
            l_vnf = copy.deepcopy(servers[i_server].vnf_possible)
            while j < servers[i_server].num_vnfs_limit:
                if random.randint(1, 4) > 1:
                    vnf_choice = random.choice(l_vnf)
                    tmp.append(vnf_choice)
                else:
                    tmp.append(-1)
                j += 1

            while len(tmp) < d_max:
                tmp.append(-1)

            invid.extend(tmp)
        temp_set.add(tuple(invid))

    global Pt
    Pt = list(temp_set)


def mutation(solution):
    mutation_prob = random.random()
    if mutation_prob < 0.1:
        idx1 = random.randint(0, len(solution) - 1)
        idx2 = random.randint(0, len(solution) - 1)
        solution[idx1], solution[idx2] = solution[idx2], solution[idx1]
    return solution


def crossover(solution1, solution2):
    points_crossover = len(solution1) // 2

    child1 = copy.deepcopy(solution1)
    child2 = copy.deepcopy(solution2)

    for _ in range(points_crossover):
        random_idx = random.randint(0, len(solution1) - 1)
        child1[random_idx] = solution2[random_idx]

        random_idx = random.randint(0, len(solution1) - 1)
        child2[random_idx] = solution1[random_idx]

    return child1, child2


def fast_non_dominated_sort(values1, values2):
    c = [0 for _ in range(len(values1))]  # số cá thể trội hơn cá thể p
    F1 = []  # gen 1
    front_perato = []  # các đường perato
    S = [[] for _ in range(len(values1))]  # tập các cá thể bị trội

    for i in range(len(values1)):
        for j in range(len(values1)):
            if j == i:
                continue
            if (values1[i] <= values1[j] and values2[i] <= values2[j]) and (
                    values1[i] < values1[j] or values2[i] < values2[j]):  # so sánh xem i dominate j không
                if j not in S[i]:
                    S[i].append(j)
            elif (values1[i] >= values1[j] and values2[i] >= values2[j]) and (
                    values1[i] > values1[j] or values2[i] > values2[j]):  # so sánh xem i bị dominate bởi j không
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


def sort_by_values(front, values):
    sorted_list = []
    while len(sorted_list) != len(front):
        if values.index(min(values)) in front:
            sorted_list.append(values.index(min(values)))
        values[values.index(min(values))] = math.inf
    return sorted_list


def crowding_distance(value1, value2, front_perato):
    distance = [0 for _ in range(len(front_perato))]
    distance[0] = math.inf
    distance[len(distance) - 1] = math.inf

    sorted_idx_f1 = sort_by_values(front_perato, copy.deepcopy(value1))
    sorted_idx_f2 = sort_by_values(front_perato, copy.deepcopy(value2))

    for node in range(1, len(front_perato) - 1):
        distance[node] = value1[sorted_idx_f1[node + 1]] - value1[sorted_idx_f1[node - 1]] / (max(value1) - min(value1))
        distance[node] += value2[sorted_idx_f2[node + 1]] - value2[sorted_idx_f2[node - 1]] / (
                max(value2) - min(value2))

    return distance


def get_better_solution(num_of_solution, front, crowding_distance_values, population):
    solutions = []
    i = 0
    while len(solutions) <= num_of_solution and i < len(front[0]):
        solutions.append(population[front[0][i]])
        i += 1

    if len(solutions) <= num_of_solution:
        rank = 1
        while len(solutions) < num_of_solution:
            for point in front[rank]:
                if len(solutions) == num_of_solution:
                    break
                idx = crowding_distance_values[rank].index(max(crowding_distance_values[rank]))

                solutions.append(population[front[rank][idx]])

                crowding_distance_values[rank][idx] = -math.inf
            rank += 1
        return solutions


def main(p_size, num_loop, birth_rate):
    global Pt
    servers = network.N_server
    generate_population(servers, p_size)  # sinh ra cách phương pháp đi
    print(Pt)
    f1_re = f1(servers, Pt)
    f2_re = f2(servers, Pt, SFCs)
    print(f2_re)

    front_perato = fast_non_dominated_sort(f1_re, f2_re)

    # create offspring
    gen_max = num_loop

    gen_num = 0
    while gen_num <= gen_max:
        num_offspring = int(birth_rate * len(Pt) / 100)
        Qt = set()
        for i in range(num_offspring // 2):
            random_front = random.randint(0, len(front_perato) - 1)
            random_sol1 = Pt[front_perato[random_front][random.randint(0, len(front_perato[random_front]) - 1)]]
            random_front = random.randint(0, len(front_perato) - 1)
            random_sol2 = Pt[front_perato[random_front][random.randint(0, len(front_perato[random_front]) - 1)]]

            child1, child2 = crossover(list(random_sol1), list(random_sol2))
            child_mutate1 = mutation(child1)
            child_mutate2 = mutation(child2)

            Qt.add(tuple(child_mutate1))
            Qt.add(tuple(child_mutate2))

        Rt = []  # combine Pt and Qt
        Rt.extend(Pt)
        Rt.extend(list(Qt))

        f1_re = f1(servers, Rt)
        f2_re = f2(servers, Rt, SFCs)

        front_perato = fast_non_dominated_sort(f1_re, f2_re)
        crowding_distance_values = []

        for i in range(0, len(front_perato)):
            crowding_distance_values.append(
                crowding_distance(copy.deepcopy(f1_re), copy.deepcopy(f2_re), front_perato[i]))

        num_of_solution = p_size
        Pt = get_better_solution(num_of_solution, copy.deepcopy(front_perato),
                                 copy.deepcopy(crowding_distance_values), Rt)

        f1_re = f1(servers, Pt)
        re_path = []
        if gen_num == gen_max:
            re_path, f2_re = f2(servers, Pt, SFCs, True)
        else:
            f2_re = f2(servers, Pt, SFCs)

        front_perato = fast_non_dominated_sort(f1_re, f2_re)
        print(f"Done gen {gen_num}")
        gen_num += 1

    print(f"Best solutions after {gen_num}:")
    print(front_perato[0])
