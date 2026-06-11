import torch
import numpy as np


DEMAND_SCALER_MAP = {
    20: 30,
    30: 35,
    40: 38,
    50: 40,
    100: 50,
    200: 70,
    500: 130,
    1000: 230,
}


def get_demand_scaler(problem_size):
    if problem_size in DEMAND_SCALER_MAP:
        return DEMAND_SCALER_MAP[problem_size]
    sizes = sorted(DEMAND_SCALER_MAP.keys())
    if problem_size < sizes[0]:
        return DEMAND_SCALER_MAP[sizes[0]]
    if problem_size > sizes[-1]:
        return DEMAND_SCALER_MAP[sizes[-1]]
    for i in range(len(sizes) - 1):
        if sizes[i] < problem_size < sizes[i+1]:
            ratio = (problem_size - sizes[i]) / (sizes[i+1] - sizes[i])
            scaler = DEMAND_SCALER_MAP[sizes[i]] + \
                ratio * (DEMAND_SCALER_MAP[sizes[i+1]] - DEMAND_SCALER_MAP[sizes[i]])
            return scaler
    raise ValueError(f"Cannot determine demand_scaler for problem_size={problem_size}")


def get_random_problems_mixed(batch_size, problem_size, problem_type):
    """
    ★ 返回值新增第9个: max_stops_per_route
    """
    depot_xy = torch.rand(size=(batch_size, 1, 2))
    node_xy = torch.rand(size=(batch_size, problem_size, 2))

    demand_scaler = get_demand_scaler(problem_size)
    node_demand = torch.randint(1, 10, size=(batch_size, problem_size)) / float(demand_scaler)

    node_serviceTime = torch.zeros(size=(batch_size, problem_size))
    node_lengthTW = torch.zeros(size=(batch_size, problem_size))
    node_earlyTW = torch.zeros(size=(batch_size, problem_size))
    node_lateTW = node_earlyTW + node_lengthTW

    route_length_limit = torch.zeros(size=(batch_size, problem_size))
    route_open = torch.zeros(size=(batch_size, problem_size))
    max_stops_per_route = torch.zeros(size=(batch_size, problem_size))

    seed = np.random.rand()

    if ((problem_type == 'unified' and seed >= 0.2 and seed < 0.4) or 'L' in problem_type):
        route_length_limit = 3.0 * torch.ones(size=(batch_size, problem_size))

    if ((problem_type == 'unified' and seed >= 0.4 and seed < 0.6) or 'TW' in problem_type):
        node_serviceTime = torch.rand(size=(batch_size, problem_size)) * 0.05 + 0.15
        node_lengthTW = torch.rand(size=(batch_size, problem_size)) * 0.05 + 0.15
        d0i = ((node_xy - depot_xy.expand(size=(batch_size, problem_size, 2)))**2).sum(2).sqrt()
        ei = torch.rand(size=(batch_size, problem_size)).mul(
            (torch.div((4.6 * torch.ones(size=(batch_size, problem_size)) - node_serviceTime - node_lengthTW), d0i) - 1) - 1) + 1
        node_earlyTW = ei.mul(d0i)
        node_lateTW = node_earlyTW + node_lengthTW

    if ((problem_type == 'unified' and seed >= 0.6 and seed <= 0.8) or 'O' in problem_type):
        node_demand = torch.randint(1, 10, size=(batch_size, problem_size)) / float(demand_scaler)
        route_open = torch.ones(size=(batch_size, problem_size))

    if ((problem_type == 'unified' and seed >= 0.8) or 'B' in problem_type):
        node_demand = torch.randint(1, 10, size=(batch_size, problem_size)) / float(demand_scaler)
        linehaul = int(0.8 * problem_size)
        node_demand[:, linehaul:] = -node_demand[:, linehaul:]

    if 'K' in problem_type:
        K_val = max(3, problem_size // 5)
        max_stops_per_route = float(K_val) * torch.ones(size=(batch_size, problem_size))

    return (depot_xy, node_xy, node_demand, node_earlyTW, node_lateTW,
            node_serviceTime, route_open, route_length_limit,
            max_stops_per_route)


def augment_xy_data_by_8_fold(xy_data):
    x = xy_data[:, :, [0]]
    y = xy_data[:, :, [1]]
    dat1 = torch.cat((x, y), dim=2)
    dat2 = torch.cat((1 - x, y), dim=2)
    dat3 = torch.cat((x, 1 - y), dim=2)
    dat4 = torch.cat((1 - x, 1 - y), dim=2)
    dat5 = torch.cat((y, x), dim=2)
    dat6 = torch.cat((1 - y, x), dim=2)
    dat7 = torch.cat((y, 1 - x), dim=2)
    dat8 = torch.cat((1 - y, 1 - x), dim=2)
    aug_xy_data = torch.cat((dat1, dat2, dat3, dat4, dat5, dat6, dat7, dat8), dim=0)
    return aug_xy_data
