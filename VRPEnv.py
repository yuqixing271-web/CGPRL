from dataclasses import dataclass
import torch

from VRProblemDef import get_random_problems_mixed, augment_xy_data_by_8_fold


@dataclass
class Reset_State:
    depot_xy: torch.Tensor = None
    node_xy: torch.Tensor = None
    node_demand: torch.Tensor = None
    node_earlyTW: torch.Tensor = None
    node_lateTW: torch.Tensor = None


@dataclass
class Step_State:
    BATCH_IDX: torch.Tensor = None
    POMO_IDX: torch.Tensor = None
    selected_count: int = None
    load: torch.Tensor = None
    time: torch.Tensor = None
    route_open: torch.Tensor = None
    length: torch.Tensor = None
    current_node: torch.Tensor = None
    ninf_mask: torch.Tensor = None
    finished: torch.Tensor = None


class VRPEnv:
    def __init__(self, **env_params):
        self.env_params = env_params
        self.problem_size = env_params['problem_size']
        self.pomo_size = env_params['pomo_size']
        self.problem_type = env_params['problem_type']

        self.FLAG__use_saved_problems = False
        self.saved_depot_xy = None
        self.saved_node_xy = None
        self.saved_node_demand = None
        self.saved_index = None

        self.batch_size = None
        self.BATCH_IDX = None
        self.POMO_IDX = None
        self.depot_node_xy = None
        self.depot_node_demand = None
        self.depot_node_earlyTW = None
        self.depot_node_lateTW = None
        self.depot_node_servicetime = None
        self.length = None

        self.attribute_c = False
        self.attribute_tw = False
        self.attribute_o = False
        self.attribute_b = False
        self.attribute_l = False
        self.attribute_k = False              

        self.max_stops_per_route = None       
        self.stop_count = None                

        self.selected_count = None
        self.current_node = None
        self.selected_node_list = None

        self.at_the_depot = None
        self.load = None
        self.time = None
        self.route_open = None
        self.visited_ninf_flag = None
        self.ninf_mask = None
        self.finished = None

        self.reset_state = Reset_State()
        self.step_state = Step_State()

    def set_problem_size(self, problem_size, pomo_size=None):
        self.problem_size = problem_size
        self.pomo_size = pomo_size if pomo_size is not None else problem_size
        self.BATCH_IDX = None
        self.POMO_IDX = None

    def get_problem_size(self):
        return self.problem_size, self.pomo_size

    def use_saved_problems(self, filename, device):
        self.FLAG__use_saved_problems = True
        loaded_dict = torch.load(filename, map_location=device)
        self.saved_depot_xy = loaded_dict['depot_xy']
        self.saved_node_xy = loaded_dict['node_xy']
        self.saved_node_demand = loaded_dict['node_demand']
        self.saved_node_earlyTW = loaded_dict['node_earlyTW']
        self.saved_node_lateTW = loaded_dict['node_lateTW']
        self.saved_node_servicetime = loaded_dict['node_serviceTime']
        self.saved_route_open = loaded_dict['route_open']
        self.saved_route_length = loaded_dict['route_length_limit']
        # ★ 兼容旧数据文件 (可能没有 max_stops_per_route)
        self.saved_max_stops = loaded_dict.get('max_stops_per_route', None)
        self.saved_index = 0

    def load_problems(self, batch_size, aug_factor=1):
        self.batch_size = batch_size

        if not self.FLAG__use_saved_problems:
            # ★ 接收第9个返回值
            (depot_xy, node_xy, node_demand, node_earlyTW, node_lateTW,
             node_servicetime, route_open, route_length_limit,
             max_stops_per_route) = \
                get_random_problems_mixed(batch_size, self.problem_size, self.problem_type)
        else:
            depot_xy = self.saved_depot_xy[self.saved_index:self.saved_index+batch_size]
            node_xy = self.saved_node_xy[self.saved_index:self.saved_index+batch_size]
            node_demand = self.saved_node_demand[self.saved_index:self.saved_index+batch_size]
            node_earlyTW = self.saved_node_earlyTW[self.saved_index:self.saved_index+batch_size]
            node_lateTW = self.saved_node_lateTW[self.saved_index:self.saved_index+batch_size]
            node_servicetime = self.saved_node_servicetime[self.saved_index:self.saved_index+batch_size]
            route_open = self.saved_route_open[self.saved_index:self.saved_index+batch_size]
            route_length_limit = self.saved_route_length[self.saved_index:self.saved_index+batch_size]
            # ★ 兼容: 旧文件可能没有这个字段
            if self.saved_max_stops is not None:
                max_stops_per_route = self.saved_max_stops[self.saved_index:self.saved_index+batch_size]
            else:
                max_stops_per_route = torch.zeros(size=(batch_size, self.problem_size))
            self.saved_index += batch_size

        if aug_factor > 1:
            if aug_factor == 8:
                self.batch_size = self.batch_size * 8
                depot_xy = augment_xy_data_by_8_fold(depot_xy)
                node_xy = augment_xy_data_by_8_fold(node_xy)
                node_demand = node_demand.repeat(8, 1)
                node_earlyTW = node_earlyTW.repeat(8, 1)
                node_lateTW = node_lateTW.repeat(8, 1)
                node_servicetime = node_servicetime.repeat(8, 1)
                route_open = route_open.repeat(8, 1)
                route_length_limit = route_length_limit.repeat(8, 1)
                max_stops_per_route = max_stops_per_route.repeat(8, 1)  # ★ 新增
            else:
                raise NotImplementedError

        self.route_open = route_open
        self.length = route_length_limit
        self.max_stops_per_route = max_stops_per_route  # ★ 新增

        self.depot_node_xy = torch.cat((depot_xy, node_xy), dim=1)
        depot_demand = torch.zeros(size=(self.batch_size, 1))
        self.depot_node_demand = torch.cat((depot_demand, node_demand), dim=1)

        depot_earlyTW = torch.zeros(size=(self.batch_size, 1))
        depot_lateTW = 4.6 * torch.ones(size=(self.batch_size, 1))
        depot_servicetime = torch.zeros(size=(self.batch_size, 1))
        self.depot_node_earlyTW = torch.cat((depot_earlyTW, node_earlyTW), dim=1)
        self.depot_node_lateTW = torch.cat((depot_lateTW, node_lateTW), dim=1)
        self.depot_node_servicetime = torch.cat((depot_servicetime, node_servicetime), dim=1)

        self.BATCH_IDX = torch.arange(self.batch_size)[:, None].expand(self.batch_size, self.pomo_size)
        self.POMO_IDX = torch.arange(self.pomo_size)[None, :].expand(self.batch_size, self.pomo_size)

        self.reset_state.depot_xy = depot_xy
        self.reset_state.node_xy = node_xy
        self.reset_state.node_demand = node_demand
        self.reset_state.node_earlyTW = node_earlyTW
        self.reset_state.node_lateTW = node_lateTW

        self.step_state.BATCH_IDX = self.BATCH_IDX
        self.step_state.POMO_IDX = self.POMO_IDX

        self.attribute_c = True if node_demand.sum() > 0 else False
        self.attribute_tw = True if node_lateTW.sum() > 0 else False
        self.attribute_o = True if route_open.sum() > 0 else False
        self.attribute_l = True if route_length_limit.sum() > 0 else False
        self.attribute_k = True if max_stops_per_route.sum() > 0 else False  # ★ 新增

    def reset(self):
        self.selected_count = 0
        self.current_node = None
        self.selected_node_list = torch.zeros((self.batch_size, self.pomo_size, 0), dtype=torch.long)

        self.at_the_depot = torch.ones(size=(self.batch_size, self.pomo_size), dtype=torch.bool)
        self.load = torch.ones(size=(self.batch_size, self.pomo_size))
        self.time = torch.zeros(size=(self.batch_size, self.pomo_size))
        self.length = 3.0 * torch.ones(size=(self.batch_size, self.pomo_size))
        self.stop_count = torch.zeros(size=(self.batch_size, self.pomo_size))  # ★ 新增
        self.visited_ninf_flag = torch.zeros(size=(self.batch_size, self.pomo_size, self.problem_size + 1))
        self.ninf_mask = torch.zeros(size=(self.batch_size, self.pomo_size, self.problem_size + 1))
        self.finished = torch.zeros(size=(self.batch_size, self.pomo_size), dtype=torch.bool)

        reward = None
        done = False
        return self.reset_state, reward, done

    def pre_step(self):
        self.step_state.selected_count = self.selected_count
        self.step_state.load = self.load
        self.step_state.current_node = self.current_node
        self.step_state.ninf_mask = self.ninf_mask
        self.step_state.finished = self.finished
        self.step_state.time = self.time
        self.step_state.route_open = self.route_open
        self.step_state.length = self.length.clone()

        reward = None
        done = False
        return self.step_state, reward, done

    def step(self, selected):
        self.selected_count += 1
        self.current_node = selected
        self.selected_node_list = torch.cat((self.selected_node_list, self.current_node[:, :, None]), dim=2)

        self.at_the_depot = (selected == 0)

        demand_list = self.depot_node_demand[:, None, :].expand(-1, self.pomo_size, -1)
        gathering_index = selected[:, :, None]
        selected_demand = demand_list.gather(dim=2, index=gathering_index).squeeze(dim=2)

        self.load -= selected_demand
        self.load[self.at_the_depot] = 1

        self.visited_ninf_flag[self.BATCH_IDX, self.POMO_IDX, selected] = float('-inf')
        self.visited_ninf_flag[:, :, 0][~self.at_the_depot] = 0

        self.ninf_mask = self.visited_ninf_flag.clone()
        round_error_epsilon = 0.000001
        demand_too_large = self.load[:, :, None] + round_error_epsilon < demand_list
        self.ninf_mask[demand_too_large] = float('-inf')

        servicetime_list = self.depot_node_servicetime[:, None, :].expand(-1, self.pomo_size, -1)
        selected_servicetime = servicetime_list.gather(dim=2, index=gathering_index).squeeze(dim=2)

        earlyTW_list = self.depot_node_earlyTW[:, None, :].expand(-1, self.pomo_size, -1)
        selected_earlyTW = earlyTW_list.gather(dim=2, index=gathering_index).squeeze(dim=2)

        xy_list = self.depot_node_xy[:, None, :, :].expand(-1, self.pomo_size, -1, -1)
        gathering_index = selected[:, :, None, None].expand(-1, -1, -1, 2)
        selected_xy = xy_list.gather(dim=2, index=gathering_index).squeeze(dim=2)

        if self.selected_node_list.size()[2] == 1:
            gathering_index_last = self.selected_node_list[:, :, -1][:, :, None, None].expand(-1, -1, -1, 2)
        else:
            gathering_index_last = self.selected_node_list[:, :, -2][:, :, None, None].expand(-1, -1, -1, 2)
        last_xy = xy_list.gather(dim=2, index=gathering_index_last).squeeze(dim=2)
        selected_time = ((selected_xy - last_xy)**2).sum(dim=2).sqrt()

        if self.attribute_tw:
            self.time = torch.max((self.time + selected_time), selected_earlyTW)
            self.time += selected_servicetime
            self.time[self.at_the_depot] = 0
            time_to_next = ((selected_xy[:, :, None, :].expand(-1, -1, self.problem_size + 1, -1) - xy_list)**2).sum(dim=3).sqrt()
            time_too_late = self.time[:, :, None] + time_to_next > self.depot_node_lateTW[:, None, :].expand(-1, self.pomo_size, -1)
            time_too_late[self.depot_node_lateTW[:, None, :].expand(-1, self.pomo_size, -1) == 0] = 0
            self.ninf_mask[time_too_late] = float('-inf')

        if self.attribute_l:
            self.step_state.length -= selected_time
            self.step_state.length[self.at_the_depot] = self.length[0][0]
            length_to_next = ((selected_xy[:, :, None, :].expand(-1, -1, self.problem_size + 1, -1) - xy_list)**2).sum(dim=3).sqrt()
            depot_xy = xy_list[:, :, 0, :]
            next_to_depot = ((depot_xy[:, :, None, :].expand(-1, -1, self.problem_size + 1, -1) - xy_list)**2).sum(dim=3).sqrt()
            if self.attribute_o:
                length_too_small = self.step_state.length[:, :, None] - round_error_epsilon < length_to_next
            else:
                length_too_small = self.step_state.length[:, :, None] - round_error_epsilon < (length_to_next + next_to_depot)
            self.ninf_mask[length_too_small] = float('-inf')
            self.ninf_mask[:, :, 0][~self.at_the_depot] = 0

        # ★★★ 新增: K 约束 (Route Stop Limit) ★★★
        if self.attribute_k:
            self.stop_count = self.stop_count + (~self.at_the_depot).float()
            self.stop_count[self.at_the_depot] = 0

            K_val = self.max_stops_per_route[:, 0]  # (B,), 每个batch的K值
            at_limit = self.stop_count >= K_val[:, None].expand_as(self.stop_count)  # (B, pomo)

            if at_limit.any():
                limit_mask = at_limit[:, :, None].expand(
                    self.batch_size, self.pomo_size, self.problem_size)
                self.ninf_mask[:, :, 1:][limit_mask] = float('-inf')

            self.ninf_mask[:, :, 0][~self.at_the_depot] = 0

        newly_finished = (self.visited_ninf_flag == float('-inf')).all(dim=2)
        self.finished = self.finished + newly_finished

        self.ninf_mask[:, :, 0][self.finished] = 0

        self.step_state.selected_count = self.selected_count
        self.step_state.load = self.load
        self.step_state.current_node = self.current_node
        self.step_state.ninf_mask = self.ninf_mask
        self.step_state.finished = self.finished

        done = self.finished.all()
        if done:
            reward = -self._get_travel_distance()
        else:
            reward = None

        return self.step_state, reward, done

    def _get_travel_distance(self):
        gathering_index = self.selected_node_list[:, :, :, None].expand(-1, -1, -1, 2)
        all_xy = self.depot_node_xy[:, None, :, :].expand(-1, self.pomo_size, -1, -1)
        ordered_seq = all_xy.gather(dim=2, index=gathering_index)
        rolled_seq = ordered_seq.roll(dims=2, shifts=-1)
        segment_lengths = ((ordered_seq - rolled_seq)**2).sum(3).sqrt()
        if self.attribute_o:
            segment_lengths[self.selected_node_list.roll(dims=2, shifts=-1) == 0] = 0
        travel_distances = segment_lengths.sum(2)
        return travel_distances

    def get_node_seq(self):
        gathering_index = self.selected_node_list[:, :, :, None].expand(-1, -1, -1, 2)
        all_xy = self.depot_node_xy[:, None, :, :].expand(-1, self.pomo_size, -1, -1)
        ordered_seq = all_xy.gather(dim=2, index=gathering_index)
        return gathering_index, ordered_seq

    def get_selected_node_list(self):
        return self.selected_node_list.clone()
