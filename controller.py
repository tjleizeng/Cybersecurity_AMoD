import numpy as np
from collections import namedtuple, deque
from random import sample
from lapsolver import solve_dense


class Platform:
    def __init__(self, travel_distance, num_zone):
        self.travel_distance = travel_distance
        self.num_zone = num_zone
        self.size = int(np.sqrt(num_zone))

    def batch_matching(self, pass_count, veh_locs):
        # return: reposition cost and matching result
        veh_schedule = []
        # solve integer program
        ## get the cost matrix
        total_cost = 0
        r_count = pass_count
        r_sum = np.sum(r_count)
        c_sum = len(veh_locs)
        if r_sum > 0 and c_sum > 0:
            tlist = [loc[-1] for loc in veh_locs]
            rlist = sum([[i]*min(c_sum,r_count[i]) for i in range(self.num_zone)], [])
            clist = [loc[2]*self.size + loc[3] for loc in veh_locs]
            costs = np.zeros((len(rlist), len(clist)))
            for i in range(len(rlist)):
                for j in range(len(clist)):
                    costs[i,j] = self.travel_distance[rlist[i], clist[j]] + tlist[j]

            if costs.shape[0] < costs.shape[1]:
                rids, cids = solve_dense(costs)
            else:
                cids, rids = solve_dense(costs.T)
            for r,c in zip(rids, cids):
                veh_schedule.append((rlist[r], veh_locs[c]))
                total_cost += costs[r,c]
        # veh_schedule is a three element tuple
        return total_cost, veh_schedule