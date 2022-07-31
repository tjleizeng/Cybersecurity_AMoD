from collections import deque
import numpy as np
from lapsolver import solve_dense

class Attacker:
    def __init__(self, travel_distance, degree, betweenness, size = 5, power = 10, frequency = 12, duration = 12, strategy = 'dummy', demand_pattern = None):
        self.size = size
        self.num_zone = size * size
        self.power = power
        self.frequency = frequency # every 
        self.duration = duration # short, mid, max
        self.strategy = strategy
        self.demand_pattern = demand_pattern
        self.travel_distance = travel_distance
        self.degree = degree
        self.betweenness = betweenness
        self.rand_int = list(np.random.randint(0,self.num_zone,size=10000))
        self.num_veh = 25
        print("Attack strategy: {} with power {} duration {} at frequency {} ".format(strategy, power, duration, frequency))

    def generate_attack_msg(self, t, supplies = None):
        if self.strategy == 'random':
            if not len(self.rand_int)>=self.power:
                self.rand_int = list(np.random.randint(0,self.num_zone,size=10000))
            return [self.rand_int.pop() for i in range(self.power)]

        # Demand based attack
        elif self.strategy == 'remote':
            p = 1 - np.sum(self.demand_pattern[t,:,:], axis = 1)/np.sum(self.demand_pattern[t,:,:])
            p /= np.sum(p)
            # print(p)
            return np.random.choice(list(range(self.num_zone)), size=self.power, p = p, replace = True)
        elif self.strategy == 'center':
            p = np.sum(self.demand_pattern[t,:,:], axis = 1)/np.sum(self.demand_pattern[t,:,:])
            # print(p)
            return np.random.choice(list(range(self.num_zone)), size=self.power, p = p, replace = True)
        elif self.strategy == 'highest':
            highest = np.argmax(np.sum(self.demand_pattern[t,:,:], axis = 1))
            return [highest] * self.power
        elif self.strategy == 'lowest':
            lowest = np.argmin(np.sum(self.demand_pattern[t,:,:], axis = 1))
            return [lowest] * self.power
        # Graph-based attack
        elif self.strategy == 'degree_high':
            candidates = np.arange(self.num_zone)[self.degree == np.max(self.degree)] # get a sample among the nodes with the highest degree
            return np.random.choice(candidates, size = self.power, replace = True)
        elif self.strategy == 'degree_low':
            candidates = np.arange(self.num_zone)[self.degree == np.min(self.degree)]
            return np.random.choice(candidates, size = self.power, replace = True)
        elif self.strategy == 'betweeness_high':
            candidates = np.arange(self.num_zone)[self.betweenness == np.max(self.betweenness)] # get a sample among the nodes with the highest betweeness
            return np.random.choice(candidates, size = self.power, replace = True)
        elif self.strategy == 'betweeness_low':
            candidates = np.arange(self.num_zone)[self.betweenness == np.min(self.betweenness)]
            return np.random.choice(candidates, size = self.power, replace = True)
        # Operation-based attack
        elif self.strategy == 'ILP':
            # as different as possible
            # Mento Carlo, find the most effective attack distribution, the objective make the sol as different as the opt one
            # generate a set of candidate attacks
            attacks = []
            for j in range(10): # 10 from random
                if not len(self.rand_int)>=self.power:
                    self.rand_int = list(np.random.randint(0,self.num_zone,size=10000))
                attacks.append([self.rand_int.pop() for i in range(self.power)])
            # some low probability choice, attack a specific node
            attacks.append([np.argmax(np.sum(self.demand_pattern[t,:,:], axis = 1))] * self.power)
            attacks.append([np.argmin(np.sum(self.demand_pattern[t,:,:], axis = 1))] * self.power)
            # generate a set of demand, vehicle distribution
            demands = []
            for j in range(10):
                demands.append(np.random.poisson(np.sum(self.demand_pattern[t,:,:], axis = 1)))#[np.random.poisson(np.sum(self.demand_pattern[t,:,:], axis = 1)) for i in range(self.duration)])

            # calculate the objective func according to equation 1
            res = []
            for attack in attacks:
                effect = 0
                for j in range(10):
                    effect+=self.evaluate_attack(attack.copy(), demands[j], supplies)
                res.append(effect)
            # for attack in attacks:
            #     effect = 0
            #     for j in range(10):
            #         effect+=self.evaluate_attack(attack.copy(), demands[j], supplies)
            #     res.append(effect)
            # find the greatest one
            # print(res)
            return attacks[np.argmax(res)]

        return []

    def evaluate_attack(self, attack, demands, supplies):
        res = 0

        # demand_agg = sum(demands).clip(max = 1)
        # demand_agg = sum([[i]*demand_agg[i] for i in range(self.num_zone)],[])

        if np.sum(demands)>0:
            # for t in range(self.duration):
            #     # print(t)
            #     cost, attack = self.batch_matching(attack, demands[t], supplies, self.duration - t)
            #     res +=  cost
            #     if len(attack) == 0:
            #         break
            # for t in range(self.duration):
            # print(t)
            cost, attack = self.batch_matching(attack, demands, supplies)
            res +=  cost
            # if len(attack) == 0:
            #     break
        return res

    def batch_matching(self, attack, pass_count, veh_locs):
        # return: reposition cost and matching result
        # veh_schedule = []
        # solve integer program
        ## get the cost matrix
        total_cost = 0
        r_count= pass_count
        r_sum = np.sum(r_count)
        c_sum = len(veh_locs)
        a_sum = len(attack)

        # get the original matching
        if r_sum > 0 and c_sum > 0:
            tlist = [loc[-1] for loc in veh_locs]
            rlist = sum([[i]*min(c_sum,r_count[i]) for i in range(self.num_zone)], [])
            clist = [loc[2]*self.size + loc[3] for loc in veh_locs]

            costs2 = np.zeros((len(rlist), len(clist)))
            for i in range(len(rlist)):
                for j in range(len(clist)):
                    costs2[i,j] = self.travel_distance[clist[j], rlist[i]] + tlist[j]
            if costs2.shape[0] < costs2.shape[1]:
                rids2, cids2 = solve_dense(costs2)
            else:
                cids2, rids2 = solve_dense(costs2.T)
            rids2 = list(rids2)
            cids2 = list(cids2)

            # get the new matching
            alist = attack
            costs = np.zeros((len(rlist)+len(alist), len(clist)))
            for i in range(len(rlist)):
                for j in range(len(clist)):
                    costs[i,j] = self.travel_distance[rlist[i], clist[j]] + tlist[j]
            for i in range(len(alist)):
                for j in range(len(clist)):
                    costs[i+len(rlist),j] = self.travel_distance[alist[i], clist[j]] + tlist[j]

            if costs.shape[0] < costs.shape[1]:
                rids, cids = solve_dense(costs)
            else:
                cids, rids = solve_dense(costs.T)

            for r,c in zip(rids2, cids2):
                if r in rids: # still served
                    new_c = cids[rids2.index(r)]
                    total_cost += costs[r,new_c] - costs2[r,c] # additioal cost for serve the same pass
                else: 
                    total_cost += costs2[r,c] # unserved, need another veh to serve
            # to_remove = []
            # for r,c in zip(rids, cids):
            #     if r >= len(rlist):
            #         # to_remove.append(r - len(rlist))
            #         if c in cids2: # this veh was supposed to serve a real passenger in original matching
            #             rid = rids2[cids2.index(c)]
            #             if rid not in rids: # now the request is unserved
            #                 total_cost += costs[r,c]-\
            #                     costs2[rid, c] + self.travel_distance[alist[r-len(rlist)], rlist[rid]] # >0 then the vehicles is dragging away from its original matching
            #         # else:
            #         #     total_cost += min(max_drag, costs[r,c]) # vehicle is not supposed to match with any pass, the attack is just a blocking of service
            #     elif r in rids2: # if the passenger is also served in the previous solution
            #         cid = cids2[rids2.index(r)]
            #         total_cost += costs[r,c] - costs2[r,cid] # compare the previous matching cost and the new matching cost
            # if total_cost < 0: # why ?
            #     print("TYPE2")
            #     print("now {} previous {}".format(costs[r,c], costs2[r,cid]))
            #     print(rlist)
            #     print(clist)
            #     print(alist)
            #     print(rids)
            #     print(cids)
            #     print(rids2)
            #     print(cids2)
            #     print(r)
            #     print(c)
            #     print(cid)
            #     print(costs)
            #     print(costs2)
            #     print("______")

            # for a in to_remove:
            #     del attack[a]
        # veh_schedule is a three element tuple
        return total_cost, attack





