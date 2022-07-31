import numpy as np
import matplotlib.pyplot as plt

class Environment:
    # A grid network for testing the implication of different attacks
    # Each link has a capacity, mark as the time step for travelling through the link
    # Each intersection can allow the pass of one link, and each veh per direction (hence lane changing is avoided)
    # Matching, repositioning
    # Metrics are num of served pass, waiting time, and empty distances
    # Visualization module
    # 1 s per step
    def __init__(self, size = 5, num_veh = 25, capacity = 10, max_try=5, max_waiting = 30, max_duration = 12, matching_freq = 3):
        self.size = size
        self.capacity = capacity
        self.num_zone = size * size
        self.num_veh = num_veh
        self.max_try = max_try
        self.max_duration = max_duration
        self.matching_freq = matching_freq
        self.actions = [[1, 0], [-1, 0], [0, 1], [0, -1]]
        self.t = 0

        # data for sim
        self.pass_count = np.zeros((max_try, self.num_zone, self.num_zone), dtype=int)
        self.attack_count = np.zeros((max_try, max_duration, self.num_zone), dtype=int)

        # initialize the map
        self.map = dict()
        # complexity O(n^2 ln(n))
        for i in range(size-1):
            for j in range(size):
                for k in range(capacity):
                    self.map[(i,j,i+1,j,k)] = 0
                    self.map[(i+1,j,i,j,k)] = 0
        for i in range(size):
            for j in range(size - 1):
                for k in range(capacity):
                    self.map[(i,j,i,j+1,k)] = 0
                    self.map[(i,j+1,i,j,k)] = 0

        self.plot_map = np.zeros((2 * (self.size  + (self.size - 1) * self.capacity), 2*(self.size  + (self.size - 1) * self.capacity))) # map for plot
        for i in range(0, self.plot_map.shape[0] + 1, 2*(self.capacity + 1)):
            self.plot_map[i, :] = 1
            self.plot_map[:, i] = 1
            self.plot_map[i+1, :] = 1
            self.plot_map[:, i+1] = 1

        for i in range(0, self.plot_map.shape[0] + 1, 2*(self.capacity + 1)):
            for j in range(0, self.plot_map.shape[1] + 1, 2*(self.capacity + 1)):
                self.plot_map[i, j] = 2
                self.plot_map[i+1, j] = 2
                self.plot_map[i, j+1] = 2
                self.plot_map[i+1, j+1] = 2

        # initialize the path and distance matrix
        self.paths = dict()
        self.distance = np.zeros((self.num_zone, self.num_zone))

        def generate_path(x1, y1, x2, y2, delta_x, delta_y):
            # reach destination
            # either move according to delta_x or move according to delta_y
            res = []
            if x1 + delta_x == x2 and y1 == y2:
                return [[(x1, y1, x2, y2)]]
            elif abs(x1 + delta_x - x2) < abs(x1 - x2): # get closer to destination
                for tmp in generate_path(x1+delta_x, y1, x2, y2, delta_x, delta_y):
                    res.append([(x1, y1, x1+delta_x, y1)] + tmp)
            if x1 == x2 and y1 + delta_y == y2:
                return [[(x1, y1, x2, y2)]]
            elif abs(y1 + delta_y - y2) < abs(y1 - y2):
                for tmp in generate_path(x1, y1 +delta_y, x2, y2, delta_x, delta_y):
                    res.append([(x1, y1, x1, y1+delta_y)] + tmp)
            return res

        for x1 in range(size):
            for y1 in range(size):
                for x2 in range(size):
                    for y2 in range(size):
                        delta_x = np.sign(x2 - x1)
                        delta_y = np.sign(y2 - y1)
                        # question: how to generate all candidate paths?
                        self.paths[((x1,y1),(x2,y2))] = generate_path(x1,y1,x2,y2,delta_x,delta_y)
                        self.distance[(x1*self.size + y1,x2*self.size + y2)] = (np.abs(x2 - x1) + np.abs(y2 - y1))*capacity

                        assert len(self.paths[((x1,y1),(x2,y2))])>0, print([x1,y1,x2,y2,delta_x, delta_y])


        # initailize veh
        self.veh_list = []  # list of all vehicles
        self.initialize_veh()
        
        # generate random numbers (uniformly distributed) for future usage, each one is from 0 to 1
        # self.rand_int = list(np.random.randint(0,max_waiting-12,size=10000)) # waiting time randomness
        self.rand_double = list(np.random.random(size=10000))

    def zone_shortest_path(self, x1, y1, x2, y2):
        # Randomly select one of the shortest path
        if x1 != x2 or y1 != y2:
            n = len(self.paths[((x1, y1), (x2, y2))])
            if len(self.rand_double) == 0:
                self.rand_double = list(np.random.random(size=10000))
            rand_num = self.rand_double.pop()
            rand_num = int(np.floor(rand_num*n))
            return self.paths[((x1,y1), (x2, y2))][rand_num].copy()
        else:
            return []

    def shortest_path(self, x1, y1, x2, y2, t, x3, y3):
        return self.zone_shortest_path(x2, y2, x3, y3)
        # return list of intermediate points

    def initialize_veh(self):
        veh_locs = np.random.choice(len(self.map), self.num_veh, replace=False)
        vid = 1
        for v_loc in veh_locs:
            v = Vehicle(vid, list(self.map.keys())[v_loc])
            self.map[list(self.map.keys())[v_loc]] = vid
            self.veh_list.append(v)
            vid += 1

    def shortest_distance(self, x1, y1, x2, y2, t, x3, y3):
        return t + self.distance[x2*self.size + y2, x3*self.size + y3] + 1

    def get_veh_locs(self):
        return [veh.loc for veh in self.veh_list if veh.state == 0]

    def get_valid_action(self, x, y):
        return [(i, j) for i,j in self.actions if ((x+i)>=0 and (x+i)<self.size and (y+j)>=0 and (y+j)<self.size)]

    def get_degree(self):
        connectivity = [0] * self.num_zone
        for i in range(self.size):
            for j in range(self.size):
                connectivity[i*self.size + j] = len(self.get_valid_action(i,j))
        return connectivity

    def get_betweenness(self):
        betweenness = [0] * self.num_zone
        for pathsets in self.paths.values():
            for path in pathsets:
                for x1, y1, x2, y2 in path:
                    betweenness[x1*self.size + y1] += 1
                betweenness[x2*self.size + y2] += 1 # end node
        return betweenness

    def veh_movement(self, v):
        # print(v.id)
        # print(v.state)
        # print(v.path)
        # print(v.loc)
        # print(v.destination)
        if v.loc[-1] == 0: # reaching to the end of the current link
            # try to get to the next link
            x1, y1, x2, y2 = v.path[0]
            if self.is_valid_movement(x1, y1, x2, y2, self.capacity-1):
                self.map[v.loc] = 0
                v.loc = (x1, y1, x2, y2, self.capacity-1)
                self.map[v.loc] = v.id
                del v.path[0]
            else:
                return False
        else:
            x1, y1, x2, y2, t = v.loc
            if self.is_valid_movement(x1, y1, x2, y2, t-1):
                self.map[v.loc] = 0
                v.loc = (x1, y1, x2, y2, t-1)
                self.map[v.loc] = v.id
            else:
                return False
        return True

    def veh_reposition(self,v):
        x1, y1, x2, y2, t = v.loc
        if len(self.rand_double) == 0:
            self.rand_double = list(np.random.random(size=10000))
        rand_num = self.rand_double.pop()
        candidates = self.get_valid_action(x2,y2)
        delta_x, delta_y = candidates[int(np.floor(rand_num * len(candidates)))]
        v.reposition(delta_x, delta_y)


    def step(self, new_demand, fake_demand, veh_schedule):
        #### at time t
        # dispatch_veh
        occu_time = 0
        pickup_time = 0
        attack_time = 0
        empty_time = 0
        congest_time = 0
        waiting_time = 0
        generate_pass = 0
        assign_pass = 0
        assign_attack = 0
        pick_pass = 0
        served_pass = 0
        left_pass = 0

        # print(self.zone_shortest_path(0,0,4,4))
        # print(self.shortest_path(0,0,0,1,5,4,4))

        # time step t
        ## update veh schedules
        for p_loc, v_loc in veh_schedule:
            p_loc2 = self.num_zone
            for j in range(self.max_try - 1, -1, -1):
                for i in range(self.num_zone):
                    if self.pass_count[j, p_loc, i] > 0:
                        self.pass_count[j, p_loc, i] -= 1
                        p_loc2 = i
                        break
                if p_loc2 != self.num_zone:
                    break
                # check if there is attack
                for i in range(self.max_duration - 1, -1, -1):
                    if self.attack_count[j, i, p_loc] >0:
                        self.attack_count[j, i, p_loc] -= 1
                        p_loc2 = i - self.max_duration
                        break
                if p_loc2 != self.num_zone:
                    break
            assert self.map[v_loc] > 0 and p_loc2  != self.num_zone, print(
                "Error" + "," + str(np.sum(self.pass_count, axis=(0, 2)))+"," + str(np.sum(self.attack_count, axis=(0, 1))) + "," + str(p_loc) + "," + str(p_loc2) + ","+ str(v_loc))
            v = self.veh_list[self.map[v_loc]-1]
            x1, y1, x2, y2, t = v_loc
            x3, y3 = p_loc//self.size, p_loc%self.size
            if p_loc2 >= 0:
                x4, y4 = p_loc2//self.size, p_loc2 % self.size
                # print((x1, y1, x2, y2, t, x3, y3))
                # print((x3,y3,x4,y4))
                v.path = self.shortest_path(x1, y1, x2, y2, t, x3, y3) + self.zone_shortest_path(x3,y3,x4,y4)
                v.destination = (x3, y3)
                v.state = 1
                assign_pass += 1
            else:
                v.path = self.shortest_path(x1, y1, x2, y2, t, x3, y3) 
                v.destination = (x3, y3)
                # if not v.path:
                #     print(v.loc)
                #     print(v.destination)
                #     print((x2,y2,x3,y3))
                #     print(self.paths[((x2,y2), (x3, y3))])
                v.state = p_loc2
                assign_attack += 1

        # between time t and t+1
        # update_veh
        for v in self.veh_list:
            ## vehicle movement
            if v.state == 0:
                # print(v.loc)
                empty_time += 1
                if v.loc[-1] == 0 and not v.path: # reaching to the end of the current link
                    self.veh_reposition(v)
                if not self.veh_movement(v):
                    congest_time += 1
                # print(v.loc)

            ## vehicles pickup
            elif v.state == 1:
                # print(v.loc)
                # print(v.path)
                pickup_time += 1
                # vehicle approaches passengers
                if tuple(v.loc[2:4]) == v.destination and v.loc[-1] == 0:
                    pick_pass += 1
                    v.state = 2
                    v.destination = v.path[-1][-2:]
                    # del v.path[0]
                else:
                    if not self.veh_movement(v):
                        congest_time += 1

            elif v.state == 2:
                occu_time += 1
                if tuple(v.loc[2:4]) == v.destination and v.loc[-1] == 0:
                    served_pass += 1
                    v.state = 0
                    v.destination = None
                    self.veh_reposition(v)
                else:
                    if not self.veh_movement(v):
                        congest_time += 1

            ## vehicles attacked
            elif v.state < 0:
                # print(v.loc)
                # print(v.path)
                attack_time += 1
                empty_time += 1
                # vehicle approaches fake destination
                if tuple(v.loc[2:4]) == v.destination and v.loc[-1] == 0:
                    v.state = 0
                    v.destination = None
                    self.veh_reposition(v)
                else:
                    v.state += 1
                    if v.state == 0 and v.loc[-1] == 0 and not v.path: # reaching to the end of the current link
                        self.veh_reposition(v) # attack timeout
                    elif not self.veh_movement(v):
                        congest_time += 1

        # generate_pass
        if self.t % self.matching_freq == 0:
            left_pass += np.sum(self.pass_count[-1, :, :])
            self.pass_count[1:, :, :] = self.pass_count[0:-1, :, :]
            self.pass_count[0, :, :] = 0
            self.attack_count[1:, :, :] = self.attack_count[0:-1, :, :] # attack not respond
            self.attack_count[0, :, :] = 0

        self.pass_count[0, :, :] += new_demand
        # print(np.sum(self.pass_count,(1,2)))
        generate_pass += np.sum(new_demand)
        self.attack_count[:,1:,:] = self.attack_count[:,0:-1,:] # attack died out
        self.attack_count[:,0,:] = 0

        # print(np.sum(self.attack_count))

        for o_tmp in fake_demand:
                self.attack_count[0, 0, o_tmp] += 1
        
        waiting_time = np.sum(self.pass_count)

        self.t += 1

        return occu_time, pickup_time, empty_time, congest_time, waiting_time, generate_pass, assign_pass, pick_pass, served_pass, left_pass, assign_attack, attack_time

    def is_valid_movement(self, x1, y1, x2, y2, t):
        return not self.map[(x1, y1, x2, y2, t)]

    def reset(self):
        self.pass_count *= 0
        # initialize the map
        self.map = dict()
        # complexity O(n^2 ln(n))
        for i in range(self.size-1):
            for j in range(self.size):
                for k in range(self.capacity):
                    self.map[(i,j,i+1,j,k)] = 0
                    self.map[(i+1,j,i,j,k)] = 0
        for i in range(self.size):
            for j in range(self.size - 1):
                for k in range(self.capacity):
                    self.map[(i,j,i,j+1,k)] = 0
                    self.map[(i,j+1,i,j,k)] = 0
        self.veh_list = []  # list of all vehicles
        self.initialize_veh()
        self.t = 0

    def plot(self, ax):
        # plot the service, grid with two colors, grey for the road, black for the zone
        # yellow for the cab, and red for the passenger
        # plot the info of all vehicles
        x_empty, y_empty = [], []
        x_pickup, y_pickup = [], []
        x_occupied, y_occupied = [], []
        x_attacked, y_attacked = [], []
        for veh in self.veh_list:
            x1, y1, x2, y2, t = veh.loc
            # think of this
            x_tmp = 2*(x2*(self.capacity+1) + (x1 - x2)*t) + (y2 - y1 < 0) + 2 * (y2 - y1 == 0)
            y_tmp = 2*(y2*(self.capacity+1) + (y1 - y2)*t) + (x2 - x1 > 0) + 2 * (x2 - x1 == 0)
            if veh.state == 0:
                x_empty.append(x_tmp)
                y_empty.append(y_tmp)
            elif veh.state == 1:
                x_pickup.append(x_tmp)
                y_pickup.append(y_tmp)
            elif veh.state == 2:
                x_occupied.append(x_tmp)
                y_occupied.append(y_tmp)
            else:
                x_attacked.append(x_tmp)
                y_attacked.append(y_tmp)


        # plot and mark the info of the passenger
        im1 = ax.scatter(x_empty, y_empty, color = 'Yellow')
        im2 = ax.scatter(x_pickup, y_pickup, color = 'Green')
        im3 = ax.scatter(x_occupied, y_occupied, color = 'Blue')
        im4 = ax.scatter(x_attacked, y_attacked, color = 'Red')
        # plot the request info
        pass_sum = np.sum(self.pass_count, axis = (0,2)).reshape(self.size, self.size)
        im5 = []
        for i in range(pass_sum.shape[0]):
            for j in range(pass_sum.shape[1]):
                im_tmp = ax.text(i*(1+self.capacity)*2+2, j*(1+self.capacity)*2-2, pass_sum[i,j])
                im5.append(im_tmp)
        
        return [im1, im2, im3, im4] + im5


class Vehicle:
    def __init__(self, vid, vloc):
        self.id = vid
        self.loc = vloc
        self.state = 0  # 0 for available, 1 in a pickup trip, 2 occupied, <0 matched to a fake demand
        self.path = [] # sequence of (x1,y1,x2,y2) to follow on the map
        self.destination = None

        # data
        self.occu_time = 0
        self.empty_time = 0

    def reposition(self, delta_x, delta_y):
        x1, y1, x2, y2, t = self.loc
        self.path = [(x2, y2, x2 + delta_x, y2 + delta_y)]
        self.destination = None



