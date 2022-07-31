import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def run(env, platform, attacker, dd, T, args, matching_freq = 3):
    res = []
    total_time_step = 0
    total_occu_time = 0
    total_pick_time = 0
    total_empty_time = 0
    total_congest_time = 0
    total_waiting_time = 0
    total_generate_pass = 0
    total_assign_pass = 0
    total_pick_pass = 0
    total_served_pass = 0
    total_left_pass = 0
    total_assign_attack = 0
    total_attack_time = 0
    temp_demands = np.zeros((args.frequency, env.num_zone))

    dd_ = []
    for t in range(T):
        dd_.append(np.random.poisson(dd[t, :, :]).astype(int))

    if args.visualize:
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(env.plot_map, cmap = "Greys", zorder = 1) 
        plt.axis("off")
        ims = []

    for t in range(T):
        if t % matching_freq == 0:
            temp_cost, temp_schedule = platform.batch_matching(np.sum(env.pass_count, axis=(0, 2))+np.sum(env.attack_count, axis=(0,1)),
                                                                 env.get_veh_locs())
        else:
            temp_cost, temp_schedule = 0, []
        
        # print(np.sum(dd_))
        temp_demand = dd_[t]
        # to do, insert attacker here
        fake_demand = []
        if t % attacker.frequency == 0:
            if args.mode == 'ILP':
                fake_demand = attacker.generate_attack_msg(t, env.get_veh_locs())
            else:
                fake_demand = attacker.generate_attack_msg(t)
        occu_time, pickup_time, empty_time, congest_time, waiting_time, generate_pass, assign_pass, pick_pass, served_pass, left_pass, assign_attack, attack_time =\
         env.step(temp_demand, fake_demand, temp_schedule)
        total_time_step += 1
        total_occu_time += occu_time
        total_pick_time += pickup_time
        total_empty_time += empty_time
        total_congest_time += congest_time
        total_waiting_time += waiting_time
        total_generate_pass += generate_pass
        total_assign_pass += assign_pass
        total_pick_pass += pick_pass
        total_served_pass += served_pass
        total_left_pass += left_pass
        total_assign_attack += assign_attack
        total_attack_time += attack_time

        res.append([t, occu_time, pickup_time, empty_time, congest_time, waiting_time, generate_pass, assign_pass, pick_pass, served_pass, left_pass, assign_attack, attack_time])
        if t % 120 == 0:
            # writer.flush()
            # print(np.diag(env.veh_count))
            print("Time step: " + str(t) + ", congest_time: " + str(congest_time) + ", assign_pass: " + str(
                assign_pass) + \
                  ", left_pass: " + str(left_pass) + \
                  ", new demand: " + str(np.sum(temp_demand)))
        if args.visualize:
            # add score board
            im5 = [ax.text(0+8, -7, 'Time:\n ' + str(t),ha='center',weight="bold"),\
            ax.text(20+8, -7, 'Served Pass:\n ' + str(total_served_pass),ha='center',weight="bold"),\
            ax.text(45+8, -7, 'Congestion Time:\n ' + str(total_congest_time),ha='center',weight="bold"),\
            ax.text(72+8, -7, 'Assign Attack:\n ' + str(total_assign_attack),ha='center',weight="bold")]
            ims.append(env.plot(ax) + im5)
    print("Total time step: " + str(total_time_step) +", generated_pass: " + str(
        total_generate_pass) +  ", served_pass: " + str(
        total_served_pass) + ", assigned_pass: " + str(
        total_assign_pass) + ", waiting_time: " + str(total_waiting_time) + \
          ", empty_time: " + str(total_empty_time + total_pick_time) + ", occu_time: " + str(
        total_occu_time) +", congest_time: " + str(
        total_congest_time) + ", left_pass: " + str(
        total_left_pass)+ ", assign_attack: " + str(
        total_assign_attack))
    res = pd.DataFrame(res, columns=['t', 'occu_time', 'pickup_time', 'empty_time', 'congest_time', 'waiting_time', 'generate_pass', 'assign_pass', 'pick_pass', 'served_pass', 'left_pass', 'assign_attack', 'attack_time'])
    res.to_csv(
        args.store_res_folder + "/" + f"result_log.csv",
        index=None)
    # env.reset()
    if args.visualize:
        ani = animation.ArtistAnimation(fig, ims)
        ani.save('im.mp4', dpi=120, fps = 12)