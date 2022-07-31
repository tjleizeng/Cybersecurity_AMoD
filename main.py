# input: parameters
# output: trainded model under "train mode", tested results under "test mode"
from env import Environment
from controller import Platform
from attacker import Attacker
import sys
import numpy as np
import argparse
import os
from runner import run


# usage example
# python main.py -d grid_small_dynamic/3/0/ --pricing_alg TD3_MLP -alr 0.00001 -clr 0.001 --n_epochs 50 -m all -n 0 --batch_size 32 --seed 5 -sa Gaussian -pd 0 -pe 0 -ac
# TODO:
# Check the policy
# Implement PPO


def get_arguments(argv):
    parser = argparse.ArgumentParser(description='SpatioTemporal Pricing')
    # Simulation settings
    parser.add_argument('-d', '--data_folder', default="data/",
                        help='the folder that contains all the input data')
    parser.add_argument('-s', '--seed', type=int, default=47, help='random seed')
    # Attack settings
    parser.add_argument('-m', '--mode', type=str, default='dummy',
                    help='attack strategy, (default: no attack)')
    parser.add_argument('-b', '--budget', type=int, default=1,
                    help='attack budget, i.e., how many fake requests can be made')
    parser.add_argument('-f', '--frequency', type=int, default=12,
                        help='attack freqency')  # attack every minute
    parser.add_argument('-l', '--length', type=int, default=12,
                        help='length of attack')
    parser.add_argument('-v', '--visualize',  action='store_true', default=False, help='animate the vehicle dynamics')
    args = parser.parse_args(argv)

    return args


if __name__ == '__main__':

    # loading arguments
    args = get_arguments(sys.argv[1:])
    np.random.seed(args.seed)

    scenario_tag = args.data_folder.replace("/", "_") + args.mode + "_" + str(args.budget)+ "_" + str(args.frequency) + "_" + str(args.length) + "_" + str(args.seed)

    # prepare the folders for storing the results
    # store_run_folder = "runs/" + scenario_tag
    store_res_folder = "results/" + scenario_tag 

    # if (not os.path.isdir(
    #         store_run_folder + "/")):
    #     os.makedirs(
    #         store_run_folder + "/")

    if (not os.path.isdir(
            store_res_folder + "/")):
        os.makedirs(
            store_res_folder + "/")

    # args.store_run_folder = store_run_folder
    args.store_res_folder = store_res_folder

    # time_tag = datetime.datetime.now().strftime("%m-%d-%H-%M")
    dd = np.load(args.data_folder + 'dd.npy')
    args.size = int(np.sqrt(dd.shape[1]))
    T = dd.shape[0]

    # initialize environment
    env = Environment(args.size, max_duration = args.length)
    platform = Platform(env.distance, env.num_zone)

    # input: travel distance, travel time matrices, initial vehicle distribution
    # output: time step, profit, reposition cost
    attacker = Attacker(env.distance, env.get_degree(), env.get_betweenness(), args.size, strategy=args.mode, demand_pattern=dd, power = args.budget, frequency= args.frequency, duration = args.length)

    # run exp
    run(env, platform, attacker, dd, T, args)