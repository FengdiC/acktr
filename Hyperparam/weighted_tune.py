import os, sys, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
granddir = os.path.dirname(parentdir)
sys.path.insert(0, parentdir)
from weighted_main import NormalizedEnv, AsyncNGAgent, parser
sys.path.insert(0,granddir)
from Components import logger
import itertools
import numpy as np
import gym
import random
import tensorflow.compat.v1 as tf
# tf.compat.v1.disable_v2_behavior()

param = {'scale': [20,40,60,80,100],'lr_wf':[0.01,0.003,0.001],'kl_desired':[0.002,0.001,6e-4]}

args = parser.parse_args()

name = ['acktr','Hopper']
name.append('weighted')
name.append(str(args.seed))
name.append(str(args.kl_desired))
print("hyperparam", '-'.join(name))

logger.configure(args.log_dir, ['csv'], log_suffix='-'.join(name))
title = range(800)

for values in list(itertools.product(param['scale'], param['lr_wf'])):
    args.scale=values[0]
    args.lr_wf=values[1]
    args.agent='weighted'
    # args.kl_desired=values[2]
    returns = []

    checkpoint = 2500
    random.seed(args.seed)
    np.random.seed(args.seed)
    tf.set_random_seed(args.seed)
    env = gym.make(args.env_id)
    # if args.use_pixels:
    #     env = RGBEnv(env)
    # else:
    #     env = NormalizedEnv(env)
    env = NormalizedEnv(env)
    agent = AsyncNGAgent(env, args)
    result, steps = agent.learn()

    ret = np.array(result)
    print(ret.shape)
    returns.append(ret)
    name = [str(k) for k in values]
    name.append(str(args.kl_desired))
    name.append('weighted')
    name.append(str(args.seed))
    print("hyperparam", '-'.join(name))
    logger.logkv("hyperparam", '-'.join(name))
    for n in range(len(steps)):
        logger.logkv(str(title[n]), ret[n])
    logger.dumpkvs()
    logger.logkv("hyperparam", 'time' + '-'.join(name))
    for n in range(len(steps)):
        logger.logkv(str(title[n]), steps[n])
    logger.dumpkvs()

    tf.reset_default_graph()