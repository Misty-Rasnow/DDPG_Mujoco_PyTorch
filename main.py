import os
import random
import numpy as np
from tqdm import tqdm
from matplotlib import animation
import matplotlib.pyplot as plt

import torch

from agent import DDPGAgent, DPGAgent
from env import MyEnv
from memory import ReplayMemory
from utils import OUNoise, NormalNoise

import warnings
warnings.filterwarnings("ignore", category=UserWarning)


GAMMA = 0.95
TAU = 0.01
GLOBAL_SEED = 0
MEM_SIZE = 50_000
RENDER = False
MODEL_PREFIX = "./models"
EVAL_PREFIX = "./eval"

SIGMA_START = 0.2
SIGMA_END = 0.01
SIGMA_DECAY = (SIGMA_START - SIGMA_END) / 1_000_000

BATCH_SIZE = 128
WARM_STEPS = 5_000
MAX_STEPS = 2_000_000
SAVE_FREQ = 10_000

rand = random.Random()
rand.seed(GLOBAL_SEED)
new_seed = lambda: rand.randint(0, 1_000_000)
if os.path.exists(MODEL_PREFIX) == False :
    os.mkdir(MODEL_PREFIX)
if os.path.exists(EVAL_PREFIX) == False :
    os.mkdir(EVAL_PREFIX)

torch.manual_seed(new_seed())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

env = MyEnv(device)
noise = OUNoise(size = env.get_action_dim(), seed = new_seed, sigma=SIGMA_START, sigma_min = SIGMA_END, sigma_decay = SIGMA_DECAY)
# noise = NormalNoise(size = env.get_action_dim(), seed = new_seed, sigma=SIGMA_START, sigma_min = SIGMA_END, sigma_decay = SIGMA_DECAY)
agent = DDPGAgent(
    env.get_state_dim(),
    env.get_action_dim(),
    device,
    GAMMA,
    TAU,
    new_seed(),
    noise
)
memory = ReplayMemory(env.get_state_dim(), env.get_action_dim(), MEM_SIZE, device)

#### Training ####
state = None
done = True

fp = open("rewards.txt", "w")
fp.truncate()
fp.close()
# train for
progressive = tqdm(range(MAX_STEPS), total=MAX_STEPS,
                   ncols=50, leave=False, unit="b")
for step in progressive:
    # 重置环境
    if done:
        state = env.reset()
    training = len(memory) > WARM_STEPS
    action = agent.choose_action(state, training = True)
    action = action.cpu().detach().numpy()
    next_state, reward, done = env.step(action)
    memory.push(state, next_state, torch.Tensor(action), reward, done)
    state = next_state
    if training :
        agent.learn(memory, BATCH_SIZE)
        agent.sync()
    # 
    if step % SAVE_FREQ == 0:
        avg_reward, frames = env.evaluate(agent, render=RENDER)
        with open("rewards.txt", "a") as fp:
            fp.write("%3d %8d %1f\n" % (step // SAVE_FREQ, step, avg_reward))
        if RENDER:
            prefix = f"eval_{step // SAVE_FREQ:03d}.gif"
            patch = plt.imshow(frames[0])
            plt.axis('off')
            def animate(i):
                patch.set_data(frames[i])
            anim = animation.FuncAnimation(plt.gcf(), animate, frames=len(frames), interval=5)
            anim.save(os.path.join(EVAL_PREFIX, prefix), fps=30)
        agent.save(os.path.join(
            MODEL_PREFIX, f"model_{step // SAVE_FREQ:03d}"))
        done = True
