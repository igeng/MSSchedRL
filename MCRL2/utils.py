import torch
import random
import numpy as np

def save(args, save_name, model, ep=None):
    import os
    save_dir = './trained_models/' 
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if not ep == None:
        torch.save(model.state_dict(), save_dir + save_name + "_" + str(ep) + ".pth")
    else:
        torch.save(model.state_dict(), save_dir + save_name + ".pth")

def collect_random(env, dataset, num_samples=200):
    state = env._reset()
    for _ in range(num_samples):
        action = random.choice(np.arange(15))
        next_state, reward, done = env._step(action)
        dataset.add(state, action, reward, next_state, done)
        state = next_state
        if done:
            state = env._reset()
