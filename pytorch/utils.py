from __future__ import print_function
import math
import os
import random
import copy
import scipy
import imageio
import string
import numpy as np
from skimage.transform import resize
import torch
from torch import nn
from torch.distributions import MultivariateNormal
try:  # SciPy >= 0.19
    from scipy.special import comb
except ImportError:
    from scipy.misc import comb

e_greedy = 0.2

def bernstein_poly(i, n, t):
    """
     The Bernstein polynomial of n, i as a function of t
    """

    return comb(n, i) * ( t**(n-i) ) * (1 - t)**i

def bezier_curve(points, nTimes=1000):
    """
       Given a set of control points, return the
       bezier curve defined by the control points.

       Control points should be a list of lists, or list of tuples
       such as [ [1,1], 
                 [2,3], 
                 [4,5], ..[Xn, Yn] ]
        nTimes is the number of time steps, defaults to 1000

        See http://processingjs.nihongoresources.com/bezierinfo/
    """

    nPoints = len(points)
    xPoints = np.array([p[0] for p in points])
    yPoints = np.array([p[1] for p in points])

    t = np.linspace(0.0, 1.0, nTimes)

    polynomial_array = np.array([ bernstein_poly(i, nPoints-1, t) for i in range(0, nPoints)   ])
    
    xvals = np.dot(xPoints, polynomial_array)
    yvals = np.dot(yPoints, polynomial_array)

    return xvals, yvals

def data_augmentation(x, y, prob=0.5):
    # augmentation by flipping
    cnt = 3
    while random.random() < prob and cnt > 0:
        degree = random.choice([0, 1, 2])
        x = np.flip(x, axis=degree)
        y = np.flip(y, axis=degree)
        cnt = cnt - 1

    return x, y

def nonlinear_transformation(x, prob=0.5):
    if random.random() >= prob:
        return x
    points = [[0, 0], [random.random(), random.random()], [random.random(), random.random()], [1, 1]]
    xpoints = [p[0] for p in points]
    ypoints = [p[1] for p in points]
    xvals, yvals = bezier_curve(points, nTimes=100000)
    if random.random() < 0.5:
        # Half change to get flip
        xvals = np.sort(xvals)
    else:
        xvals, yvals = np.sort(xvals), np.sort(yvals)
    nonlinear_x = np.interp(x, xvals, yvals)
    return nonlinear_x

def local_pixel_shuffling(x, prob=0.5):
    if random.random() >= prob:
        return x
    image_temp = copy.deepcopy(x)
    orig_image = copy.deepcopy(x)
    _, img_rows, img_cols, img_deps = x.shape
    num_block = 10000
    for _ in range(num_block):
        block_noise_size_x = random.randint(1, img_rows//10)
        block_noise_size_y = random.randint(1, img_cols//10)
        block_noise_size_z = random.randint(1, img_deps//10)
        noise_x = random.randint(0, img_rows-block_noise_size_x)
        noise_y = random.randint(0, img_cols-block_noise_size_y)
        noise_z = random.randint(0, img_deps-block_noise_size_z)
        window = orig_image[0, noise_x:noise_x+block_noise_size_x, 
                               noise_y:noise_y+block_noise_size_y, 
                               noise_z:noise_z+block_noise_size_z,
                           ]
        window = window.flatten()
        np.random.shuffle(window)
        window = window.reshape((block_noise_size_x, 
                                 block_noise_size_y, 
                                 block_noise_size_z))
        image_temp[0, noise_x:noise_x+block_noise_size_x, 
                      noise_y:noise_y+block_noise_size_y, 
                      noise_z:noise_z+block_noise_size_z] = window
    local_shuffling_x = image_temp

    return local_shuffling_x

def image_in_painting(x):
    _, img_rows, img_cols, img_deps = x.shape
    cnt = 5
    while cnt > 0 and random.random() < 0.95:
        block_noise_size_x = random.randint(img_rows//6, img_rows//3)
        block_noise_size_y = random.randint(img_cols//6, img_cols//3)
        block_noise_size_z = random.randint(img_deps//6, img_deps//3)
        noise_x = random.randint(3, img_rows-block_noise_size_x-3)
        noise_y = random.randint(3, img_cols-block_noise_size_y-3)
        noise_z = random.randint(3, img_deps-block_noise_size_z-3)
        x[:, 
          noise_x:noise_x+block_noise_size_x, 
          noise_y:noise_y+block_noise_size_y, 
          noise_z:noise_z+block_noise_size_z] = np.random.rand(block_noise_size_x, 
                                                               block_noise_size_y, 
                                                               block_noise_size_z, ) * 1.0
        cnt -= 1
    return x

def image_out_painting(x):
    _, img_rows, img_cols, img_deps = x.shape
    image_temp = copy.deepcopy(x)
    x = np.random.rand(x.shape[0], x.shape[1], x.shape[2], x.shape[3], ) * 1.0
    block_noise_size_x = img_rows - random.randint(3*img_rows//7, 4*img_rows//7)
    block_noise_size_y = img_cols - random.randint(3*img_cols//7, 4*img_cols//7)
    block_noise_size_z = img_deps - random.randint(3*img_deps//7, 4*img_deps//7)
    noise_x = random.randint(3, img_rows-block_noise_size_x-3)
    noise_y = random.randint(3, img_cols-block_noise_size_y-3)
    noise_z = random.randint(3, img_deps-block_noise_size_z-3)
    x[:, 
      noise_x:noise_x+block_noise_size_x, 
      noise_y:noise_y+block_noise_size_y, 
      noise_z:noise_z+block_noise_size_z] = image_temp[:, noise_x:noise_x+block_noise_size_x, 
                                                       noise_y:noise_y+block_noise_size_y, 
                                                       noise_z:noise_z+block_noise_size_z]
    cnt = 4
    while cnt > 0 and random.random() < 0.95:
        block_noise_size_x = img_rows - random.randint(3*img_rows//7, 4*img_rows//7)
        block_noise_size_y = img_cols - random.randint(3*img_cols//7, 4*img_cols//7)
        block_noise_size_z = img_deps - random.randint(3*img_deps//7, 4*img_deps//7)
        noise_x = random.randint(3, img_rows-block_noise_size_x-3)
        noise_y = random.randint(3, img_cols-block_noise_size_y-3)
        noise_z = random.randint(3, img_deps-block_noise_size_z-3)
        x[:, 
          noise_x:noise_x+block_noise_size_x, 
          noise_y:noise_y+block_noise_size_y, 
          noise_z:noise_z+block_noise_size_z] = image_temp[:, noise_x:noise_x+block_noise_size_x, 
                                                           noise_y:noise_y+block_noise_size_y, 
                                                           noise_z:noise_z+block_noise_size_z]
        cnt -= 1
    return x
                
def rl_data_augmentation(x, y, action, prob=0.5):
    # augmentation by flipping
    cnt = 3
    while action[0] < prob and cnt > 0:
        degree = np.argmax(action[1:4])
        x = np.flip(x, axis=degree)
        y = np.flip(y, axis=degree)
        cnt = cnt - 1

    return x, y

def rl_nonlinear_transformation(x, action, prob=0.5):
    if action[0] >= prob:
        return x
    points = [[0, 0], [action[1], action[2]], [action[3], action[4]], [1, 1]]
    xpoints = [p[0] for p in points]
    ypoints = [p[1] for p in points]
    xvals, yvals = bezier_curve(points, nTimes=100000)
    if action[5] < 0.5:
        # Half change to get flip
        xvals = np.sort(xvals)
    else:
        xvals, yvals = np.sort(xvals), np.sort(yvals)
    nonlinear_x = np.interp(x, xvals, yvals)
    return nonlinear_x

def rl_image_in_painting(x, action):
    _, img_rows, img_cols, img_deps = x.shape
    cnt = 4
    while cnt >= 0:
        block_noise_size_x = img_rows//6 + int((img_rows//3-img_rows//6)*action[0 + cnt*6])
        block_noise_size_y = img_cols//6 + int((img_cols//3-img_cols//6)*action[1 + cnt*6])
        block_noise_size_z = img_deps//6 + int((img_deps//3-img_deps//6)*action[2 + cnt*6])
        noise_x = 3 + int((img_rows-block_noise_size_x-3-3) * action[3 + cnt*6])
        noise_y = 3 + int((img_cols-block_noise_size_y-3-3) * action[4 + cnt*6])
        noise_z = 3 + int((img_deps-block_noise_size_z-3-3) * action[5 + cnt*6])
        x[:, 
          noise_x:noise_x+block_noise_size_x, 
          noise_y:noise_y+block_noise_size_y, 
          noise_z:noise_z+block_noise_size_z] = np.random.rand(block_noise_size_x, 
                                                               block_noise_size_y, 
                                                               block_noise_size_z, ) * 1.0
        cnt -= 1
    return x

def rl_image_out_painting(x, action):
    _, img_rows, img_cols, img_deps = x.shape
    image_temp = copy.deepcopy(x)
    x = np.random.rand(x.shape[0], x.shape[1], x.shape[2], x.shape[3], ) * 1.0
    block_noise_size_x = img_rows - int(3*img_rows//7 + (4*img_rows//7 - 3*img_rows//7)*action[0])
    block_noise_size_y = img_cols - int(3*img_cols//7 + (4*img_cols//7 - 3*img_cols//7)*action[1])
    block_noise_size_z = img_deps - int(3*img_deps//7 + (4*img_deps//7 - 3*img_deps//7)*action[2])
    noise_x = int(3 + (img_rows-block_noise_size_x-3-3) * action[3])
    noise_y = int(3 + (img_cols-block_noise_size_y-3-3) * action[4])
    noise_z = int(3 + (img_deps-block_noise_size_z-3-3) * action[5])
    
    x[:, 
      noise_x:noise_x+block_noise_size_x, 
      noise_y:noise_y+block_noise_size_y, 
      noise_z:noise_z+block_noise_size_z] = image_temp[:, noise_x:noise_x+block_noise_size_x, 
                                                       noise_y:noise_y+block_noise_size_y, 
                                                       noise_z:noise_z+block_noise_size_z]
    cnt = 4
    while cnt > 0:
        block_noise_size_x = img_rows - int(3*img_rows//7 + (4*img_rows//7 - 3*img_rows//7)*action[0 + cnt*6])
        block_noise_size_y = img_cols - int(3*img_cols//7 + (4*img_cols//7 - 3*img_cols//7)*action[1 + cnt*6])
        block_noise_size_z = img_deps - int(3*img_deps//7 + (4*img_deps//7 - 3*img_deps//7)*action[2 + cnt*6])
        noise_x = int(3 + (img_rows-block_noise_size_x-3-3) * action[3 + cnt*6])
        noise_y = int(3 + (img_cols-block_noise_size_y-3-3) * action[4 + cnt*6])
        noise_z = int(3 + (img_deps-block_noise_size_z-3-3) * action[5 + cnt*6])
        
        x[:, 
          noise_x:noise_x+block_noise_size_x, 
          noise_y:noise_y+block_noise_size_y, 
          noise_z:noise_z+block_noise_size_z] = image_temp[:, noise_x:noise_x+block_noise_size_x, 
                                                           noise_y:noise_y+block_noise_size_y, 
                                                           noise_z:noise_z+block_noise_size_z]
        cnt -= 1
    return x

def generate_pair(x_train, batch_size, config, status="test"):
    while True:
        # x_train file for loop
        for x_i in range(len(x_train)):
            s = np.load(x_train[x_i])
            img = np.expand_dims(np.array(s), axis=1)
            img_rows, img_cols, img_deps = img.shape[2], img.shape[3], img.shape[4]
            index = [i for i in range(len(img))]
            random.shuffle(index)
            for index_i in range(len(s)//batch_size):
                y = img[index[index_i*batch_size:(index_i+1)*batch_size]]
                x = copy.deepcopy(y)

                for n in range(x.shape[0]):
                    
                    # Autoencoder
                    x[n] = copy.deepcopy(y[n])
                    
                    # Flip
                    x[n], y[n] = data_augmentation(x[n], y[n], config.flip_rate)

                    # # Local Shuffle Pixel
                    x[n] = local_pixel_shuffling(x[n], prob=config.local_rate)
                    
                    # Apply non-Linear transformation with an assigned probability
                    x[n] = nonlinear_transformation(x[n], config.nonlinear_rate)
                    
                    # Inpainting & Outpainting
                    if random.random() < config.paint_rate:
                        if random.random() < config.inpaint_rate:
                            # Inpainting
                            x[n] = image_in_painting(x[n])
                        else:
                            # Outpainting
                            x[n] = image_out_painting(x[n])

                # Save sample images module
                if config.save_samples is not None and status == "train" and random.random() < 0.01:
                    n_sample = random.choice( [i for i in range(x.shape[0])] )
                    sample_1 = np.concatenate((x[n_sample,0,:,:,2*img_deps//6], y[n_sample,0,:,:,2*img_deps//6]), axis=1)
                    sample_2 = np.concatenate((x[n_sample,0,:,:,3*img_deps//6], y[n_sample,0,:,:,3*img_deps//6]), axis=1)
                    sample_3 = np.concatenate((x[n_sample,0,:,:,4*img_deps//6], y[n_sample,0,:,:,4*img_deps//6]), axis=1)
                    sample_4 = np.concatenate((x[n_sample,0,:,:,5*img_deps//6], y[n_sample,0,:,:,5*img_deps//6]), axis=1)
                    final_sample = np.concatenate((sample_1, sample_2, sample_3, sample_4), axis=0)
                    final_sample = final_sample * 255.0
                    final_sample = final_sample.astype(np.uint8)
                    file_name = ''.join([random.choice(string.ascii_letters + string.digits) for n in range(10)])+'.'+config.save_samples
                    imageio.imwrite(os.path.join(config.sample_path, config.exp_name, file_name), final_sample)

                yield (x, y)

def rl_generate_pair(x_train, batch_size, config, netM, status="test"):
    while True:
        # x_train file for loop
        for x_i in range(len(x_train)):
            s = np.load(x_train[x_i])
            img = np.expand_dims(np.array(s), axis=1)
            img_rows, img_cols, img_deps = img.shape[2], img.shape[3], img.shape[4]
            index = [i for i in range(len(img))]
            random.shuffle(index)
            for index_i in range(len(s)//batch_size):
                y = img[index[index_i*batch_size:(index_i+1)*batch_size]]
                x = copy.deepcopy(y)

                for n in range(x.shape[0]):
                    
                    # Autoencoder
                    x[n] = copy.deepcopy(y[n])

                    action = netM.select_action(torch.tensor(np.expand_dims(x[n], 0)).float().cuda(0)).squeeze()
                    
                    # Flip
                    x[n], y[n] = rl_data_augmentation(x[n], y[n], action[0:4].detach().cpu().numpy(), config.flip_rate)

                    # no Local Shuffle Pixel on rl
                    # x[n] = local_pixel_shuffling(x[n], prob=config.local_rate)
                    
                    # Apply non-Linear transformation with an assigned probability
                    x[n] = rl_nonlinear_transformation(x[n], action[4:10].detach().cpu().numpy(), config.nonlinear_rate)
                    
                    # Inpainting & Outpainting
                    if random.random() < config.paint_rate:
                        if random.random() < config.inpaint_rate:
                            # Inpainting
                            x[n] = rl_image_in_painting(x[n], action[10:40].detach().cpu().numpy())
                        else:
                            # Outpainting
                            x[n] = rl_image_out_painting(x[n], action[40:70].detach().cpu().numpy())

                # Save sample images module
                if config.save_samples is not None and status == "train" and random.random() < 0.01:
                    n_sample = random.choice( [i for i in range(x.shape[0])] )
                    sample_1 = np.concatenate((x[n_sample,0,:,:,2*img_deps//6], y[n_sample,0,:,:,2*img_deps//6]), axis=1)
                    sample_2 = np.concatenate((x[n_sample,0,:,:,3*img_deps//6], y[n_sample,0,:,:,3*img_deps//6]), axis=1)
                    sample_3 = np.concatenate((x[n_sample,0,:,:,4*img_deps//6], y[n_sample,0,:,:,4*img_deps//6]), axis=1)
                    sample_4 = np.concatenate((x[n_sample,0,:,:,5*img_deps//6], y[n_sample,0,:,:,5*img_deps//6]), axis=1)
                    final_sample = np.concatenate((sample_1, sample_2, sample_3, sample_4), axis=0)
                    final_sample = final_sample * 255.0
                    final_sample = final_sample.astype(np.uint8)
                    file_name = ''.join([random.choice(string.ascii_letters + string.digits) for n in range(10)])+'.'+config.save_samples
                    imageio.imwrite(os.path.join(config.sample_path, config.exp_name, file_name), final_sample)

                yield (x, y)

class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []

    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]

class ActorCritic(nn.Module):
    def __init__(self, gpu=0):
        super(ActorCritic, self).__init__()
        self.gpu = gpu
        # actor
        self.actor = PPO_Base_Model(
            nc=1, out_channels=70
        )
        # critic
        self.critic = PPO_Base_Model(nc=1, out_channels=1)

        self.action_var = torch.full((1, 70), 0.2 * 0.2).to(self.gpu)

    def forward(self):
        raise NotImplementedError

    def act(self, state):
        action = self.actor(state)
        # print(action[0].shape) # torch.Size([4, 8, 72, 74, 88])
        action_logprob = torch.log(action + 1e-5)

        return action.detach(), action_logprob.detach()

    def evaluate(self, state, action):
        action_mean = self.actor(state)
        action_var = self.action_var.expand_as(action_mean)
        cov_mat = torch.diag_embed(action_var).to(self.gpu)
        dist = MultivariateNormal(action_mean, cov_mat)

        # action_logprobs = torch.log(action + 1e-5)
        action_logprobs = dist.log_prob(action)
        
        dist_entropy = dist.entropy().mean()

        # nc = 2
        # action_mask = self.action_mask(action)
        # state_values = self.critic(torch.concat([state, action_mask], dim=1))

        # nc = 1
        state_values = self.critic(state)

        return action_logprobs, state_values, dist_entropy

class PPO_Base_Model(nn.Module):
    def __init__(self, nc=1, ndf=8, out_channels=1):
        super(PPO_Base_Model, self).__init__()
        self.main = nn.Sequential(
            # input is (nc) x 128 x 128 x 128
            nn.Conv3d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 64 x 64 x 64
            nn.Conv3d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm3d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 32 x 32 x 32
            nn.Conv3d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm3d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 16 x 16 x 16
            nn.Conv3d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm3d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(ndf * 8, ndf * 4, 3, 1, 0, bias=False), 
            # nn.Conv3d(ndf * 16, 1, 4, 1, 0, bias=False), # for 121x121x145
            # nn.Sigmoid(), # for 121x121x145
        )
        # for 121x121x145
        self.fc_layer = nn.Sequential(
            nn.Linear(256, out_channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        out = self.main(x)
        # for 121x121x145
        out = out.view(out.size(0), -1)
        # print(out.shape)
        out = self.fc_layer(out)

        return out

class PPO:
    def __init__(self, lr_actor, lr_critic, eps_clip, K_epochs, gpu):
        self.gpu = gpu
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs

        self.buffer = RolloutBuffer()

        self.policy = ActorCritic(self.gpu).cuda(self.gpu)
        self.optimizer = torch.optim.Adam(
            [
                {"params": self.policy.actor.parameters(), "lr": lr_actor},
                {"params": self.policy.critic.parameters(), "lr": lr_critic},
            ]
        )

        self.policy_old = ActorCritic(self.gpu).cuda(self.gpu)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.MseLoss = nn.MSELoss()

    def select_action(self, state):
        if np.random.random() < e_greedy:
            action = torch.rand((state.shape[0], 70)).cuda(self.gpu)
            action = action / 2 + np.random.uniform(0, 0.5)
            action_logprob = torch.log(action + 1e-5)
        else:
            with torch.no_grad():
                action, action_logprob = self.policy_old.act(state)

        self.buffer.states.append(state.detach())
        self.buffer.actions.append(action.detach())
        self.buffer.logprobs.append(action_logprob.detach())

        return action

    def only_action(self, state):
        with torch.no_grad():
            action, action_logprob = self.policy_old.act(state)

        return action

    def cuda(self, gpu):
        self.policy.cuda(gpu)
        self.policy_old.cuda(gpu)

    def update(self):
        # rewards to tensor and normalize
        # rewardss = torch.cat(self.buffer.rewards, dim=0).cuda(self.gpu)
        # if torch.isnan(rewards.std()).all() == False:
        #     rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)

        # convert list to tensor
        # old_statess = torch.cat(self.buffer.states, dim=0).detach().cuda(self.gpu)
        # old_actionss = torch.cat(self.buffer.actions, dim=0).detach().cuda(self.gpu)
        # old_logprobss = torch.cat(self.buffer.logprobs, dim=0).detach().cuda(self.gpu)
        # Optimize policy for K epochs
        
        rewardss = torch.cat(self.buffer.rewards, dim=0).cuda(self.gpu)
        for _ in range(self.K_epochs):
            for i in range(len(self.buffer.states)):
                old_states = self.buffer.states[i].cuda(self.gpu)
                old_actions = self.buffer.actions[i].cuda(self.gpu)
                old_logprobs = self.buffer.logprobs[i].cuda(self.gpu)
                rewards = rewardss[i].cuda(self.gpu)
                # Evaluating old actions and values
                logprobs, state_values, dist_entropy = self.policy.evaluate(
                    old_states, old_actions
                )

                # match state_values tensor dimensions with rewards tensor
                state_values = torch.squeeze(state_values)
                # Finding the ratio (pi_theta / pi_theta__old)
                ratios = torch.exp(logprobs - old_logprobs.detach()).mean(
                    dim=[1]
                )

                # Finding Surrogate Loss
                advantages = rewards.view(-1) - state_values.detach()

                surr1 = ratios * advantages
                surr2 = (
                    torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip)
                    * advantages
                )
                # final loss of clipped objective PPO
                loss = (
                    -torch.min(surr1, surr2).mean()
                    + 0.5 * self.MseLoss(state_values, rewards.view(-1))
                    - 0.01 * dist_entropy
                )
                # print(
                #     "loss: {}\navtg: {}\nratio: {}\nreward: {}".format(
                #         loss.data, advantages.data, ratios.data, rewards.data
                #     )
                # )
                # take gradient step
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        # clear buffer
        self.buffer.clear()

    def load(self, checkpoint_path):
        self.policy_old.load_state_dict(
            torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
        )
        self.policy.load_state_dict(
            torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
        )