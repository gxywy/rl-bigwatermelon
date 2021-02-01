import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Implementation of Twin Delayed Deep Deterministic Policy Gradients (TD3)
# Paper: https://arxiv.org/abs/1802.09477

class ActorCritic(torch.nn.Module):
	def __init__(self):
		super(ActorCritic, self).__init__()
		self.conv1 = nn.Conv2d(1, 32, 3, stride=2, padding=1)
		self.conv2 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
		self.conv3 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
		self.conv4 = nn.Conv2d(32, 32, 3, stride=2, padding=1)

		self.l1_a = nn.Linear(800, 256)
		self.l2_a = nn.Linear(256, 256)
		self.l3_a = nn.Linear(256, 1)

		self.l1_c = nn.Linear(800 + 1, 256)
		self.l2_c = nn.Linear(256, 256)
		self.l3_c = nn.Linear(256, 1)

		self.l4_c = nn.Linear(800 + 1, 256)
		self.l5_c = nn.Linear(256, 256)
		self.l6_c = nn.Linear(256, 1)

		self.max_action = 160

	def pi(self, inputs):
		x = F.elu(self.conv1(inputs))
		x = F.elu(self.conv2(x))
		x = F.elu(self.conv3(x))
		x = F.elu(self.conv4(x))

		x = x.view(-1, 800)
		x = F.relu(self.l1_a(x))
		x = F.relu(self.l2_a(x))
		return self.max_action * torch.tanh(self.l3_a(x))
	

	def v(self, state, action):
		x = F.elu(self.conv1(state))
		x = F.elu(self.conv2(x))
		x = F.elu(self.conv3(x))
		x = F.elu(self.conv4(x))

		x = x.view(-1, 800)
		
		sa = torch.cat([x, action], 1)

		q1 = F.relu(self.l1_c(sa))
		q1 = F.relu(self.l2_c(q1))
		q1 = self.l3_c(q1)

		q2 = F.relu(self.l4_c(sa))
		q2 = F.relu(self.l5_c(q2))
		q2 = self.l6_c(q2)
		return q1, q2


	def Q1(self, state, action):
		x = F.elu(self.conv1(state))
		x = F.elu(self.conv2(x))
		x = F.elu(self.conv3(x))
		x = F.elu(self.conv4(x))

		x = x.view(-1, 800)
		
		sa = torch.cat([x, action], 1)

		q1 = F.relu(self.l1_c(sa))
		q1 = F.relu(self.l2_c(q1))
		q1 = self.l3_c(q1)
		return q1

class TD3(object):
	def __init__(
		self,
		state_dim,
		action_dim,
		max_action,
		discount=0.99,
		tau=0.005,
		policy_noise=0.2,
		noise_clip=0.5,
		policy_freq=2
	):

		self.actor_critic = ActorCritic().to(device)
		self.actor_critic_target = copy.deepcopy(self.actor_critic)
		self.optimizer = torch.optim.Adam(self.actor_critic.parameters(), lr=3e-4)

		self.max_action = max_action
		self.discount = discount
		self.tau = tau
		self.policy_noise = policy_noise
		self.noise_clip = noise_clip
		self.policy_freq = policy_freq

		self.total_it = 0


	def select_action(self, state):
		state = torch.FloatTensor(state).to(device)
		return self.actor_critic.pi(state.unsqueeze(0)).cpu().data.numpy().flatten()


	def train(self, replay_buffer, batch_size=100):
		self.total_it += 1

		# Sample replay buffer 
		state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

		with torch.no_grad():
			# Select action according to policy and add clipped noise
			noise = (
				torch.randn_like(action) * self.policy_noise
			).clamp(-self.noise_clip, self.noise_clip)
			
			next_action = (
				self.actor_critic_target.pi(next_state) + noise
			).clamp(-self.max_action / 2, self.max_action / 2) + (self.max_action / 2)

			# Compute the target Q value
			target_Q1, target_Q2 = self.actor_critic_target.v(next_state, next_action)
			target_Q = torch.min(target_Q1, target_Q2)
			target_Q = reward + not_done * self.discount * target_Q

		# Get current Q estimates
		current_Q1, current_Q2 = self.actor_critic.v(state, action)

		# Compute critic loss
		critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)


		# Compute actor losse
		actor_loss = -self.actor_critic.Q1(state, self.actor_critic.pi(state)).mean()
		
		# Optimize the actor 
		self.optimizer.zero_grad()
		actor_loss.backward()
		self.optimizer.step()

		# Update the frozen target models
		for param, target_param in zip(self.actor_critic.parameters(), self.actor_critic_target.parameters()):
			target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)


	def save(self, filename):
		torch.save(self.actor_critic.state_dict(), filename)
		torch.save(self.optimizer.state_dict(), filename + "_optimizer")


	def load(self, filename):
		self.actor_critic.load_state_dict(torch.load(filename))
		self.optimizer.load_state_dict(torch.load(filename + "_optimizer"))
		self.actor_critic_target = copy.deepcopy(self.actor_critic)