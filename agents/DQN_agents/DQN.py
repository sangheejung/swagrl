from collections import Counter
import copy
import time
import torch
import random
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from agents.Base_Agent import Base_Agent
from exploration_strategies.Epsilon_Greedy_Exploration import Epsilon_Greedy_Exploration
from utilities.data_structures.Replay_Buffer import Replay_Buffer
from agents.DQN_agents.SWAG_DQN import SWAG

class DQN(Base_Agent):
    """A deep Q learning agent"""
    agent_name = "DQN"
    def __init__(self, config):
        Base_Agent.__init__(self, config)
        self.memory = Replay_Buffer(self.hyperparameters["buffer_size"], self.hyperparameters["batch_size"], config.seed, self.device)
        self.q_network_local = self.create_NN(input_dim=self.state_size, output_dim=self.action_size)
        self.q_network_optimizer = optim.Adam(self.q_network_local.parameters(),
                                              lr=self.hyperparameters["learning_rate"], eps=1e-4)
        self.exploration_strategy = Epsilon_Greedy_Exploration(config)
        
        self.collect_freq=self.hyperparameters["collect_freq"]
        self.sample_freq=self.hyperparameters["sample_freq"]
        self.collect_start=self.hyperparameters["collect_start"]
        self.sample_start=self.hyperparameters["sample_start"]
        
        self.epi_scores=[[],[],[]]
        
        self.q_swa=self.create_NN(input_dim=self.state_size, output_dim=self.action_size)
        self.q_swag=self.create_NN(input_dim=self.state_size, output_dim=self.action_size)
        self.swag_agent=SWAG(self.q_network_local)
        

    def run_n_episodes(self, num_episodes=None, show_whether_achieved_goal=True, save_and_print_results=True):
        """Runs game to completion n times and then summarises results and saves model (if asked to)"""
        if num_episodes is None: num_episodes = self.config.num_episodes_to_run
        start = time.time()
        while self.episode_number < num_episodes:
            self.reset_game()
            self.step()
            for i in range(3):
                if i==0:
                    self.epi_scores[i].append(self.total_episode_score_so_far)
                else:
                    self.epi_scores[i].append(self.other_scores[i-1])
        time_taken = time.time() - start
        if show_whether_achieved_goal: self.show_whether_achieved_goal()
        if self.config.save_model: self.locally_save_policy()
        return self.epi_scores, None, time_taken


    def update_learning_rate(self, starting_lr, optimizer):

        t = (self.episode_number) / self.config.num_episodes_to_run
        lr_ratio = 0.01 #self.swag_lr / starting_lr
        if t <= 0.5:
            factor = 1.0
        elif t <= 0.9:
            factor = 1.0 - (1.0 - lr_ratio) * (t - 0.5) / 0.4
        else:
            factor = lr_ratio
        new_lr=starting_lr * factor
        for g in optimizer.param_groups:
            g['lr'] = new_lr
        if random.random() < 0.001: self.logger.info("Learning rate {}".format(new_lr))


    def reset_game(self):
        """Resets the game information so we are ready to play a new episode"""
        self.environment[0].seed(self.config.seed)
        s = self.environment[0].reset()
        self.environment[1]=copy.deepcopy(self.environment[0])
        self.environment[2]=copy.deepcopy(self.environment[0])
        self.state=[s, copy.deepcopy(s), copy.deepcopy(s)]
        self.next_state = [None, None, None]
        self.actions = [None, None, None]
        self.reward = [None, None, None]
        self.done = [False, False, False]
        self.total_episode_score_so_far = 0
        self.other_scores=[0, 0]
        self.episode_states = []
        self.episode_rewards = []
        self.episode_actions = []
        self.episode_next_states = []
        self.episode_dones = []
        self.episode_desired_goals = []
        self.episode_achieved_goals = []
        self.episode_observations = []
        if "exploration_strategy" in self.__dict__.keys(): self.exploration_strategy.reset()
        self.logger.info("Reseting game -- New start state {}".format(self.state))

        self.update_learning_rate(self.hyperparameters["learning_rate"], self.q_network_optimizer)

    def conduct_action(self, actions):
        """Conducts an action in the environment"""
        for i in range(3):
            if not self.done[i]:
                self.next_state[i], self.reward[i], self.done[i], _ = self.environment[i].step(actions[i])
                if i==0:
                    self.total_episode_score_so_far+= self.reward[i]
                else:
                    self.other_scores[i-1]+=self.reward[i]
                if self.hyperparameters["clip_rewards"]: self.reward[i] =  max(min(self.reward[i], 1.0), -1.0)
            else:
                self.next_state[i]=None

    def save_experience(self, memory=None, experience=None):
        """Saves the recent experience to the memory buffer"""
        if memory is None: memory = self.memory
        if experience is None: experience = self.state[0], self.actions[0], self.reward[0], self.next_state[0], self.done[0]
        memory.add_experience(*experience)

    def step(self):
        """Runs a step within a game including a learning step if required"""
        while not all(self.done):
            if self.episode_number<self.collect_start:
                a = self.pick_action()
                self.actions=[a, copy.deepcopy(a), copy.deepcopy(a)]
            elif self.episode_number>=self.collect_start and self.episode_number<self.sample_start:
                self.actions=[self.pick_action(),self.pick_swa_action(),self.pick_action()]
            else:
                self.actions=[self.pick_action(),self.pick_swa_action(),self.pick_swag_action()]

            self.conduct_action(self.actions)
            if self.time_for_q_network_to_learn():
                for _ in range(self.hyperparameters["learning_iterations"]):
                    self.learn()
            if not self.done[0]:
                self.save_experience()
            for i in range(3):
                self.state[i] = self.next_state[i] #this is to set the state for the next iteration
            self.global_step_number += 1
        self.episode_number += 1
        if (self.episode_number>self.collect_start) and (self.episode_number%self.collect_freq==0):
            self.swag_agent.collect_model(self.q_network_local)
            self.swag_agent.sample(self.q_swa,add_swag=False)
        if (self.episode_number>self.sample_start) and (self.episode_number%self.sample_freq==0):
            self.swag_agent.sample(self.q_swag)        
        if self.episode_number%10==0:
            print("EPISODE ",self.episode_number, self.total_episode_score_so_far,self.other_scores)
        self.other_scores=[0,0]

    def pick_action(self, state=None):
        """Uses the local Q network and an epsilon greedy policy to pick an action"""
        # PyTorch only accepts mini-batches and not single observations so we have to use unsqueeze to add
        # a "fake" dimension to make it a mini-batch rather than a single observation
        if state is None: state = self.state[0]
        if state is None: return
        if isinstance(state, np.int64) or isinstance(state, int): state = np.array([state])
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        if len(state.shape) < 2: state = state.unsqueeze(0)
        self.q_network_local.eval() #puts network in evaluation mode
        with torch.no_grad():
            action_values = self.q_network_local(state)

        self.q_network_local.train() #puts network back in training mode
        action = self.exploration_strategy.perturb_action_for_exploration_purposes({"action_values": action_values,
                                                                                    "turn_off_exploration": self.turn_off_exploration,
                                                                                    "episode_number": self.episode_number})
        self.logger.info("Q values {} -- Action chosen {}".format(action_values, action))
        return action
    
    def pick_swa_action(self, state=None):
        if state is None: state = self.state[1]
        if state is None: return
        if isinstance(state, np.int64) or isinstance(state, int): state = np.array([state])
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        if len(state.shape) < 2: state = state.unsqueeze(0)
        self.q_swa.eval() #puts network in evaluation mode
        with torch.no_grad():
            action_values = self.q_swa(state)
        return torch.argmax(action_values).item()

    def pick_swag_action(self, state=None):
        if state is None: state = self.state[2]
        if state is None: return
        if isinstance(state, np.int64) or isinstance(state, int): state = np.array([state])
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        if len(state.shape) < 2: state = state.unsqueeze(0)
        self.q_swag.eval() #puts network in evaluation mode
        with torch.no_grad():
            action_values = self.q_swag(state)
        return torch.argmax(action_values).item()

    def learn(self, experiences=None):
        """Runs a learning iteration for the Q network"""
        if experiences is None: states, actions, rewards, next_states, dones = self.sample_experiences() #Sample experiences
        else: states, actions, rewards, next_states, dones = experiences
        loss = self.compute_loss(states, next_states, rewards, actions, dones)
        
        actions_list = [action_X.item() for action_X in actions ]
        self.logger.info("Action counts {}".format(Counter(actions_list)))
        self.take_optimisation_step(self.q_network_optimizer, self.q_network_local, loss, self.hyperparameters["gradient_clipping_norm"])

    def compute_loss(self, states, next_states, rewards, actions, dones):
        """Computes the loss required to train the Q network"""
        with torch.no_grad():
            Q_targets = self.compute_q_targets(next_states, rewards, dones)
        Q_expected = self.compute_expected_q_values(states, actions)
        loss = F.mse_loss(Q_expected, Q_targets)
        return loss

    def compute_q_targets(self, next_states, rewards, dones):
        """Computes the q_targets we will compare to predicted q values to create the loss to train the Q network"""
        Q_targets_next = self.compute_q_values_for_next_states(next_states)
        Q_targets = self.compute_q_values_for_current_states(rewards, Q_targets_next, dones)
        return Q_targets

    def compute_q_values_for_next_states(self, next_states):
        """Computes the q_values for next state we will use to create the loss to train the Q network"""
        Q_targets_next = self.q_network_local(next_states).detach().max(1)[0].unsqueeze(1)
        return Q_targets_next

    def compute_q_values_for_current_states(self, rewards, Q_targets_next, dones):
        """Computes the q_values for current state we will use to create the loss to train the Q network"""
        Q_targets_current = rewards + (self.hyperparameters["discount_rate"] * Q_targets_next * (1 - dones))
        return Q_targets_current

    def compute_expected_q_values(self, states, actions):
        """Computes the expected q_values we will use to create the loss to train the Q network"""
        Q_expected = self.q_network_local(states).gather(1, actions.long()) #must convert actions to long so can be used as index
        return Q_expected

    def locally_save_policy(self):
        """Saves the policy"""
        torch.save(self.q_network_local.state_dict(), "Models/{}_local_network.pt".format(self.agent_name))

    def time_for_q_network_to_learn(self):
        """Returns boolean indicating whether enough steps have been taken for learning to begin and there are
        enough experiences in the replay buffer to learn from"""
        return self.right_amount_of_steps_taken() and self.enough_experiences_to_learn_from()

    def right_amount_of_steps_taken(self):
        """Returns boolean indicating whether enough steps have been taken for learning to begin"""
        return self.global_step_number % self.hyperparameters["update_every_n_steps"] == 0

    def sample_experiences(self):
        """Draws a random sample of experience from the memory buffer"""
        experiences = self.memory.sample()
        states, actions, rewards, next_states, dones = experiences
        return states, actions, rewards, next_states, dones
