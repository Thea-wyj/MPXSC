"""
An agent implementing the Deep Q-Learning algorithm.
"""
import io
import json

import torch
import numpy as np
from PIL import Image
from captum.attr import IntegratedGradients, Lime
from captum.metrics import sensitivity_max
from matplotlib import pyplot as plt
from skimage.segmentation import slic
from sklearn.cluster import DBSCAN
from torchvision.transforms import ToPILImage, Resize, ToTensor

from GohTest.saliency import goh
from MFPP.test_mpx import mpx
from deepQLearning.dql_net import DqlNet
from general.evaluate import get_infidelity, get_sensitivity, get_stability, get_validity, get_sufficiency, \
    mpx_explain_fn, goh_explain_fn, sarfa_explain_fn
from general.general_agent import AtariAgent
from collections import Counter

from sarfa.sarfa_saliency import sarfa


class DQLAtari(AtariAgent):
    """
    This is an agent for the Deep Q-Learning algorithm.
    
    Args:
        action_space: An array containing all actions.
        action_name_list: A list of action names corresponding to the action space.
        memory_par: A tuple including the size of the memory space and multi-frames image size.
        game: A tuple including the game name and the gym environment.
        epsilon: A tuple including the epsilon and minimum epsilon.
        reward_clip: A boolean indicating whether to clip rewards in the range [-1, 1].
    """
    def __init__(self, action_space: np.array, action_name_list: np.array, memory_par: tuple, game: tuple,
                 epsilon: tuple, reward_clip: bool):
        super().__init__(action_space, memory_par, game, reward_clip)
        self.episodes = 2
        self.learn_replace = 5000
        self.epsilon, self.mini_epsilon = epsilon
        self.action_name_list = action_name_list
        self.infidelities = []
        self.sensitivities = []
        self.stabilities = []
        self.validities = []
        self.sufficiencies = []
        self.attributions = []
        with torch.no_grad():
            self.target_net = DqlNet(img_size=self.multi_frames_size, out_channels=self.action_space_len)
        self.behavior_net = DqlNet(img_size=self.multi_frames_size, out_channels=self.action_space_len)

    def format_image(self, img):
        """
        Format the input image to the required size and shape.
        
        Args:
            img: The input image tensor.
        
        Returns:
            The formatted image tensor.
        """
        img = img.permute(2, 0, 1)
        to_pil = ToPILImage()
        img_pil = to_pil(img)

        # Resize the image
        resize = Resize((84, 84))
        img_resized_pil = resize(img_pil)

        # Convert the PIL image back to a tensor
        to_tensor = ToTensor()
        img_resized = to_tensor(img_resized_pil)
        return img_resized.permute(1, 2, 0)

    def predict_action_q(self, state):
        """
        Calculate Q-values for different actions under a certain state.
        
        Args:
            state: The current state tensor.
        
        Returns:
            The Q-values tensor.
        """
        with torch.no_grad():
            return self.behavior_net.forward(state[None, ...].to(self.behavior_net.device))

    def get_action(self, s: torch.Tensor, eval=False) -> int:
        """
        Choose an action under a certain policy using the epsilon-greedy method.
        
        Args:
            s: The current state tensor.
            eval: A boolean indicating whether to evaluate the model only.
        
        Returns:
            An action for the current state under the certain policy.
        """
        action_q = self.predict_action_q(s)
        a = torch.argmax(action_q).item()
        if np.random.rand() < self.epsilon and not eval:
            return self.action_space[np.random.randint(self.action_space_len)]
        else:
            return a

    def update_target(self):
        """
        Update the target network after a certain number of learning steps.
        """
        if self.step_count % self.learn_replace == 0:
            self.target_net.load_state_dict(self.behavior_net.state_dict())

    def update_episode(self):
        """
        Update the epsilon value after each learning episode.
        """
        if self.frames_count > self.explore_frame:
            self.epsilon = self.final_epsilon
        else:
            self.epsilon = self.epsilon - self.epsilon_decay if self.epsilon > self.mini_epsilon else self.mini_epsilon

    def learn(self):
        """
        Learn from memory and update the network.
        """
        if self.step_count < self.learn_start_step or self.step_count % self.learn_interval != 0:
            return

        s, a, r, s_, t = self.sample()
        q_behavior = self.behavior_net.forward(s.to(self.behavior_net.device).float())
        q_behavior_a = q_behavior[np.arange(self.batch_size), np.array(a)]
        q_target = self.target_net.forward(s_.to(self.target_net.device).float())
        q_target_max = torch.max(q_target, dim=1)[0]

        q_target_max[t] = 0.0
        q = r.to(self.behavior_net.device).float() + self.gamma * q_target_max

        self.behavior_net.optimizer.zero_grad()
        self.behavior_net.loss(q, q_behavior_a).backward()
        self.behavior_net.optimizer.step()
        self.learn_cur += 1
        self.update_target()

    def load_model(self, net_path: str, eval=False, start_episodes=0):
        """
        Load an existing model. If in evaluation mode, load the behavior network only.
        
        Args:
            net_path: The path containing all models.
            eval: A boolean indicating whether to evaluate only.
            start_episodes: The number of the start episode.
        """
        if eval:
            self.behavior_net.load_state_dict(torch.load(net_path + '/behavior.pth'))
            self.behavior_net.eval()
        if start_episodes != 0 and not eval:
            self.behavior_net.load_state_dict(torch.load(net_path + '/behavior{}.pth'.format(start_episodes)))
            self.target_net.load_state_dict(torch.load(net_path + '/target{}.pth'.format(start_episodes)))
            self.behavior_net.optimizer.load_state_dict(torch.load(net_path + '/optimizer{}.pth'.format(start_episodes)))
            self.scores = np.load(net_path + '/scores{}.npy'.format(start_episodes))
            self.learn_cur += 1

    def check_image(self, img):
        """
        Check if the image is valid and display it.
        
        Args:
            img: The image tensor to be checked.
        """
        try:
            plt.imshow(img)
            print("Image is valid")
        except Exception as e:
            print("Image might be corrupted, error message:")
            print(e)

    def process_results(self, episode, eval):
        """
        Save models and plot results after certain episodes.
        
        Args:
            episode: The current episode number.
            eval: A boolean indicating whether to evaluate only.
        """
        print('Episodes: {}, AveScores: {}, Epsilon: {}, Steps: {}'.format(
            episode + 1, self.scores[episode], self.epsilon, self.step_count))
        # if episode % 10 == 9:
        #     ave = np.mean(self.scores[episode - 9:episode])
        #     print('Episodes: {}, AveScores: {}, Epsilon: {}, Steps: {}'.format(
        #         episode + 1, ave, self.epsilon, self.step_count))
        # if eval:
        #     if episode % 100 == 99:
        #         s1 = './' + self.game_name + '/'
        #         np.save(s1 + 'scores_eval{}.npy'.format(episode + 1), self.scores)
        #         print('Evaluation results saved!')
        # else:
        #     if episode % 200 == 199:
        #         s1 = './' + self.game_name + '/'
        #         s_pth = '{}.pth'.format(episode + 1)
        #         torch.save(self.behavior_net.state_dict(), s1 + self.game_name + s_pth)
        #         torch.save(self.target_net.state_dict(), s1 + 'target' + s_pth)
        #         torch.save(self.behavior_net.optimizer.state_dict(), s1 + 'optimizer' + s_pth)
        #         np.save(s1 + 'scores{}.npy'.format(episode + 1), self.scores)
        #
        #         self.plot_array(episode)
        #         print('Model salved!')
        #         print('Total {} frames!'.format(self.frames_count))

    def explain_by_mpx(self):
        """
        Explain the model's decisions using mpx （Local explanation in Chapter 3.2）
        """
        self.infidelities = []
        self.sensitivities = []
        self.stabilities = []
        self.validities = []
        self.sufficiencies = []
        self.attributions = []
        anti_attribution_list = []
        sensitivity_process_list = []
        for index, input in enumerate(self.max_input_list):
            obs_input = self.max_obs_list[index]
            input = input.unsqueeze(0)  # 1 4 84 84
            obs_input = obs_input.unsqueeze(0)  # 1 210 160 3
            out = self.max_output_list[index]
            reward, _ = torch.max(out.data, 1)
            action = self.max_action_list[index]
            min_val, min_idx = torch.min(out, dim=1)

            attribution = mpx(self.behavior_net, obs_input, input, action.data.item())  # 1 210 160 3
            anti_attribution = mpx(self.behavior_net, obs_input, input, min_idx.data.item())  # 1 4 84 84
            sensitivity_process = sensitivity_max(mpx_explain_fn, input, model=self.behavior_net, img_file=obs_input, target=action.data.item())

            if attribution.max() > 0:
                self.attributions.append(attribution)
                anti_attribution_list.append(anti_attribution)
                sensitivity_process_list.append(sensitivity_process)

                plt.figure()  # Create a new figure window
                plt.imshow(self.max_obs_list[index].cpu().detach().numpy())
                plt.savefig("output/" + self.game_name + "/mpx/original-img/" + str(index) + ".jpg")
                print("output/" + self.game_name + "/mpx/original-img/" + str(index) + ".jpg is saved")
                mean_val = attribution.clamp(min=0).mean().data.item()
                max_val = attribution.clamp(min=0).max().data.item()
                plt.imshow(attribution.clamp(min=0.7 * max_val).squeeze().mean(dim=-1).cpu().detach().numpy(), cmap='hot')
                plt.text(0, -5, 'Original Image predict:{}'.format(self.action_name_list[action.data.item()]), fontsize=12, color='white')
                plt.savefig("output/" + self.game_name + "/mpx/heat/" + str(index) + ".jpg")
                plt.clf()  # Clear the current figure window

        self.evaluate_attribution(anti_attribution_list, sensitivity_process_list)
        data = {
            "infidelity": sum(self.infidelities) / len(self.infidelities),
            "sensitivity": sum(self.sensitivities) / len(self.sensitivities),
            "stability": sum(self.stabilities) / len(self.stabilities),
            "validity": sum(self.validities) / len(self.validities),
            "sufficiency": sum(self.sufficiencies) / len(self.sufficiencies),
        }
        with open("output/" + self.game_name + "/mpx/" + "mpx" + ".json", "w") as f:
            json.dump(data, f)
        print("output/" + self.game_name + "/mpx/" + "mpx" + ".json is saved")

    def explain_by_goh(self):
        """
        Explain the model's decisions using GOH (Gradient-based Occlusion Heatmap).
        """
        self.infidelities = []
        self.sensitivities = []
        self.stabilities = []
        self.validities = []
        self.sufficiencies = []
        self.attributions = []
        anti_attribution_list = []
        sensitivity_process_list = []
        for index, input in enumerate(self.max_input_list):
            obs_input = self.max_obs_list[index]
            input = input.unsqueeze(0)  # 1 4 84 84
            obs_input = obs_input.unsqueeze(0)  # 1 210 160 3
            out = self.max_output_list[index]
            reward, _ = torch.max(out.data, 1)
            action = self.max_action_list[index]
            # goh
            attribution = goh(self.behavior_net, input, obs_input)
            sensitivity_process = sensitivity_max(goh_explain_fn, input, model=self.behavior_net, img_file=obs_input)

            if attribution.max() > 0:
                self.attributions.append(attribution)
                sensitivity_process_list.append(sensitivity_process)
                mean_val = attribution.clamp(min=0).mean().data.item()
                max_val = attribution.clamp(min=0).max().data.item()
                plt.figure()  # Create a new figure window
                plt.imshow(self.max_obs_list[index].cpu().detach().numpy())
                plt.savefig("output/" + self.game_name + "/goh/original-img/" + str(index) + ".jpg")
                print("output/" + self.game_name + "/goh/original-img/" + str(index) + ".jpg is saved")
                plt.imshow(attribution.clamp(min=0.7 * max_val).squeeze().mean(dim=-1).cpu().detach().numpy(), cmap='hot')
                plt.text(0, -5, 'Original Image predict:{}'.format(self.action_name_list[action.data.item()]), fontsize=12, color='white')
                plt.savefig("output/" + self.game_name + "/goh/heat/" + str(index) + ".jpg")
                plt.clf()  # Clear the current figure window

        self.evaluate_attribution(anti_attribution_list, sensitivity_process_list)
        data = {
            "infidelity": sum(self.infidelities) / len(self.infidelities),
            "stability": sum(self.stabilities) / len(self.stabilities),
            "validity": sum(self.validities) / len(self.validities),
            "sufficiency": sum(self.sufficiencies) / len(self.sufficiencies),
        }
        with open("output/" + self.game_name + "/goh/" + "goh" + ".json", "w") as f:
            json.dump(data, f)
        print("output/" + self.game_name + "/goh/" + "goh" + ".json is saved")

    def explain_by_sarfa(self):
        """
        Explain the model's decisions using SARFA (Saliency-based Feature Perturbation).
        """
        # Initialize lists to store various metrics and attributions
        self.infidelities = []
        self.sensitivities = []
        self.stabilities = []
        self.validities = []
        self.sufficiencies = []
        self.attributions = []
        anti_attribution_list = []
        sensitivity_process_list = []

        # Iterate over the input list to compute attributions
        for index, input in enumerate(self.max_input_list):
            obs_input = self.max_obs_list[index]
            input = input.unsqueeze(0)  # Reshape input to (1, 4, 84, 84)
            obs_input = obs_input.unsqueeze(0)  # Reshape obs_input to (1, 210, 160, 3)
            out = self.max_output_list[index]
            reward, _ = torch.max(out.data, 1)
            action = self.max_action_list[index]
            min_val, min_idx = torch.min(out, dim=1)

            # Compute SARFA attributions
            attribution = sarfa(self.behavior_net, input, obs_input, action.data.item())
            anti_attribution = sarfa(self.behavior_net, input, obs_input, min_idx.data.item())
            sensitivity_process = sensitivity_max(sarfa_explain_fn, input, model=self.behavior_net,
                                                  target=action.data.item(), img_file=obs_input)

            if attribution.max() > 0:
                self.attributions.append(attribution)
                anti_attribution_list.append(anti_attribution)
                sensitivity_process_list.append(sensitivity_process)
                mean_val = attribution.clamp(min=0).mean().data.item()
                max_val = attribution.clamp(min=0).max().data.item()
                plt.figure()  # Create a new figure window
                plt.imshow(self.max_obs_list[index].cpu().detach().numpy())
                plt.savefig("output/" + self.game_name + "/sarfa/original-img/" + str(index) + ".jpg")
                print("output/" + self.game_name + "/sarfa/original-img/" + str(index) + ".jpg is saved")
                plt.imshow(attribution.clamp(min=0.7 * max_val).squeeze().mean(dim=-1).cpu().detach().numpy(),
                           alpha=0.5)
                plt.text(0, -5, 'Original Image predict:{}'.format(self.action_name_list[action.data.item()]),
                         color='black')
                plt.savefig("output/" + self.game_name + "/sarfa/heat/" + str(index) + ".jpg")
                plt.clf()  # Clear the current figure window
                print("output/" + self.game_name + "/sarfa/heat/" + str(index) + ".jpg is saved")

        # Evaluate the attributions
        self.evaluate_attribution(anti_attribution_list, sensitivity_process_list)
        data = {
            "infidelity": sum(self.infidelities) / len(self.infidelities),
            "sensitivity": sum(self.sensitivities) / len(self.sensitivities),
            "stability": sum(self.stabilities) / len(self.stabilities),
            "validity": sum(self.validities) / len(self.validities),
            "sufficiency": sum(self.sufficiencies) / len(self.sufficiencies),
        }
        with open("output/" + self.game_name + "/sarfa/" + "sarfa" + ".json", "w") as f:
            json.dump(data, f)
        print("output/" + self.game_name + "/sarfa/" + "sarfa" + ".json is saved")

    def global_explain_2(self):
        """
        Categorize heatmaps by action and original image, and perform clustering using DBSCAN. （Global explanation in Chapter 3.3）
        """
        # Initialize lists to store attributions, observations, outputs, and actions for each action
        action_attr_list = [[] for _ in range(len(self.action_name_list))]
        action_obs_list = [[] for _ in range(len(self.action_name_list))]
        action_output_list = [[] for _ in range(len(self.action_name_list))]
        action_real_list = [[] for _ in range(len(self.action_name_list))]

        # Populate the lists with corresponding data
        for index, attr in enumerate(self.attributions):
            reward, _ = torch.max(self.max_output_list[index].data, 1)
            action = self.max_action_list[index]
            action_attr_list[action.data.item()].append(attr)
        for index, obs in enumerate(self.max_obs_list):
            reward, _ = torch.max(self.max_output_list[index].data, 1)
            action = self.max_action_list[index]
            action_obs_list[action.data.item()].append(obs)
        for index, output in enumerate(self.max_output_list):
            reward, _ = torch.max(self.max_output_list[index].data, 1)
            action = self.max_action_list[index]
            action_output_list[action.data.item()].append(output)
        for index, output in enumerate(self.max_output_list):
            reward, _ = torch.max(self.max_output_list[index].data, 1)
            action = self.max_action_list[index]
            action_real_list[action.data.item()].append(action)

        # Define DBSCAN epsilon values for each action
        dbscan_eps_list = [10.0, 10.0, 14.0, 8.0]

        # Perform clustering for each action
        for action, attr_list in enumerate(action_attr_list):
            # Get the corresponding observations, outputs, and real actions
            obs_list = action_obs_list[action]
            output_list = action_output_list[action]
            real_list = action_real_list[action]

            # Flatten the attributions for clustering
            data = [attribution.mean(dim=1).squeeze().cpu().detach().numpy().flatten() for attribution in attr_list]
            if len(data) > 0:
                # Convert the list to a numpy array
                data = np.array(data)
                # Perform DBSCAN clustering
                dbscan = DBSCAN(eps=7, min_samples=3).fit(data)
                labels = dbscan.labels_

                # Calculate the number of clusters
                n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                print(action, n_clusters)

                # Iterate over each cluster label
                for label in set(labels):
                    if label != -1:  # Ignore noise points labeled as -1
                        mask = (labels == label)
                        max_diff_item = 0
                        max_diff_index = 0
                        max_diff_attr = None
                        obs_label_list = [obs_list[i] for i in range(len(obs_list)) if mask[i]]
                        # Find the attribution with the maximum difference in reward
                        for index, attr in enumerate([attr_list[i] for i in range(len(attr_list)) if mask[i]]):
                            plt.figure()  # Create a new figure
                            reward, _ = torch.max(output_list[index].data, 1)
                            action = real_list[index]
                            min_reward, min_action = torch.min(output_list[index].data, 1)
                            if reward.data.item() - min_reward.data.item() > max_diff_item:
                                max_diff_item = reward.data.item() - min_reward.data.item()
                                max_diff_index = index
                                max_diff_attr = attr
                        # Plot and save the heatmap
                        plt.imshow(obs_label_list[max_diff_index].squeeze().cpu().detach().numpy())
                        max_diff_attr.clamp(min=0).mean().data.item()
                        max_val = max_diff_attr.clamp(min=0).max().data.item()
                        plt.imshow(max_diff_attr.clamp(min=0.7 * max_val).squeeze().mean(dim=-1).cpu().detach().numpy(),
                                   alpha=0.5)
                        plt.text(0, -5,
                                 'Original Image predict:{},label is {}'.format(self.action_name_list[action], label),
                                 color='black')
                        plt.savefig(
                            "output/" + self.game_name + "/mpx/action-group-heat/" + self.action_name_list[
                                action] + "/" + str(label) + ".jpg")
                        print("output/" + self.game_name + "/mpx/action-group-heat/" + self.action_name_list[
                            action] + "/" + str(label) + ".jpg is saved")
                        plt.clf()  # Clear the current figure
            else:
                print("action", self.action_name_list[action], "has no data")

    def evaluate_attribution(self, anti_attribution_list, sensitivity_process_list):
        """
        Evaluate the attributions by calculating various metrics such as infidelity, sensitivity, stability, validity, and sufficiency. in Chapter 4.1

        Args:
            anti_attribution_list: List of anti-attributions for sensitivity calculation.
            sensitivity_process_list: List of sensitivity processes for stability calculation.
        """
        for index, attribution in enumerate(self.attributions):
            # Calculate the maximum value of the attribution
            max_val = torch.max(attribution)
            threshold = 0 * max_val.item()
            reward, _ = torch.max(self.max_output_list[index].data, 1)
            action = self.max_action_list[index]

            # Calculate infidelity
            self.infidelities.append(
                get_infidelity(self.behavior_net, self.max_input_list[index].unsqueeze(0), attribution,
                               action).data.item()
            )

            # Calculate sensitivity
            if len(anti_attribution_list) > 0:
                sensitivity = get_sensitivity(attribution, anti_attribution_list[index])
                if not torch.isnan(sensitivity):
                    self.sensitivities.append(sensitivity.data.item())

            # Calculate stability
            self.stabilities.append(get_stability(sensitivity_process_list[index]).data.item())

            # Calculate validity
            self.validities.append(
                get_validity(attribution, threshold, self.behavior_net, self.max_input_list[index].unsqueeze(0), action,
                             reward).data.item()
            )

            # Calculate sufficiency
            self.sufficiencies.append(
                get_sufficiency(attribution, threshold, self.behavior_net, self.max_input_list[index].unsqueeze(0),
                                action, reward).data.item()
            )
