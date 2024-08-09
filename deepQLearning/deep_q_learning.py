"""
An agent implement Deep Q-Learning algorithm.
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
from MFPP.test_sarfa import mfpp_sarfa
from deepQLearning.dql_net import DqlNet
from general.evaluate import get_infidelity, get_sensitivity, get_stability, get_validity, get_sufficiency, \
    mfpp_sarfa_explain_fn, goh_explain_fn, sarfa_explain_fn
from general.general_agent import AtariAgent
from collections import Counter

from sarfa.sarfa_saliency import sarfa


class DQLAtari(AtariAgent):
    def __init__(self, action_space: np.array, action_name_list: np.array, memory_par: tuple, game: tuple,
                 epsilon: tuple, reward_clip: bool):
        """
        This is an agent for Deep Q-Learning algorithm.

        Args:
            action_space: An array contains all actions.
            memory_par: Including the size of the memory space and multi-frames image size.
            game: Including the game name and the gym environment.
            epsilon: Includes the epsilon and minimum epsilon.
            reward_clip: Clip reward in [-1, 1] range if True.
        """
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
        img = img.permute(2, 0, 1)
        to_pil = ToPILImage()
        img_pil = to_pil(img)

        # 然后，我们可以使用Resize函数来改变图像的大小
        resize = Resize((84, 84))
        img_resized_pil = resize(img_pil)

        # 最后，我们可以将PIL图像转换回Tensor
        to_tensor = ToTensor()
        img_resized = to_tensor(img_resized_pil)
        # img_resized现在是一个84x84x3的Tensor
        return img_resized.permute(1, 2, 0)

    def predict_action_q(self, state):
        """
        Calculate q values about different actions under certain state.
        """
        with torch.no_grad():
            return self.behavior_net.forward(state[None, ...].to(self.behavior_net.device))

    def get_action(self, s: torch.Tensor, eval=False) -> int:
        """
        Choose action under certain policy with epsilon-greedy method.

        Returns:
            An action for current state under certain policy.
        """
        action_q = self.predict_action_q(s)
        a = torch.argmax(action_q).item()
        if np.random.rand() < self.epsilon and not eval:
            return self.action_space[np.random.randint(self.action_space_len)]
        else:
            return a

    def update_target(self):
        """
        Update the target network under certain learning times.
        """
        if self.step_count % self.learn_replace == 0:
            self.target_net.load_state_dict(self.behavior_net.state_dict())

    def update_episode(self):
        """
        Update the epsilon after each learning.
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
        Load existent model. If in evaluate mode then load behavior network only.

        Args:
            net_path: The path that contains all of models.
            eval: True represents evaluate only.
            start_episodes: The num of the start episode.
        """
        if eval:
            self.behavior_net.load_state_dict(
                torch.load(net_path + '/' + self.game_name + '.pth'))
            self.behavior_net.eval()
        if start_episodes != 0 and not eval:
            self.behavior_net.load_state_dict(
                torch.load(net_path + '/' + self.game_name + '{}.pth'.format(start_episodes)))
            self.target_net.load_state_dict(torch.load(net_path + '/target{}.pth'.format(start_episodes)))
            self.behavior_net.optimizer.load_state_dict(
                torch.load(net_path + '/optimizer{}.pth'.format(start_episodes)))
            self.scores = np.load(net_path + '/scores{}.npy'.format(start_episodes))
            self.learn_cur += 1

    def check_image(self, img):
        try:
            # 尝试显示图像
            plt.imshow(img)
            print("图像正常")
        except Exception as e:
            print("图像可能已损坏，错误信息如下：")
            print(e)

    def process_results(self, episode, eval):
        """
        Salve models and plot results after certain episodes.
        input: self.multi_frames tensor (4,84,84)
        output: self.predict_action_q(self.multi_frames) tensor (1,9)
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

    def explain_by_ig(self):
        self.infidelities = []
        self.sensitivities = []
        self.stabilities = []
        self.validities = []
        self.sufficiencies = []
        self.attributions = []
        ig = IntegratedGradients(self.behavior_net)
        for index, input in enumerate(self.max_input_list):
            input = input.unsqueeze(0)
            out = self.max_output_list[index]
            reward, action = torch.max(out.data, 1)
            min_val, min_idx = torch.min(out, dim=1)

            attribution = ig.attribute(input, target=action.data.item(), n_steps=50).mean(dim=1).repeat(1, 4, 1, 1)
            anti_attribution = ig.attribute(input, target=min_idx.data.item(), n_steps=50).mean(dim=1).repeat(1, 4, 1,
                                                                                                              1)  # n_steps：近似法所用的步数
            sensitivity_process = sensitivity_max(ig.attribute, input, target=action.data.item(), n_steps=50)

            if attribution.max() > 0:
                self.attributions.append(attribution)
                plt.figure()  # 创建一个新的图像窗口
                # occ (4,84,84)
                plt.imshow(input.squeeze()[-1, :, :].cpu().detach().numpy())
                plt.imshow((attribution.clamp(min=0).mean(dim=1).squeeze().cpu().detach().numpy()), alpha=0.5,
                           cmap='rainbow')
                plt.text(0, -5, 'Original Image predict:{}'.format(self.action_name_list[action.data.item()]),
                         color='black')
                plt.savefig("output/ig/" + str(index) + ".jpg")
                plt.clf()  # 清除当前图像窗口
                print("output/ig/" + str(index) + ".jpg is saved")

                # 计算各指标
                max_val = torch.max(attribution)
                threshold = 0 * max_val.item()
                # infidelity
                self.infidelities.append(get_infidelity(self.behavior_net, input, attribution, action).data.item())
                # sensitivity
                sensitivity = get_sensitivity(attribution, anti_attribution)
                if not torch.isnan(sensitivity):
                    self.sensitivities.append(sensitivity.data.item())
                # stability
                self.stabilities.append(get_stability(sensitivity_process).data.item())
                # validity
                self.validities.append(
                    get_validity(attribution, threshold, self.behavior_net, input, action, reward).data.item())
                # sufficiency
                self.sufficiencies.append(
                    get_sufficiency(attribution, threshold, self.behavior_net, input, action, reward).data.item())
            data = {
                "infidelity": sum(self.infidelities) / len(self.infidelities),
                "sensitivity": sum(self.sensitivities) / len(self.sensitivities),
                "stability": sum(self.stabilities) / len(self.stabilities),
                "validity": sum(self.validities) / len(self.validities),
                "sufficiency": sum(self.sufficiencies) / len(self.sufficiencies),
            }
            with open("output/mfpp_sarfa/" + "mfpp_sarfa" + ".json", "w") as f:
                json.dump(data, f)

    def explain_by_lime(self):
        self.infidelities = []
        self.sensitivities = []
        self.stabilities = []
        self.validities = []
        self.sufficiencies = []
        self.attributions = []
        lime = Lime(self.behavior_net)
        for index, input in enumerate(self.max_input_list):
            input = input.unsqueeze(0)
            out = self.max_output_list[index]
            reward, action = torch.max(out.data, 1)
            min_val, min_idx = torch.min(out, dim=1)

            test = input.permute([0, 2, 3, 1]).squeeze(0).cpu().numpy()
            segments = slic(test, n_segments=800, sigma=5)
            segments = torch.tensor(segments).to(self.device)
            attribution = lime.attribute(input, target=action.data.item(), n_samples=8000, feature_mask=segments)
            anti_attribution = lime.attribute(input, target=min_idx.data.item(), n_samples=8000,
                                              feature_mask=segments)
            sensitivity_process = sensitivity_max(lime.attribute, input, target=action.data.item(),
                                                  n_samples=8000,
                                                  feature_mask=segments)

            if attribution.max() > 0:
                self.attributions.append(attribution)
                plt.figure()  # 创建一个新的图像窗口
                # occ (4,84,84)
                plt.imshow(input.squeeze()[-1, :, :].cpu().detach().numpy())
                plt.imshow((attribution.clamp(min=0).mean(dim=1).squeeze().cpu().detach().numpy()), alpha=0.5,
                           cmap='rainbow')
                plt.text(0, -5, 'Original Image predict:{}'.format(self.action_name_list[action.data.item()]),
                         color='black')
                plt.savefig("output/lime/" + str(index) + ".jpg")
                plt.clf()  # 清除当前图像窗口
                print("output/lime/" + str(index) + ".jpg is saved")

                # 计算各指标
                max_val = torch.max(attribution)
                threshold = 0 * max_val.item()
                # infidelity
                self.infidelities.append(get_infidelity(self.behavior_net, input, attribution, action).data.item())
                # sensitivity
                sensitivity = get_sensitivity(attribution, anti_attribution)
                if not torch.isnan(sensitivity):
                    self.sensitivities.append(sensitivity.data.item())
                # stability
                self.stabilities.append(get_stability(sensitivity_process).data.item())
                # validity
                self.validities.append(
                    get_validity(attribution, threshold, self.behavior_net, input, action, reward).data.item())
                # sufficiency
                self.sufficiencies.append(
                    get_sufficiency(attribution, threshold, self.behavior_net, input, action, reward).data.item())
        data = {
            "infidelity": sum(self.infidelities) / len(self.infidelities),
            "sensitivity": sum(self.sensitivities) / len(self.sensitivities),
            "stability": sum(self.stabilities) / len(self.stabilities),
            "validity": sum(self.validities) / len(self.validities),
            "sufficiency": sum(self.sufficiencies) / len(self.sufficiencies),
        }
        with open("output/mfpp_sarfa/" + "mfpp_sarfa" + ".json", "w") as f:
            json.dump(data, f)

    def explain_by_mfpp_sarfa(self):
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

            attribution = mfpp_sarfa(self.behavior_net, obs_input, input,
                                     action.data.item())  # 1 210 160 3
            anti_attribution = mfpp_sarfa(self.behavior_net, obs_input, input,
                                          min_idx.data.item())  # 1 4 84 84
            sensitivity_process = sensitivity_max(mfpp_sarfa_explain_fn, input, model=self.behavior_net,
                                                  img_file=obs_input,
                                                  target=action.data.item())

            if attribution.max() > 0:
                self.attributions.append(attribution)

                anti_attribution_list.append(anti_attribution)
                sensitivity_process_list.append(sensitivity_process)

                plt.figure()  # 创建一个新的图像窗口
                # occ (4,84,84)
                plt.imshow(self.max_obs_list[index].cpu().detach().numpy())
                plt.savefig("output/" + self.game_name + "/mfpp_sarfa/original-img/" + str(index) + ".jpg")
                print("output/" + self.game_name + "/mfpp_sarfa/original-img/" + str(index) + ".jpg is saved")
                mean_val = attribution.clamp(min=0).mean().data.item()
                max_val = attribution.clamp(min=0).max().data.item()
                plt.imshow(attribution.clamp(min=0.7 * max_val).squeeze().mean(dim=-1).cpu().detach().numpy(),
                           alpha=0.5)
                plt.text(0, -5, 'Original Image predict:{}'.format(self.action_name_list[action.data.item()]),
                         color='black')
                plt.savefig("output/" + self.game_name + "/mfpp_sarfa/heat/" + str(index) + ".jpg")
                plt.clf()  # 清除当前图像窗口
                print("output/" + self.game_name + "/mfpp_sarfa/heat/" + str(index) + ".jpg is saved")
        # 局部解释评估
        self.evaluate_attribution(anti_attribution_list, sensitivity_process_list)
        data = {
            "infidelity": sum(self.infidelities) / len(self.infidelities),
            "sensitivity": sum(self.sensitivities) / len(self.sensitivities),
            "stability": sum(self.stabilities) / len(self.stabilities),
            "validity": sum(self.validities) / len(self.validities),
            "sufficiency": sum(self.sufficiencies) / len(self.sufficiencies),
        }
        with open("output/" + self.game_name + "/mfpp_sarfa/" + "mfpp_sarfa" + ".json", "w") as f:
            json.dump(data, f)
        print("output/" + self.game_name + "/mfpp_sarfa/" + "mfpp_sarfa" + ".json is saved")

    def explain_by_goh(self):
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
            sensitivity_process = sensitivity_max(goh_explain_fn, input, model=self.behavior_net, img_file=obs_input, )

            if attribution.max() > 0:
                self.attributions.append(attribution)
                sensitivity_process_list.append(sensitivity_process)
                mean_val = attribution.clamp(min=0).mean().data.item()
                max_val = attribution.clamp(min=0).max().data.item()
                plt.figure()  # 创建一个新的图像窗口
                # occ (4,84,84)
                plt.imshow(self.max_obs_list[index].cpu().detach().numpy())
                plt.savefig("output/" + self.game_name + "/goh/original-img/" + str(index) + ".jpg")
                print("output/" + self.game_name + "/goh/original-img/" + str(index) + ".jpg is saved")
                plt.imshow(attribution.clamp(min=0.7 * max_val).squeeze().mean(dim=-1).cpu().detach().numpy(),
                           alpha=0.5)
                plt.text(0, -5, 'Original Image predict:{}'.format(self.action_name_list[action.data.item()]),
                         color='black')
                plt.savefig("output/" + self.game_name + "/goh/heat/" + str(index) + ".jpg")
                plt.clf()  # 清除当前图像窗口
                print("output/" + self.game_name + "/goh/heat/" + str(index) + ".jpg is saved")
            # 局部解释评估
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

            # sarfa
            attribution = sarfa(self.behavior_net, input, obs_input, action.data.item())
            anti_attribution = sarfa(self.behavior_net, input, obs_input, min_idx.data.item())
            sensitivity_process = sensitivity_max(sarfa_explain_fn, input, model=self.behavior_net, target=action.data.item(),img_file=obs_input)


            if attribution.max() > 0:
                self.attributions.append(attribution)
                anti_attribution_list.append(anti_attribution)
                sensitivity_process_list.append(sensitivity_process)
                mean_val = attribution.clamp(min=0).mean().data.item()
                max_val = attribution.clamp(min=0).max().data.item()
                plt.figure()  # 创建一个新的图像窗口
                # occ (4,84,84)
                plt.imshow(self.max_obs_list[index].cpu().detach().numpy())
                plt.savefig("output/" + self.game_name + "/sarfa/original-img/" + str(index) + ".jpg")
                print("output/" + self.game_name + "/sarfa/original-img/" + str(index) + ".jpg is saved")
                plt.imshow(attribution.clamp(min=0.7 * max_val).squeeze().mean(dim=-1).cpu().detach().numpy(),
                           alpha=0.5)
                plt.text(0, -5, 'Original Image predict:{}'.format(self.action_name_list[action.data.item()]),
                         color='black')
                plt.savefig("output/" + self.game_name + "/sarfa/heat/" + str(index) + ".jpg")
                plt.clf()  # 清除当前图像窗口
                print("output/" + self.game_name + "/sarfa/heat/" + str(index) + ".jpg is saved")
            # 局部解释评估
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


    def global_explain(self):
        # 将每一帧的attribution和out拼接在一起，并将所有帧的数据放在一个列表中
        data = [attribution.mean(dim=1).squeeze().cpu().detach().numpy().flatten() for
                attribution in self.attributions]

        input_data = [input.squeeze()[-1, :, :].cpu().detach().numpy().flatten() for input in self.max_input_list]

        # 将列表转换为numpy数组
        data = np.array(data)
        occ_data = np.array(input_data)

        # 使用DBSCAN进行聚类
        # dbscan = DBSCAN(eps=7.15, min_samples=3).fit(data) #oc
        # dbscan = DBSCAN(eps=0.1, min_samples=3).fit(data)  # ig
        dbscan = DBSCAN(eps=7.0, min_samples=3).fit(data)  # mfpp-sarfa
        dbscan_occ = DBSCAN(eps=60.0, min_samples=3).fit(occ_data)  # mfpp-sarfa
        # dbscan是你的DBSCAN模型
        labels = dbscan.labels_
        labels_occ = dbscan_occ.labels_

        # 计算总的类别数量
        # 注意：我们需要加1，因为标签从0开始
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_clusters_occ = len(set(labels_occ)) - (1 if -1 in labels_occ else 0)
        # 输出聚类结果
        print(n_clusters, n_clusters_occ)

        # 创建一个字典来存储每个标签下的attribution
        #
        # 遍历每个标签
        for label in set(labels):
            if label != -1:  # Noise points are labeled -1
                mask = (labels == label)
                max_diff_item = 0
                max_diff_index = 0
                max_diff_img = None
                max_diff_action = 0
                img_list = [self.max_input_list[i] for i in range(len(self.max_input_list)) if mask[i]]
                # 获取该标签下的所有attribution
                for index, img in enumerate([self.attributions[i] for i in range(len(self.attributions)) if mask[i]]):
                    plt.figure()  # 创建一个新的图像窗口
                    # occ (4,84,84)
                    reward, _ = torch.max(self.max_output_list[index].data, 1)
                    action = self.max_action_list[index]
                    min_reward, min_action = torch.min(self.max_output_list[index].data, 1)
                    if reward.data.item() - min_reward.data.item() > max_diff_item:
                        max_diff_item = reward.data.item() - min_reward.data.item()
                        max_diff_index = index
                        max_diff_img = img
                        max_diff_action = action

                plt.imshow(img_list[max_diff_index].squeeze()[-1, :, :].cpu().detach().numpy(), cmap='gray')
                max_diff_img.clamp(min=0).mean().data.item()
                max_val = max_diff_img.clamp(min=0).max().data.item()
                plt.imshow((max_diff_img.clamp(min=0.7 * max_val).mean(dim=1).squeeze().cpu().detach().numpy()),
                           alpha=0.5, cmap='rainbow')
                plt.text(0, -5,
                         'Original Image predict:{},label is {}'.format(
                             self.action_name_list[max_diff_action.data.item()],
                             label),
                         color='black')
                plt.savefig("output/" + self.game_name + "/mfpp_sarfa/group-heat/" + str(label) + ".jpg")
                print("output/" + self.game_name + "/mfpp_sarfa/group-heat/" + str(label) + ".jpg is saved")
                plt.clf()  # 清除当前图像窗口

            # 遍历每个标签
        for label in set(labels_occ):
            if label != -1:  # Noise points are labeled -1
                mask = (labels_occ == label)
                max_diff_item = 0
                max_diff_img = None
                max_diff_action = 0
                # 获取该标签下的所有attribution
                for index, img in enumerate(
                        [self.max_input_list[i] for i in range(len(self.max_input_list)) if mask[i]]):
                    plt.figure()  # 创建一个新的图像窗口
                    # occ (4,84,84)
                    reward, _ = torch.max(self.max_output_list[index].data, 1)
                    action = self.max_action_list[index]
                    min_reward, min_action = torch.min(self.max_output_list[index].data, 1)
                    if (reward.data.item() - min_reward.data.item() > max_diff_item):
                        max_diff_item = reward.data.item() - min_reward.data.item()
                        max_diff_img = img
                        max_diff_action = action

                plt.imshow(max_diff_img.squeeze()[-1, :, :].cpu().detach().numpy())
                plt.text(0, -5,
                         'Original Image predict:{},label is {}'.format(
                             self.action_name_list[max_diff_action.data.item()],
                             label),
                         color='black')
                plt.savefig("output/mfpp_sarfa/group-img/" + str(label) + ".jpg")
                print("output/mfpp_sarfa/group-img/" + str(label) + ".jpg is saved")
                plt.clf()  # 清除当前图像窗口

    def global_explain_2(self):
        # 按照动作和原图将热力图分类
        action_attr_list = [[] for _ in range(len(self.action_name_list))]
        action_obs_list = [[] for _ in range(len(self.action_name_list))]
        action_output_list = [[] for _ in range(len(self.action_name_list))]
        action_real_list = [[] for _ in range(len(self.action_name_list))]
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

        dbscan_eps_list = [10.0, 10.0, 14.0, 8.0]
        for action, attr_list in enumerate(action_attr_list):
            # 获取该动作下的所有原图
            obs_list = action_obs_list[action]
            output_list = action_output_list[action]
            real_list = action_real_list[action]
            # 将每一帧的attribution和out拼接在一起，并将所有帧的数据放在一个列表中

            data = [attribution.mean(dim=1).squeeze().cpu().detach().numpy().flatten() for
                    attribution in attr_list]
            if len(data) > 0:
                # 将列表转换为numpy数组
                data = np.array(data)
                # 使用DBSCAN进行聚类
                # dbscan = DBSCAN(eps=7.15, min_samples=3).fit(data) #oc
                # dbscan = DBSCAN(eps=0.1, min_samples=3).fit(data)  # ig
                dbscan = DBSCAN(eps=7, min_samples=3).fit(data)  # mfpp-sarfa
                # dbscan是你的DBSCAN模型
                labels = dbscan.labels_

                # 计算总的类别数量
                # 注意：我们需要加1，因为标签从0开始
                n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                # 输出聚类结果
                print(action, n_clusters)

                # 创建一个字典来存储每个标签下的attribution
                #
                # 遍历每个标签
                for label in set(labels):
                    if label != -1:  # Noise points are labeled -1
                        mask = (labels == label)
                        max_diff_item = 0
                        max_diff_index = 0
                        max_diff_attr = None
                        obs_label_list = [obs_list[i] for i in range(len(obs_list)) if mask[i]]
                        # 获取该标签下的所有attribution
                        for index, attr in enumerate(
                                [attr_list[i] for i in range(len(attr_list)) if mask[i]]):
                            plt.figure()  # 创建一个新的图像窗口
                            reward, _ = torch.max(output_list[index].data, 1)
                            action = real_list[index]
                            min_reward, min_action = torch.min(output_list[index].data, 1)
                            if reward.data.item() - min_reward.data.item() > max_diff_item:
                                max_diff_item = reward.data.item() - min_reward.data.item()
                                max_diff_index = index
                                max_diff_attr = attr
                        plt.imshow(obs_label_list[max_diff_index].squeeze().cpu().detach().numpy())
                        # 210 160 3
                        max_diff_attr.clamp(min=0).mean().data.item()
                        max_val = max_diff_attr.clamp(min=0).max().data.item()
                        plt.imshow(max_diff_attr.clamp(min=0.7 * max_val).squeeze().mean(dim=-1).cpu().detach().numpy(),
                                   alpha=0.5)
                        plt.text(0, -5,
                                 'Original Image predict:{},label is {}'.format(
                                     self.action_name_list[action],
                                     label),
                                 color='black')
                        plt.savefig(
                            "output/" + self.game_name + "/mfpp_sarfa/action-group-heat/" + self.action_name_list[
                                action] + "/" + str(label) + ".jpg")
                        print("output/" + self.game_name + "/mfpp_sarfa/action-group-heat/" + self.action_name_list[
                            action] + "/" + str(label) + ".jpg is saved")
                        plt.clf()  # 清除当前图像窗口
            else:
                print("action", self.action_name_list[action], "has no data")

    def evaluate_attribution(self, anti_attribution_list, sensitivity_process_list):
        for index, attribution in enumerate(self.attributions):
            # 计算各指标
            max_val = torch.max(attribution)
            threshold = 0 * max_val.item()
            reward, _ = torch.max(self.max_output_list[index].data, 1)
            action = self.max_action_list[index]
            # infidelity
            self.infidelities.append(
                get_infidelity(self.behavior_net, self.max_input_list[index].unsqueeze(0), attribution,
                               action).data.item())
            # sensitivity
            if(len(anti_attribution_list) > 0):
                sensitivity = get_sensitivity(attribution, anti_attribution_list[index])
                if not torch.isnan(sensitivity):
                    self.sensitivities.append(sensitivity.data.item())
            # stability
            self.stabilities.append(get_stability(sensitivity_process_list[index]).data.item())
            # validity
            self.validities.append(
                get_validity(attribution, threshold, self.behavior_net, self.max_input_list[index].unsqueeze(0), action,
                             reward).data.item())
            # sufficiency
            self.sufficiencies.append(
                get_sufficiency(attribution, threshold, self.behavior_net, self.max_input_list[index].unsqueeze(0),
                                action, reward).data.item())
