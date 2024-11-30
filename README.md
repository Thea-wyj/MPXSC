# MPXSC

> Codes for “Multi-granularity Policy Explanation of Deep Reinforcement Learning Based on Saliency Map Clustering”

## Method 

Explainable Deep Reinforcement Learning (XDRL) holds significant potential for clarifying the decision-making logic of agents in complex tasks. Current XDRL research increasingly focuses on multi-granularity policy explanation methods that integrate both local and global decision-making insights. However, most of them treat local and global explanations as separate processes, overlooking their interdependencies and compromising logical consistency. This paper introduces a Multi-granularity Policy eXplanation method based on Saliency map Clustering (MPXSC), which computes both local and global policy explanations for a DRL agent in a unified, end-to-end process. MPXSC begins by employing super-pixel perturbation to generate saliency maps for all the agent’s states, representing its local explanation. These maps are then categorized by agent's actions and clustered based on local saliency features. Subsequently, the key states within each cluster are identified and serve as decision rules for their respective actions. Collectively, these results constitute the global explanation for the agent's decision-making. The superiority of MPXSC is validated through objective experiments on both local and global explanations, with a case study visually demonstrating the method’s compelling explainable evidence regarding DRL model prediction outcomes.
## System Requirements

- Device：`Legion Y7000P2020H, Windows 11 Enterprise Edition`

- CPU: `Intel(R) Core(TM) i7-10875H CPU @ 2.30GHz 2.30 GHz`

- GPU: `NVIDIA GeForce RTX 2060`

- Python installed (preferably Python 3.9)

- Relevant libraries and dependencies for the program are installed

  `pip install -r requirements.txt`

- A compatible CUDA-enabled GPU

## Running the Program

To execute the program, use the following command in your terminal:

```shell
#Play Breakout and explain by Goh、Sarfa and MPXSC
python main.py --agent=dql --eval=True --game_index=0
#Play MsPacman and explain by Goh、Sarfa and MPXSC
python main.py --agent=dql --eval=True --game_index=1
```

### Command Explanation

- `--agent {dql,dsac}    Deep Q-learning and discrete soft Actor-Critics algorithms.`
- `--live_penalty LIVE_PENALTY: Penalties when agent lose a life in the game.`
- `--reward_clip REWARD_CLIP: Clip reward in [-1, 1] range if True.`
- `--min_epsilon MIN_EPSILON: The probability for random actions.`
- `--start_epsilon START_EPSILON: The probability for random actions.`
- `--memory_size MEMORY_SIZE: The size of the memory space.`
- `--env_name ENV_NAME: The name of the gym atari environment.`
- `--game_index {0,1,2} : Represent Breakout, MsPacman and Pong respectively.`
- `--eval EVAL : True means evaluate model only.`

## Example Result

### Local decision explanation 



![image-20240810043617394](./result/Local.png)

### Global policy explanation

![image-20240810043712108](./result/Global.png)
