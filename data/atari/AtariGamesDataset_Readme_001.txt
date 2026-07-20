Date: 18 October, 2023 


Dataset Title: Atari Games Dataset


Dataset Creators: B. Chen

Dataset Contact: Brian Chen chenbri@umich.edu, Doruk Aksoy doruk@umich.edu

Funding: W56HZV-19-2-0001 (Automotive Research Center at the University of Michigan in accordance with U.S. Army GVSC), DE-SC0020364 (DOE Office of Scientific Research, ASCR)


Key Points:
- We provide a benchmark dataset for behavioral cloning applications involving ATARI games.
- This dataset can be used to demonstrate/compare the efficacy of representation learning methods in extracting useful features from image-based datasets.

Research Overview:

The procedure followed while creating this data is summarized in Section II of Chen, Brian, et al. "Behavioral cloning in atari games using a combined variational autoencoder and predictor model." 2021 IEEE Congress on Evolutionary Computation (CEC). IEEE, 2021. This data is not a result of a research but an intermediate product that is used in research. The task this data first utilized is behavioral cloning in Atari games.
The ATARI 2600 games that are used to create this dataset are:
- BeamRider
- Breakout
- Qbert
- Pong
- MsPacman
- SpaceInvaders
- Seaquest
- Enduro

This data is currently used to compare the performance of a recently developed incremental tensor decomposition algorithm, TT-ICE (Aksoy, et al. 2023, "An incremental tensor train decomposition algorithm.") against algorithms in the existing literature.
In addition to that, this dataset is further used to compare the performance of TT-ICE against existing representation learning methods such as autoencoders/variational autoencoders in behavioral cloning tasks.


Methodology:
To generate the data in this repository,  an AI player, which is trained using the RL Baselines Zoo package v1.4, is used (A. Raffin, “Rl baselines zoo,” https://github.com/araffin/rl-baselines-zoo,
2018.). The player is represented as a Deep Q-Network (DQN) (V. Mnih, et al. “Human-level control through deep reinforcement learning,” Nature, vol. 518, pp. 529–533, 2015), which is loaded from the RL Baselines Zoo package and trained on an additional 10^7 samples (states and actions) of each Atari game using the default parameters provided by the package.


Instrument and/or Software specifications: N/A


Files contained here:

The dataset contains 8 compressed archives, each containing enumerated .npz files and folders containing .png files. Contents of a compressed archive can be extracted using any program that supports .zip files (unzip command from Linux command line, or WinZip/WinRAR). Each game type has different number of gameplay sequences with varying length. Once an archive is extracted, the directories contain 2 types of files:
1) Standard .png snapshots of gameplay sequences in folders uniquely identified using numbers, and
2) .npz files that contain information for the gameplay sequence identified with the number at the end.


The naming convention used for the .zip files are <Game name>NoFrameskip-v4.zip. Within a compressed archive, there are unique gameplay sequences, each assigned to an integer index. The naming convention used for subfolders and .npz files is <Game name>NoFrameskip-v4_<Sequence ID>. 

.npz files can be opened using Python NumPy package's load function. Each .npz file contains a dictionary of the following key-value pairs: 

- "model selected actions": Action taken by the RL agent at each timestep of the gameplay episode
- "taken actions": Action taken by the game playing anent at each timestep of the gameplay episode (Note that this might be different from "model selected actions" since there is a 25% probability that a taken action is a "sticky action", i.e., repetition of the last taken action by the game playing agent)
- "obs": The path to image file that correspond to each timestep
- "rewards": The reward associated with the action selected at the given frame in "obs"
- "episode_returns": Point value achieved by the end of the gameplay episode
- "episode_starts": Boolean variable indicating if a frame is the first frame of a gameplay episode
- "repeated": Boolean variable indicating if the action of the previous timestep is repeated (i.e., if an action is a "sticky action")

Related publication(s):
Chen, Brian, et al. "Behavioral cloning in atari games using a combined variational autoencoder and predictor model." 2021 IEEE Congress on Evolutionary Computation (CEC). IEEE, 2021.
Aksoy, Doruk, et al. "An Incremental Tensor Train Decomposition Algorithm." arXiv preprint arXiv:2211.12487 (2022).
Chen, Brian, et al. "Low-Rank Tensor-Network Encodings for Video-to-Action Behavioral Cloning", forthcoming

Use and Access: 
This data set is made available under a Attribution-NonCommercial 4.0 International (CC BY-NC 4.0) license.


To Cite Data: 
Chen B., Atari games dataset [Data set]. University of Michigan - Deep Blue.