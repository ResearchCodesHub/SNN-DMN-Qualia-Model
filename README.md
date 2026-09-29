A Spiking Neural Network Model of Elementary Self-Consciousness via Endogenous Default Mode Network Dynamics
R. Lahoz-Beltra

The Crazy Lab©, Madrid, Spain.
rafa.lahoz@protonmail.ch


Understanding the neurobiological mechanisms underlying self-referential cognition and baseline self-consciousness remains a fundamental challenge in computational neuroscience. In this work, we propose a large-scale computational model incorporating a 10,000-neuron spiking neural network (SNN) based on Izhikevich dynamics. The network is structured into two interacting subsystems: a sensory processing layer (5,000 regular-spiking cortical neurons) and an endogenous Default Mode Network (DMN) pacemaker subsystem (5,000 intrinsically bursting neurons). The DMN layer is modulated by continuous tonic currents reflecting ascending brainstem neuromodulation, maintaining intrinsic, autonomous bioelectric rhythms independent of external sensory input. To represent top-down cognitive modulation, synaptic weights are hierarchically structured such that DMN-to-network projections exceed sensory-level connections. Through numerical simulations using a modified two-step Euler integration scheme, we demonstrate how endogenous pacemaker activity interacts with transient external sensory perturbations, providing an elementary mathematical framework for the emergence of a persistent, self-sustaining neural representation of “Self”.

<img width="449" height="251" alt="imagen" src="https://github.com/user-attachments/assets/61c833fb-8c5a-4314-85e5-5a9be400e287" />

Architecture and quantitative framework of the 10,000-neuron Spiking Neural Network (SNN) for computational consciousness and Qualia simulation. Top: Network architecture comprising a Sensory Layer (N=5,000 regular spiking neurons) driven by external current Iext, bidirectionally coupled via top-down (wDMN = 0.6) and bottom-up (wSensory = 0.4) weights to a Default Mode Network (DMN; N=5,000 intrinsically bursting pacemaker neurons) subject to persistent tonic current (Itonic = 7.0 pA). Bottom left: Mathematical formulation and matrix transformations for computing integrated information, utilizing binary spike matrices (X) and global vs. partition cross-covariance matrices. Bottom right: Continuous dynamic trajectories reflecting simulated Qualia states over time.
