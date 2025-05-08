# QWOP Reinforcement Learning

This project implements a reinforcement learning model using the Proximal Policy Optimization (PPO) algorithm from the Stable Baselines3 library. The model is trained to play the QWOP game, a challenging physics-based game where players control a runner's legs to navigate a track.

## Installation Instructions

1. Clone the repository:
   ```
   git clone <repository-url>
   cd QWOP
   ```

2. Install the required packages:
   ```
   pip install stable-baselines3 gymnasium qwop-gym
   ```

## Usage

To train the model, run the following command:
```
python ppo_training.py
```

This will set up the QWOP environment, train the PPO model for 100,000 timesteps, and save the trained model to a file named `ppo_qwop`.

## Testing the Model

After training, the model can be tested by running the same script. The model will predict actions based on the current observations and render the game environment.

## Contributing

Feel free to submit issues or pull requests if you have suggestions or improvements for the project.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.