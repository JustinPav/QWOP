# QWOP Reinforcement Learning

This project implements a reinforcement learning model using the Proximal Policy Optimization (PPO) algorithm from the Stable Baselines3 library. The model is trained to play the QWOP game, a challenging physics-based game where players control a runner's legs to navigate a track.

## Full Setup Guide: qwop-gym on Ubuntu 22.04

### 1. Update your system
```bash
sudo apt update && sudo apt upgrade -y
```

### 2. Install Python 3.10+ and pip
Ubuntu 22.04 comes with Python 3.10, but let's make sure:
```bash
python3 --version
sudo apt install python3-pip python3-venv -y
```

### 3. Install Google Chrome (Official .deb version)
Install stable version of chrome:
```bash
wget https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb
sudo apt install ./google-chrome-stable_current_amd64.deb
```

Test:
```bash
google-chrome --version
# Should print something like: Google Chrome 135.0.7049.114
```

### 4. Install matching ChromeDriver
ChromeDriver version must match your Chrome version (e.g., 135).
Go to https://googlechromelabs.github.io/chrome-for-testing/ and find the appropriate chrome driver version. Download it and navigate to the directory where the zip file is stored in your terminal.

Install:
```bash
unzip chromedriver-linux64.zip
sudo mv chromedriver-linux64/chromedriver /usr/local/bin/chromedriver
sudo chmod +x /usr/local/bin/chromedriver
```
Test:
```bash
chromedriver --version
# Should print: ChromeDriver 135.0.xxxxxx
```

### 5. Create and activate a virtual environment


Clone the repository:
```bash
git clone https://github.com/JustinPav/QWOP.git
cd QWOP
```

Create virtual environment:
```bash
python3 -m venv qwop-env
source qwop-env/bin/activate
```

### 6. Install qwop-gym
```bash
pip install --upgrade pip
pip install qwop-gym jinja2 typeguard stable-baselines3[extra] gymnasium tensorboard
```

### 7. Initial setup of QWOP environment
```bash
qwop-gym bootstrap
```
When prompted:
- Browser path: `/usr/bin/google-chrome`
- Chromedriver path: `/usr/local/bin/chromedriver`

### 8. Patch the QWOP game
```bash
curl -sL https://www.foddy.net/QWOP.min.js | qwop-gym patch
```

### 9. Test it out
```bash
qwop-gym play
```
You should see the QWOP game window appear and be able to play.

## Usage

To train the model, run one of the following commands:

1. Runs using the standard reward path in the environment but will result in knee sliding
```bash
python ppo_training.py
```

2. Requires longer training time but will use our custom reward path and will attempt to run faster and more upright.
```bash
python torch_PPO_training.py
```
This will set up the QWOP environment, train the PPO model for 100,000 timesteps, and save the trained model to a file named `ppo_qwop`.

## Testing the Model

After training, the model can be tested by running the `ppo_test.py` or `torch_PPO_test.py` script. The model will predict actions based on the current observations and render the game environment.
