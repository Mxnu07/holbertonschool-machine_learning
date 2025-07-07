import gymnasium as gym
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from rl.agents.dqn import DQNAgent
from rl.policy import GreedyQPolicy
from rl.memory import SequentialMemory

# Create the environment
env = gym.make('Breakout-v4')
nb_actions = env.action_space.n

# Build the model
model = Sequential()
model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
model.add(Dense(24, activation='relu'))
model.add(Dense(24, activation='relu'))
model.add(Dense(nb_actions, activation='linear'))

# Configure and compile the agent
memory = SequentialMemory(limit=50000, window_length=1)
policy = GreedyQPolicy()
dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=10,
               target_model_update=1e-2, policy=policy)
dqn.compile('adam', metrics=['mae'])

# Load the trained weights
dqn.load_weights('policy.h5')

# Play the game
dqn.test(env, nb_episodes=5, visualize=True)
