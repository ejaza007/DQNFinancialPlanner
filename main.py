import numpy as np
import random
import tensorflow as tf
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Define financial simulation parameters
initial_salary = 85000
initial_loan_balance = 75000
monthly_savings = 1500
monthly_house_savings = 1383
loan_interest_rate = 0.06  # 6% interest rate on loans
annual_return_401k = 0.07  # Expected return on 401(k)
months_to_save_for_house = 48  # Saving for 4 years (2026-2030)
college_cost = 473000  # Total college cost by 2046
max_401k_contribution = 0.06

# RL parameters
state_size = 5  # Loan, 401(k), House, College, Time
action_size = 4  # 401(k), House, College, Loan
learning_rate = 0.001
gamma = 0.95  # Discount rate
epsilon = 1.0  # Exploration rate
epsilon_min = 0.01
epsilon_decay = 0.995


# Deep Q-Learning model
class DQNModel:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.model = self.build_model()

    def build_model(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, input_dim=self.state_size, activation='relu'),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(self.action_size, activation='linear')  # Linear output layer for Q-values
        ])
        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate))

        return model

    def predict(self, state):
        return self.model.predict(state)

    def train(self, state, target):
        self.model.fit(state, target, epochs=1, verbose=0)


# Define environment for state transitions
class FinancialEnvironment:
    def __init__(self):
        self.state = np.array([initial_loan_balance, 0, 0, 0, 0])  # Initial state [Loan, 401(k), House, College, Time]

    def reset(self):
        self.state = np.array([initial_loan_balance, 0, 0, 0, 0])
        return self.state

    def step(self, action, salary, time):
        loan, retirement, house, college, _ = self.state

        # Calculate salary and available savings
        available_savings = monthly_savings * (salary / 85000)

        # Step 1: Allocate funds for the pre-2030 period (priority on house savings)
        if time < months_to_save_for_house:  # If we're before 2030
            house_savings = monthly_house_savings  # Mandatory $1,383 to housing
            available_savings -= house_savings  # Subtract from available savings

            # Allocate remaining savings to 401(k), college, or loan repayment
            allocation_401k = min(action[0] * salary, salary * max_401k_contribution / 12)
            extra_loan_payment = action[2] * available_savings  # Use extra savings for loans
            college_savings = action[1] * available_savings  # Small allocation to college, if possible

        # Step 2: Post-2030 period, no more house savings, agent decides full allocation
        else:
            house_savings = 0  # No more housing allocation after 2030

            # Post-2030: Allocate freed-up savings between 401(k), loan repayment, college
            allocation_401k = min(action[0] * salary, salary * max_401k_contribution / 12)
            extra_loan_payment = action[2] * available_savings  # Allocate for loans
            college_savings = action[1] * available_savings  # Prioritize saving for college

        # Update the state based on actions
        loan = loan * (1 + loan_interest_rate / 12) - (300 + extra_loan_payment)  # Loan interest accrues if unpaid
        retirement += allocation_401k * 2  # Employer matches 401(k) contributions
        house += house_savings
        college += college_savings
        time += 1  # Move to next month

        # Reward: The agent is rewarded based on the balance between maximizing savings and minimizing debt
        self.state = np.array([loan, retirement, house, college, time])

        # College savings goal is $436k by 2046; penalize if not saving enough
        remaining_years_to_save = max(0, 2046 - (2026 + time // 12))  # Time left until college
        required_college_savings_per_year = 436000 / remaining_years_to_save if remaining_years_to_save > 0 else 0

        # Reward is savings, penalized if far from meeting college savings goal
        college_savings_penalty = max(0, required_college_savings_per_year * (remaining_years_to_save // 12) - college)

        reward = (retirement + house + college) - loan - college_savings_penalty
        return self.state, reward


# Run Monte Carlo simulations to generate scenarios
def monte_carlo_simulations():
    scenarios = []
    for _ in range(10):  # Run 1000 scenarios
        salary_growth = np.random.uniform(0.03, 0.07)
        market_return = np.random.uniform(0.05, 0.10)
        loan_interest = np.random.uniform(0.05, 0.07)
        scenarios.append([salary_growth, market_return, loan_interest])
    return scenarios


# Train the DQN model
def train_dqn(model, env, episodes=10):
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        for time in range(360):  # Simulate over 30 years (360 months)
            action = np.random.rand(4)  # Explore: random action allocation
            next_state, reward = env.step(action, salary=initial_salary * (1 + time // 12), time=time)
            target = reward + gamma * np.amax(model.predict(next_state.reshape(1, -1)))
            target_f = model.predict(state.reshape(1, -1))
            target_f[0][np.argmax(action)] = target
            model.train(state.reshape(1, -1), target_f)
            state = next_state
            total_reward += reward
        if episode % 50 == 0:
            print(f"Episode {episode}: Total Reward = {total_reward}")


# Simulate and analyze strategies
env = FinancialEnvironment()
dqn_model = DQNModel(state_size, action_size)
train_dqn(dqn_model, env)

# After training, apply K-means clustering to analyze strategies
scenarios = monte_carlo_simulations()
kmeans = KMeans(n_clusters=5, random_state=42)
kmeans.fit(scenarios)

# Visualize the clustering results
plt.scatter([x[0] for x in scenarios], [x[1] for x in scenarios], c=kmeans.labels_, cmap='viridis')
plt.title("Monte Carlo Scenarios with K-Means Clustering")
plt.xlabel("Salary Growth Rate")
plt.ylabel("Market Return")
plt.colorbar(label="Cluster")
plt.show()
