# Financial Decision Simulation using Deep Q-Learning (DQN)

This project demonstrates the use of **Deep Q-Learning (DQN)** and **Monte Carlo simulations** for solving a complex financial decision-making problem. The agent needs to optimally allocate a monthly savings of $1,500 across multiple priorities, such as:
- Contributing to a 401(k),
- Repaying student loans,
- Saving for a house down payment, and
- Saving for a child’s future college tuition.

The model learns how to maximize savings and minimize debt over a period of 30 years, factoring in variable interest rates, salary growth, and market returns.

---



## Project Overview

The core of this project is a **Deep Q-Learning (DQN)** algorithm that learns to optimally allocate a monthly savings amount based on multiple competing financial goals:
1. **401(k) Contributions**: Up to 6% of salary is matched by the employer.
2. **Loan Repayment**: Interest accrues on outstanding loans at 6% annually.
3. **House Savings**: $66,000 must be saved by 2030 for a house down payment.
4. **College Savings**: By 2046, $436,000 must be saved for a child's college tuition.

The model uses **Monte Carlo simulations** to simulate different financial conditions, such as varying salary growth rates and market returns.

---

## Key Features
- **Deep Q-Learning**: A reinforcement learning algorithm that adapts financial decisions over time.
- **Monte Carlo Simulation**: Simulates different salary growth rates and market returns to account for uncertainty.
- **Visualization of Best Strategies**: Plots to visualize the optimal savings allocation strategy over time.
- **Multiple Financial Priorities**: The agent learns to balance between saving for a house, 401(k), college, and repaying loans.

---

## Dependencies

Ensure you have the following libraries installed:

```txt
tensorflow==2.9.1
matplotlib==3.5.1
scikit-learn==1.0.2
numpy==1.22.4
