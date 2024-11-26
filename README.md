This repository contains an implementation of a genetic algorithm (GA) developed to evolve trading strategies within the Directional Change (DC) framework. Unlike traditional time-based methods, this approach focuses on capturing significant market moves, allowing for an event-driven analysis of price dynamics. The GA seeks to optimize trading rules represented as tree structures, combining directional change thresholds with logical operations such as AND, OR, NOR, XOR,NOT. By evolving these rules over multiple generations, the algorithm tries to identify profitable strategies for use in financial markets.

The implementation begins by generating a population of random trading strategies, represented as logic trees. These trees are refined through evolutionary operations like mutation, crossover, and tournament selection. Fitness is evaluated by simulating trades on historical market data, starting with a predefined cash balance. Directional changes are detected based on thresholds assigned to the leaf nodes of the trees, while logical operations at internal nodes define how these signals combine to generate buy or sell actions.

# Run the genetic algorithm on the price data
best_strategy = gp(prices)

# Output the best tree-based strategy
print("Best Strategy:", best_strategy)

# Further read

Gypteau, J., Otero, F. E. B., & Kampouridis, M. (2015). Generating Directional Change Based Trading Strategies with Genetic Programming.
Wang, X., Kampouridis, M., & Tsang, E. P. K. (2022). Genetic Programming for Combining Directional Changes Indicators in International Stock Markets.
Adegboye, A., Kampouridis, M., & Otero, F. E. B. (2023). Algorithmic Trading with Directional Changes.
Aloud, F., Tsang, E. P. K., & Olsen, R. (2012). A Directional Change Based Trading Strategy with Dynamic Thresholds.
Wang, X., Kampouridis, M., & Tsang, E. P. K. (2023). Multi-Objective Optimisation and Genetic Programming for Trading by Combining Technical Analysis and Directional Changes.

# Requirements
The code requires Python 3.x and NumPy for execution.
