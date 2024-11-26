This repository contains an implementation of a genetic algorithm (GA) developed to evolve trading strategies within the Directional Change (DC) framework. Unlike traditional time-based methods, this approach focuses on capturing significant market movements, allowing for an event-driven analysis of price dynamics. The GA seeks to optimize trading rules represented as tree structures, combining directional change thresholds with logical operations such as AND, OR, and NOT. By evolving these rules over multiple generations, the algorithm aims to identify profitable strategies for use in financial markets.

The implementation begins by generating a population of random trading strategies, represented as logic trees. These trees are refined through evolutionary operations like mutation, crossover, and tournament selection. Fitness is evaluated by simulating trades on historical market data, starting with a predefined cash balance. Directional changes are detected based on thresholds assigned to the leaf nodes of the trees, while logical operations at internal nodes define how these signals combine to generate buy or sell actions.

# Run the genetic algorithm on the price data
best_strategy = gp(prices)

# Output the best tree-based strategy
print("Best Strategy:", best_strategy)
