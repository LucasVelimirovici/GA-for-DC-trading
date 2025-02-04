# Scope

This repository contains implementations of genetic algorithms (GA) developed to evolve trading strategies within the Directional Change (DC) framework. Unlike traditional time-based methods, this approach focuses on capturing significant market moves, allowing for an event-driven analysis of price dynamics. The GA seeks to optimize trading rules represented as tree structures, combining directional change thresholds with logical operations such as AND, OR, NOR, XOR,NOT, or is based on indicators derived from the classic technical analysis. By evolving these rules over multiple generations, the algorithm tries to identify profitable strategies for use in financial markets.


# Gypteau implementation

Implemented after Gypteau's work (see reference [5] in further_read.txt), this algorithm aims to find profitable trading strategies based on the DC framework. Strategies are encoded as binary trees with logic functions as intermediate nodes; terminal nodes consist of DC-specific thresholds. The dataset is dissected into DC events based on the threshold parameter, and each point of the timeseries is converted into either 1 or 0, based on the type of the directional change that occurs at that specific point; the new datapoint takes value=1 if the recorded event is an Upmove, and 0 if it is a Downmove.
After the timeserie is converted into multiple sequences (of the same size) of binary values, based on different DC thresholds, the strategy is tested at each index. If the root node yields 1, the strategy buys one unit of equity, otherwise it sells one unit (if any available). The total return of the strategy after being iterated across the whole dataset represents the fitness values (greater return = greater fit). 
This whole process is repreaded across 300 generations, with standard Genetic Programming (GP) processes taking place: crossover, reproduction, mutation.
The Genetic Loop returns the strategy with the highest return.

#Long implementation

The implementation begins by generating a population of random trading strategies, represented as logic trees. These trees are refined through evolutionary operations like mutation, crossover, and tournament selection. Fitness is evaluated by simulating trades on historical market data, starting with a predefined cash balance or using the Sharpe Ratio as a gauge for invetment quality. Directional changes are detected based on thresholds assigned to the leaf nodes of the trees, while logical operations at internal nodes define how these signals combine to generate buy or sell actions.


