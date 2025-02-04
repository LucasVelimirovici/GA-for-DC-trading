# Scope

This repository contains implementations of genetic algorithms (GA) developed to evolve trading strategies within the Directional Change (DC) framework. Unlike traditional time-based methods, this approach focuses on capturing significant market moves, allowing for an event-driven analysis of price dynamics. The GA seeks to optimize trading rules represented as tree structures, combining directional change thresholds with logical operations such as AND, OR, NOR, XOR,NOT, or is based on indicators derived from the classic technical analysis. By evolving these rules over multiple generations, the algorithm tries to identify profitable strategies for use in financial markets.


# Gypteau implementation

Implemented after Gypteau's work (see reference [5] in further_read.txt), this algorithm aims to find profitable trading strategies based on the DC framework. Strategies are encoded as binary trees with logic functions as intermediate nodes; terminal nodes consist of DC-specific thresholds. The dataset is dissected into DC events based on the threshold parameter, and each point of the timeseries is converted into either 1 or 0, based on the type of the directional change that occurs at that specific point; the new datapoint takes value=1 if the recorded event is an Upmove, and 0 if it is a Downmove.
After the timeserie is converted into multiple sequences (of the same size) of binary values, based on different DC thresholds, the strategy is tested at each index. If the root node yields 1, the strategy buys one unit of equity, otherwise it sells one unit (if any available). The total return of the strategy after being iterated across the whole dataset represents the fitness values (greater return = greater fit). 
This whole process is repreaded across 300 generations, with standard Genetic Programming (GP) processes taking place: crossover, reproduction, mutation.
The Genetic Loop returns the strategy with the highest return.

# Long implementation

Based afer Long's work (see reference [10] in further_read.txt), this algorithm makes use of a similar logic binary tree strategy structure. However, the terminal nodes are not simple threshold values, but rather a triplet of (indicator, comparator, value). These indicators reflect different aspects of the Directional Change timeserie dissection, and are computed for each point of the dataset, yielding multiple same-size sequences of indicator values for each datapoint of the timeserie. The terminal nodes are testes at each index of the resulting indicator aggregate, and a binary result (0 or 1) is outputted based on whether or not the logic sence is true or falce (for example, a triplet of (IND1,>,0.33) will output 1 (True) at datapoints where indicator IND1 > 0.33 ). Once all leaf nodes are evaluated, non-terminal logic nodes come into play, and the root node will either output 1 or 0. If the strategy output is equal to 1, one unit of equity is bought and held until either one of the following conditions is met : the equity unit was held n days or the return on purached equity is greater than r (n and r serve as hyperparameters).

After running the strategy across the whole dataset and recording pairs of (buy price, sale price), the return of each pair of events is computed (and adjusted for trading fees) and finally, the Sharpe ratio of the strategy is computed. This Sharpe ratio will further serve as the fitness parameter of each individual and will further dictate its impact in the genetic processes that take place (crossover, mutation).
The 10Y US Treasury yield was used as the risk-free return rate for the computation of Sharpe ratio's.



