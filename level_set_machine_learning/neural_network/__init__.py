"""
This is a simple neural network class
for regression.


A general number of inputs and hidden 
is supported, but the output number is 
restricted to 1. 

Input (R^n) => Hidden (R^h) => Output (R)

Gradient of the squared error loss with 
respect to model parameters is implemented.

This model also automatically normalizes data
by recording empirical means and standard 
deviations of observed data (see the 
`statsrecorder.py` file).

- Matt Hancock, 2018
"""
