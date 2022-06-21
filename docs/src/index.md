# RNAForecaster.jl - Estimating Future Expression States on Short Time Scales

RNAForecaster is based on a neural network that is trained using ordinary differential equations solvers (neural ODEs). See [this paper](https://arxiv.org/abs/1806.07366) for details on neural ODEs. The goal of this method is to forecast future expression levels in a cell given transcriptomic data from said cell. Much like a weather forecaster, it can only make predictions accurately in the short to medium term and even then there will be times when it does not get it right. Despite these limitations, weather forecasts are still useful, and hopefully RNA forecasts will be useful for you too.

# Installation
Download [Julia 1.6](https://julialang.org/) or later, if you haven't already.

You can add RNAForecaster from using Julia's package manager, by typing `] add RNAForecaster` in the Julia prompt.
