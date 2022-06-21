# RNAForecaster.jl - Estimating Future Expression States on Short Time Scales

RNAForecaster is based on a neural network that is trained using ordinary differential equations solvers (neural ODEs). See https://arxiv.org/abs/1806.07366
for details on neural ODEs. The goal of this method is to forecast future expression levels in a cell given transcriptomic data from said cell over short to
intermediate time periods.


# Installation
Download [Julia 1.6](https://julialang.org/) or later, if you haven't already.

You can add RNAForecaster from using Julia's package manager, by typing `] add RNAForecaster` or `add https://github.com/rossinerbe/RNAForecaster.jl` in the Julia prompt.

