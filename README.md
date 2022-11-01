# RNAForecaster.jl - Estimating Future Expression States on Short Time Scales

RNAForecaster is based on a neural network that is trained using ordinary differential equations solvers (neural ODEs). See https://arxiv.org/abs/1806.07366
for details on neural ODEs. The goal of this method is to forecast future expression levels in a cell given transcriptomic data from said cell over short to
intermediate time periods.

For more details about the RNAForecaster method, see the preprint on [BioRXiv](https://doi.org/10.1101/2022.08.04.502825)

- **Documentation**: [![][docs-latest-img]][docs-latest-url]

[docs-latest-img]: https://img.shields.io/badge/docs-dev-blue.svg
[docs-latest-url]: https://rossinerbe.github.io/RNAForecaster.jl/dev/

# Installation
Download [Julia 1.6 or later](https://julialang.org/), if you haven't already.

You can add RNAForecaster from using Julia's package manager, by typing `] add RNAForecaster` or `] add https://github.com/rossinerbe/RNAForecaster.jl` in the Julia prompt.



