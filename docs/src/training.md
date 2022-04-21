# Training and Forecasting

Training RNAForecaster requires two expression count matrices. These count matrices should
be genes x cells and should represent two different time points from the same cells. The current
ways this can be accomplished from transcriptomic profiling is using spliced and unspliced counts
or by using labeled and unlabeled counts from metabolic labeling scRNA-seq protocols such as scEU-seq.

Here, we will generate two random matrices for illustrative purposes.

```julia
testT1 = log1p.(Float32.(abs.(randn(10,1000))))
testT2 = log1p.(0.5f0 .* testT1)
```
Note that the input matrices are expected to be of type Float32 and log transformed,
as shown above.

# Training

To train the model we call the `trainRNAForecaster` function.

```julia
testForecaster = trainRNAForecaster(testT1, testT2)
```

In the simplest case, we only need to input the matrices, but there are several options
provided to modify the training of the neural network, as shown below.
```@docs
trainRNAForecaster
```

For example, by default RNAForecaster partitions the input data into a training
and a validation set. If we want the neural network to be trained on the entire data
set, we can set `trainingProp = 1.0`.

One significant part of the training process is checking for model stability,
which is on by default. What this does is make sure that when the network tries
to make predictions several time steps into the future, that it does not make
unreasonable predictions. This is necessary because sometimes the network makes
extremely high predictions (e.g. predicting thousands of counts for a single gene).
When this happens, we fully retrain the network, and because gradient descent is
stochastic, we often find a more stable solution that does not lead to absurd predictions.
If you don't want to check stability set `checkStability = false`.

#Forecasting

Once we have trained the neural network, we can use it to forecast future expression
states. For example, to predict the next fifty time points from our test data,
we could run:

```julia
testOut2 = predictCellFutures(testForecaster[1], testT1, 50)
```

The predictions can also be conditioned on arbitrary perturbations in gene expression.
