# Training and Forecasting

Training RNAForecaster requires two expression count matrices. These count matrices should
be formatted with genes  as rows and cells as columns. Each matrix should represent
two different time points from the same cells. This can be accomplished from
transcriptomic profiling using spliced and unspliced counts
or by using labeled and unlabeled counts from metabolic labeling scRNA-seq
protocols such as scEU-seq (see [Battich et al. 2020](https://doi.org/10.1126/science.aax3072)).

Here, we will generate two random matrices for illustrative purposes.

```julia
testT0 = log1p.(Float32.(abs.(randn(10,1000))))
testT1 = log1p.(0.5f0 .* testT0)
```
Note that the input matrices are expected to be of type Float32 and log transformed,
as shown above.

# Training

To train the neural ODE network we call the `trainRNAForecaster` function.

```julia
testForecaster = trainRNAForecaster(testT0, testT1);
```

In the simplest case, we only need to input the matrices, but there are several options
provided to modify the training of the neural network, as shown below.
```@docs
trainRNAForecaster
```

For example, by default RNAForecaster partitions the input data into a training
and a validation set. If we want the neural network to be trained on the entire data
set, we can set `trainingProp = 1.0`.

When using larger data sets, such as a matrix from a normal scRNAseq experiment which
may contain thousands of variable genes and tens of thousands of cells, it becomes
inefficient to train the network on a CPU. If a GPU is available, setting
`useGPU = true` can massively speed up the training process.

#Forecasting

Once we have trained the neural network, we can use it to forecast future expression
states. For example, to predict the next fifty time points from our test data,
we could run:

```julia
testOut1 = predictCellFutures(testForecaster[1], testT0, 50);
```

The predictions can also be conditioned on arbitrary perturbations in gene expression.
```julia
geneNames = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]
testOut2 = predictCellFutures(testForecaster[1], testT0, 50, perturbGenes = ["A", "B", "F"],
geneNames = geneNames, perturbationLevels = [1.0f0, 2.0f0, 0.0f0]);
```


All options for `predictCellFutures` are shown here:
```@docs
predictCellFutures
```

Once we have forecast expression levels for each gene, we may want to know which
genes expression levels change the most over time, as these are likely to be important
in ongoing biological process we are attempting to model.
To assay this we simply run `mostTimeVariableGenes` which outputs a table of genes
ordered by the most variable over predicted time points.

```julia
geneOutputTable = mostTimeVariableGenes(testOut1, geneNames)
```

```@docs
mostTimeVariableGenes
```
