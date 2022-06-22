var documenterSearchIndex = {"docs":
[{"location":"forecastMLData/#Metabolic-Labeling-for-Forecasts","page":"Metabolic Labeling for Forecasts","title":"Metabolic Labeling for Forecasts","text":"","category":"section"},{"location":"forecastMLData/","page":"Metabolic Labeling for Forecasts","title":"Metabolic Labeling for Forecasts","text":"One way to get the two time points needed to train RNAForecaster is to use metabolic labeling transcriptomic profiling techniques such as scEU-seq. In these techniques, cells are incubated with modified uridine for a set period of time, after which they are harvested for scRNA sequencing. The transcripts with incorporated uridine are labeled, and we therefore know they were transcribed in the labeling time period. We also know the other transcripts were transcribed earlier.","category":"page"},{"location":"forecastMLData/","page":"Metabolic Labeling for Forecasts","title":"Metabolic Labeling for Forecasts","text":"We can use these labeled and unlabeled transcripts to generate our two input matrices. To get a clean time point t=0 matrix of total expression that we can use with the total expression at time point t=1 (when the cells were harvested) we do need to estimate the number of degraded transcripts in the labeling period, which we calculate via the slope between the labeled and total transcripts (see Qiu et al. 2022).","category":"page"},{"location":"forecastMLData/","page":"Metabolic Labeling for Forecasts","title":"Metabolic Labeling for Forecasts","text":"Here we use a subset of a scEUseq dataset from hTERT immortalized retinal pigment epithelium cells, published by Battich et al. 2020.","category":"page"},{"location":"forecastMLData/","page":"Metabolic Labeling for Forecasts","title":"Metabolic Labeling for Forecasts","text":"using JLD2\nrpeExampleData = load_object(\"data/rpeExampleData.jld2\");\nlabeledData = Float32.(rpeExampleData[1])\nunlabeledData = Float32.(rpeExampleData[2])\ntotalData = Float32.(rpeExampleData[3])\ngeneNames = string.(rpeExampleData[4])\nlabelingTime = vec(rpeExampleData[5])\n\nt0Estimate = estimateT0LabelingData(labeledData, totalData, unlabeledData, labelingTime)","category":"page"},{"location":"forecastMLData/","page":"Metabolic Labeling for Forecasts","title":"Metabolic Labeling for Forecasts","text":"For this tutorial, we will subset to cells labeled for 60 minutes, enabling us to forecast in one hour increments.","category":"page"},{"location":"forecastMLData/","page":"Metabolic Labeling for Forecasts","title":"Metabolic Labeling for Forecasts","text":"hourCells = findall(x->x == 60.0, labelingTime)\nt0_60min = t0Estimate[:,hourCells]\nt1_60min = totalData[:,hourCells]\n\n#log normalize\nt0_60min = log1p.(t0_60min)\nt1_60min = log1p.(t1_60min)\n\n#train RNAForecaster\ntrainedNetwork = trainRNAForecaster(t0_60min, t1_60min, trainingProp = 1.0,\n learningRate = 0.0001, batchsize = 25);","category":"page"},{"location":"forecastMLData/","page":"Metabolic Labeling for Forecasts","title":"Metabolic Labeling for Forecasts","text":"If we wanted to train the network using the GPU, we would run trainedNetwork = trainRNAForecaster(t0_60min, t1_60min, trainingProp = 1.0,  learningRate = 0.0001, batchsize = 25, useGPU = true)","category":"page"},{"location":"forecastMLData/","page":"Metabolic Labeling for Forecasts","title":"Metabolic Labeling for Forecasts","text":"Note on GPU errors: on some systems an error can arise related to the way julia and CUDA handle memory. If you encounter unexpected Out of GPU Memory errors, try setting the environment variable JULIA_CUDA_MEMORY_POOL = none","category":"page"},{"location":"forecastMLData/","page":"Metabolic Labeling for Forecasts","title":"Metabolic Labeling for Forecasts","text":"Now, we can forecast expression predictions 24 hours into the future, and get predictions at each hour.","category":"page"},{"location":"forecastMLData/","page":"Metabolic Labeling for Forecasts","title":"Metabolic Labeling for Forecasts","text":"futureExpressionPreds = predictCellFutures(trainedNetwork[1], t1_60min, 24)","category":"page"},{"location":"forecastMLData/","page":"Metabolic Labeling for Forecasts","title":"Metabolic Labeling for Forecasts","text":"Predictions can also be performed on the GPU futureExpressionPreds = predictCellFutures(trainedNetwork[1], t1_60min, 24,    useGPU = true)","category":"page"},{"location":"forecastMLData/","page":"Metabolic Labeling for Forecasts","title":"Metabolic Labeling for Forecasts","text":"The predictions can also be conditioned on arbitrary perturbations in gene expression.","category":"page"},{"location":"forecastMLData/","page":"Metabolic Labeling for Forecasts","title":"Metabolic Labeling for Forecasts","text":"futureExpressionPredsP = predictCellFutures(trainedNetwork[1], t1_60min, 24,\n perturbGenes = geneNames[1:2], geneNames = geneNames,\n perturbationLevels = [2.0f0, 0.0f0])","category":"page"},{"location":"forecastMLData/","page":"Metabolic Labeling for Forecasts","title":"Metabolic Labeling for Forecasts","text":"Once we have forecast expression levels for each gene, we may want to know which genes expression levels change the most over time, as these are likely to be important in ongoing biological process we are attempting to model. To assay this we simply run mostTimeVariableGenes which outputs a table of genes ordered by the most variable over predicted time points.","category":"page"},{"location":"forecastMLData/","page":"Metabolic Labeling for Forecasts","title":"Metabolic Labeling for Forecasts","text":"geneOutputTable = mostTimeVariableGenes(futureExpressionPreds, geneNames)","category":"page"},{"location":"forecastMLData/","page":"Metabolic Labeling for Forecasts","title":"Metabolic Labeling for Forecasts","text":"We can save an RNAForecaster network for later use with:","category":"page"},{"location":"forecastMLData/","page":"Metabolic Labeling for Forecasts","title":"Metabolic Labeling for Forecasts","text":"saveForecaster(trainedNetwork, \"exampleForecaster.jld2\")","category":"page"},{"location":"forecastMLData/","page":"Metabolic Labeling for Forecasts","title":"Metabolic Labeling for Forecasts","text":"which we can load back into memory using","category":"page"},{"location":"forecastMLData/","page":"Metabolic Labeling for Forecasts","title":"Metabolic Labeling for Forecasts","text":"loadedNetwork = loadForecaster(\"exampleForecaster.jld2\")","category":"page"},{"location":"forecastMLData/","page":"Metabolic Labeling for Forecasts","title":"Metabolic Labeling for Forecasts","text":"WARNING: If you update Flux.jl or DiffEqFlux.jl, saved networks are not guaranteed to be forward compatible.","category":"page"},{"location":"splicedData/#Predictions-from-Splicing-Data","page":"Predictions from Splicing Data","title":"Predictions from Splicing Data","text":"","category":"section"},{"location":"splicedData/","page":"Predictions from Splicing Data","title":"Predictions from Splicing Data","text":"RNAForecaster requires transcriptomics data from the same cell at two different points in time. The best way to acquire this sort of information from standard single cell RNA-seq protocols is to leverage incidental capture of introns to determine whether a transcript is spliced or unspliced. In general, it is a reasonable assumption that spliced transcripts were produced more recently than unspliced, thus giving some temporal information on the sequenced transcripts.","category":"page"},{"location":"splicedData/","page":"Predictions from Splicing Data","title":"Predictions from Splicing Data","text":"Unfortunately, the spliced and unspliced count matrices thus generated cannot be considered two equivalent time points as the count matrices come from very different distributions, which prevents us from being able to straightforwardly predict future expression states from one time point to the next. We could attempt to handle this problem using RNA velocity style assumptions about the splicing and degradation rates to update our predicted expression levels at each time point. However, with RNAForecaster this process will propagate too much error for the resulting predictions to be of value.","category":"page"},{"location":"splicedData/","page":"Predictions from Splicing Data","title":"Predictions from Splicing Data","text":"Instead, we make a single prediction about the direction of regulation, which can be compared to predictions under perturbations, allowing us to uncover key genes relating to the biological process of interest.","category":"page"},{"location":"splicedData/","page":"Predictions from Splicing Data","title":"Predictions from Splicing Data","text":"Here we will use a small subset of murine pancreatic development data published by Bastidas-Ponce et al. (2019) to demonstrate how RNAForecaster is used with splicing data.","category":"page"},{"location":"splicedData/","page":"Predictions from Splicing Data","title":"Predictions from Splicing Data","text":"#read in data\nusing DelimitedFiles\nsplicedSub = readdlm(\"data/pancSplicedExampleData.csv\", ',')\nunsplicedSub = readdlm(\"data/pancUnsplicedExampleData.csv\", ',')\ngeneNamesSub = readdlm(\"data/pancGeneNamesExampleData.csv\", ',')\ngeneNamesSub = string.(geneNamesSub)[:,1]\n\n#data must be log normalized and converted to Float32\nsplicedSub = Float32.(log1p.(splicedSub))\nunsplicedSub = Float32.(log1p.(unsplicedSub))\n\n#train RNAForecaster\ntrainedModel = trainRNAForecaster(splicedSub, unsplicedSub);","category":"page"},{"location":"splicedData/","page":"Predictions from Splicing Data","title":"Predictions from Splicing Data","text":"So we now have a neural network that can predicted unspliced counts based on spliced counts. We can now perturb each gene to see how the predictions of unspliced counts change. This can help uncover regulatory relationships and important genes underlying the system's biology.","category":"page"},{"location":"splicedData/","page":"Predictions from Splicing Data","title":"Predictions from Splicing Data","text":"Here we will simulate a KO of each gene of the 10 genes in the data from the initial conditions of 25 random cells.","category":"page"},{"location":"splicedData/","page":"Predictions from Splicing Data","title":"Predictions from Splicing Data","text":"KOEffects = perturbEffectPredictions(trainedModel[1], splicedSub, 25);","category":"page"},{"location":"splicedData/","page":"Predictions from Splicing Data","title":"Predictions from Splicing Data","text":"We can search these results for statistically significant effects.","category":"page"},{"location":"splicedData/","page":"Predictions from Splicing Data","title":"Predictions from Splicing Data","text":"findSigRegulation(KOEffects, geneNamesSub)","category":"page"},{"location":"splicedData/","page":"Predictions from Splicing Data","title":"Predictions from Splicing Data","text":"We can also specify specific genes and specific perturbations.","category":"page"},{"location":"splicedData/","page":"Predictions from Splicing Data","title":"Predictions from Splicing Data","text":"perturbEffects = perturbEffectPredictions(trainedModel[1], splicedSub, 25,\n  perturbGenes = geneNamesSub[[2,5]], geneNames = geneNamesSub,\n  perturbLevels = [1.0f0, 1.5f0]);","category":"page"},{"location":"splicedData/","page":"Predictions from Splicing Data","title":"Predictions from Splicing Data","text":"perturbEffectPredictions","category":"page"},{"location":"splicedData/#RNAForecaster.perturbEffectPredictions","page":"Predictions from Splicing Data","title":"RNAForecaster.perturbEffectPredictions","text":"perturbEffectPredictions(trainedNetwork, splicedData::Matrix{Float32}, nCells::Int;      perturbGenes::Vector{String} = Vector{String}(undef, 0), geneNames::Vector{String} = Vector{String}(undef,0),      seed::Int=123)\n\nBased on spliced/unspliced counts, predict the immediate transcriptomic effect of any or all single gene perturbations. Outputs a tuple containing the cells used for prediction, the expression predictions, the gene wise differences, and the cell-wise euclidean distances for each perturbation.\n\nThis function is capable of running on multiple parallel processes using Distributed.jl. Call addprocs(n) before running the function to add parallel workers, where n is the number of additional processes desired.\n\nRequired Arguments\n\ntrainedNetwork - trained neuralODE from trainRNAForecaster\nsplicedData - log normalized spliced counts matrix. Must be in Float32 format\nnCells - how many cells from the data should be used for prediction of perturb effect.\n\nHigher values will increase computational time required.\n\nOptional Arguments\n\nperturbGenes - list of genes to simulate a perturbation of. By default all genes are used\ngeneNames - if providing a subset of the genes to perturb, a vector of gene names to\n\nmatch against, in the order of splicedData\n\nperturbLevels - list of perturbation levels to use for each perturbed gene. By default\n\nall genes are set to zero, simulating a KO.\n\nseed - Random seed for reproducibility on the cells chosen for prediction\n\n\n\n\n\n","category":"function"},{"location":"splicedData/","page":"Predictions from Splicing Data","title":"Predictions from Splicing Data","text":"Convenience functions are provided to search the results of the perturbations","category":"page"},{"location":"splicedData/","page":"Predictions from Splicing Data","title":"Predictions from Splicing Data","text":"totalPerturbImpact\ngenePerturbExpressionChanges\ngeneResponseToPerturb","category":"page"},{"location":"splicedData/#RNAForecaster.totalPerturbImpact","page":"Predictions from Splicing Data","title":"RNAForecaster.totalPerturbImpact","text":"totalPerturbImpact(perturbData, geneNames::Vector{String})\n\nFunction to yield a sorted data frame of the size of the predicted effect of a perturbation on the cellular transcriptome. Intended to serve as a measure of more or less impactful gene perturbations.\n\nRequired Arguments\n\nperturbData - results from perturbEffectPredictions function\ngeneNames - vector of gene names in the order of the input expression data.\n\nShould only include perturbed genes\n\n\n\n\n\n","category":"function"},{"location":"splicedData/#RNAForecaster.genePerturbExpressionChanges","page":"Predictions from Splicing Data","title":"RNAForecaster.genePerturbExpressionChanges","text":"genePerturbExpressionChanges(perturbData, geneNames::Vector{String}, perturbGene::String;     genesperturbd::Vector{String} = geneNames)\n\nFunction to get a sorted data frame of the predicted effect of a gene perturb on all other genes.\n\nRequired Arguments\n\nperturbData - results from perturbEffectPredictions function\ngeneNames - vector of gene names in the order of the input expression data\nperturbGene - a gene name to query the predicted perturb effect on expression\n\nOptional Arguments\n\ngenesPerturbed - If less than all the gene perturbs were performed, the ordered names of the perturb genes must be supplied\n\n\n\n\n\n","category":"function"},{"location":"splicedData/#RNAForecaster.geneResponseToPerturb","page":"Predictions from Splicing Data","title":"RNAForecaster.geneResponseToPerturb","text":"geneResponseToPerturb(perturbData, geneNames::Vector{String}, geneOfInterest::String;     genesPerturbed::Vector{String} = geneNames)\n\nFunction to get a sorted data frame of the predicted effect of all other gene perturbations on a particular gene of interest.\n\nRequired Arguments\n\nperturbData - results from perturbEffectPredictions function\ngeneNames - vector of gene names in the order of the input expression data\ngeneOfInterest - a gene name to query\n\n\n\n\n\n","category":"function"},{"location":"splicedData/","page":"Predictions from Splicing Data","title":"Predictions from Splicing Data","text":"For example, we can get a list of the genes, and the size of the impact of the KO on the rest of the genes' expression, measured in euclidean distance, using","category":"page"},{"location":"splicedData/","page":"Predictions from Splicing Data","title":"Predictions from Splicing Data","text":"totalPerturbImpact(KOEffects, geneNamesSub)","category":"page"},{"location":"#RNAForecaster.jl-Estimating-Future-Expression-States-on-Short-Time-Scales","page":"Home","title":"RNAForecaster.jl - Estimating Future Expression States on Short Time Scales","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"RNAForecaster is based on a neural network that is trained using ordinary differential equations solvers (neural ODEs). See (https://arxiv.org/abs/1806.07366) for details on neural ODEs. The goal of this method is to forecast future expression levels in a cell given transcriptomic data from said cell. Much like a weather forecaster, it can only make predictions accurately in the short to medium term and even then there will be times when it does not get it right. Despite these limitations, weather forecasts are still useful, and hopefully RNA forecasts will be useful for you too.","category":"page"},{"location":"#Installation","page":"Home","title":"Installation","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Download Julia 1.6 or later, if you haven't already.","category":"page"},{"location":"","page":"Home","title":"Home","text":"You can add RNAForecaster from using Julia's package manager, by typing ] add RNAForecaster in the Julia prompt.","category":"page"},{"location":"training/#Training-and-Forecasting","page":"Training and Forecasting","title":"Training and Forecasting","text":"","category":"section"},{"location":"training/","page":"Training and Forecasting","title":"Training and Forecasting","text":"Training RNAForecaster requires two expression count matrices. These count matrices should be formatted with genes  as rows and cells as columns. Each matrix should represent two different time points from the same cells. This can be accomplished from transcriptomic profiling using spliced and unspliced counts or by using labeled and unlabeled counts from metabolic labeling scRNA-seq protocols such as scEU-seq (see Battich et al. 2020).","category":"page"},{"location":"training/","page":"Training and Forecasting","title":"Training and Forecasting","text":"Here, we will generate two random matrices for illustrative purposes.","category":"page"},{"location":"training/","page":"Training and Forecasting","title":"Training and Forecasting","text":"testT0 = log1p.(Float32.(abs.(randn(10,1000))))\ntestT1 = log1p.(0.5f0 .* testT0)","category":"page"},{"location":"training/","page":"Training and Forecasting","title":"Training and Forecasting","text":"Note that the input matrices are expected to be of type Float32 and log transformed, as shown above.","category":"page"},{"location":"training/#Training","page":"Training and Forecasting","title":"Training","text":"","category":"section"},{"location":"training/","page":"Training and Forecasting","title":"Training and Forecasting","text":"To train the neural ODE network we call the trainRNAForecaster function.","category":"page"},{"location":"training/","page":"Training and Forecasting","title":"Training and Forecasting","text":"testForecaster = trainRNAForecaster(testT0, testT1);","category":"page"},{"location":"training/","page":"Training and Forecasting","title":"Training and Forecasting","text":"In the simplest case, we only need to input the matrices, but there are several options provided to modify the training of the neural network, as shown below.","category":"page"},{"location":"training/","page":"Training and Forecasting","title":"Training and Forecasting","text":"trainRNAForecaster","category":"page"},{"location":"training/#RNAForecaster.trainRNAForecaster","page":"Training and Forecasting","title":"RNAForecaster.trainRNAForecaster","text":"trainRNAForecaster(expressionDataT0::Matrix{Float32}, expressionDataT1::Matrix{Float32};      trainingProp::Float64 = 0.8, hiddenLayerNodes::Int = 2*size(expressionDataT0)[1],      shuffleData::Bool = true, seed::Int = 123, learningRate::Float64 = 0.005,      nEpochs::Int = 10, batchsize::Int = 100, checkStability::Bool = false, iterToCheck::Int = 50,      stabilityThreshold::Float32 = 2*maximum(expressionDataT0), stabilityChecksBeforeFail::Int = 5,      useGPU::Bool = false)\n\nFunction to train RNAForecaster based on expression data. Main input is two matrices representing expression data from two different time points in the same cell. This can be either based on splicing or metabolic labeling currently. Each should be log normalized and have genes as rows and cells as columns.\n\nRequired Arguments\n\nexpressionDataT0 - Float32 Matrix of log-normalized expression counts in the format of genes x cells\nexpressionDataT1 - Float32 Matrix of log-normalized expression counts in the format\n\nof genes x cells from a time after expressionDataT0\n\nKeyword Arguments\n\ntrainingProp - proportion of the data to use for training the model, the rest will be\n\nused for a validation set. If you don't want a validation set, this value can be set to 1.0\n\nhiddenLayerNodes - number of nodes in the hidden layer of the neural network\nshuffleData - should the cells be randomly shuffled before training\nseed - random seed\nlearningRate - learning rate for the neural network during training\nnEpochs - how many times should the neural network be trained on the data.\n\nGenerally yields small gains in performance, can be lowered to speed up the training process\n\nbatchsize - batch size for training\ncheckStability - should the stability of the networks future time predictions be checked,\n\nretraining the network if unstable?\n\niterToCheck - when checking stability, how many future time steps should be predicted?\nstabilityThreshold - when checking stability, what is the maximum gene variance allowable across predictions?\nstabilityChecksBeforeFail - when checking stability, how many times should the network\n\nbe allowed to retrain before an error is thrown? Used to prevent an infinite loop.\n\nuseGPU - use a GPU to train the neural network? highly recommended for large data sets, if available\n\n\n\n\n\n","category":"function"},{"location":"training/","page":"Training and Forecasting","title":"Training and Forecasting","text":"For example, by default RNAForecaster partitions the input data into a training and a validation set. If we want the neural network to be trained on the entire data set, we can set trainingProp = 1.0.","category":"page"},{"location":"training/","page":"Training and Forecasting","title":"Training and Forecasting","text":"When using larger data sets, such as a matrix from a normal scRNAseq experiment which may contain thousands of variable genes and tens of thousands of cells, it becomes inefficient to train the network on a CPU. If a GPU is available, setting useGPU = true can massively speed up the training process.","category":"page"},{"location":"training/","page":"Training and Forecasting","title":"Training and Forecasting","text":"#Forecasting","category":"page"},{"location":"training/","page":"Training and Forecasting","title":"Training and Forecasting","text":"Once we have trained the neural network, we can use it to forecast future expression states. For example, to predict the next fifty time points from our test data, we could run:","category":"page"},{"location":"training/","page":"Training and Forecasting","title":"Training and Forecasting","text":"testOut1 = predictCellFutures(testForecaster[1], testT0, 50);","category":"page"},{"location":"training/","page":"Training and Forecasting","title":"Training and Forecasting","text":"The predictions can also be conditioned on arbitrary perturbations in gene expression.","category":"page"},{"location":"training/","page":"Training and Forecasting","title":"Training and Forecasting","text":"geneNames = [\"A\", \"B\", \"C\", \"D\", \"E\", \"F\", \"G\", \"H\", \"I\", \"J\"]\ntestOut2 = predictCellFutures(testForecaster[1], testT0, 50, perturbGenes = [\"A\", \"B\", \"F\"],\ngeneNames = geneNames, perturbationLevels = [1.0f0, 2.0f0, 0.0f0]);","category":"page"},{"location":"training/","page":"Training and Forecasting","title":"Training and Forecasting","text":"All options for predictCellFutures are shown here:","category":"page"},{"location":"training/","page":"Training and Forecasting","title":"Training and Forecasting","text":"predictCellFutures","category":"page"},{"location":"training/#RNAForecaster.predictCellFutures","page":"Training and Forecasting","title":"RNAForecaster.predictCellFutures","text":"predictCellFutures(trainedNetwork, expressionData::Matrix{Float32}, tSteps::Int;      perturbGenes::Vector{String} = Vector{String}(undef,0), geneNames::Vector{String} = Vector{String}(undef,0),      perturbationLevels::Vector{Float32} = Vector{Float32}(undef,0),      enforceMaxPred::Bool = true, maxPrediction::Float32 = 2*maximum(expressionData))\n\nFunction to make future expression predictions using a trained neural ODE outputs a 3d tensor containing a predicted expression counts matrix for the cell at each time step\n\nRequired Arguments\n\ntrainedNetwork - the trained neural ODE, from the trainRNAForecaster function\nexpressionData - the initial expression states that should be used to make predictions from\ntSteps - how many future time steps should be predicted. (Error will propagate\n\nwith each prediction so predictions will eventually become highly innaccurate at high numbers of time steps)\n\nKeyword Arguments\n\nperturbGenes - a vector of gene names that will have their values set to a constant 'perturbed' level.\ngeneNames - a vector of gene names in the order of the rows of the expressionData.\n\nUsed only when simulating perturbations.\n\nperturbationLevels - a vector of Float32, corresponding to the level each perturbed\n\ngene's expression should be set at.\n\nenforceMaxPred - should a maximum allowed prediction be enforced? This is used\n\nto represent prior knowledge about what sort of expression values are remotely reasonable predictions.\n\nmaxPrediction - if enforcing a maximum prediction, what should the value be?\n\n2 times the maximum of the input expression data by default (in log space).\n\n\n\n\n\n","category":"function"},{"location":"training/","page":"Training and Forecasting","title":"Training and Forecasting","text":"Once we have forecast expression levels for each gene, we may want to know which genes expression levels change the most over time, as these are likely to be important in ongoing biological process we are attempting to model. To assay this we simply run mostTimeVariableGenes which outputs a table of genes ordered by the most variable over predicted time points.","category":"page"},{"location":"training/","page":"Training and Forecasting","title":"Training and Forecasting","text":"geneOutputTable = mostTimeVariableGenes(testOut1, geneNames)","category":"page"},{"location":"training/","page":"Training and Forecasting","title":"Training and Forecasting","text":"mostTimeVariableGenes","category":"page"},{"location":"training/#RNAForecaster.mostTimeVariableGenes","page":"Training and Forecasting","title":"RNAForecaster.mostTimeVariableGenes","text":"mostTimeVariableGenes(cellFutures::AbstractArray{Float32}, geneNames::Vector{String};      statType = \"mean\")\n\nFor each cell, takes the predicted expression levels of each gene over time and finds the variance with respect to predicted time points. Then get the mean/median for each gene's variance across cells for each gene.\n\nOutputs a sorted DataFrame containing gene names and the variances over predicted time.\n\nRequired Arguments\n\ncellFutures - a 3D tensor of gene expression over time; the output from predictCellFutures\ngeneNames - a vector of gene names corresponding to the order of the genes in cellFutures\n\nOptional Arguments\n\nstatType - How to summarize the gene variances. Valid options are \"mean\" or \"median\"\n\n\n\n\n\n","category":"function"}]
}