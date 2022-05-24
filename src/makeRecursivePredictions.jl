"""
`predictCellFutures(trainedNetwork, expressionData::Matrix{Float32}, tSteps::Int;
     perturbGenes::Vector{String} = Vector{String}(undef,0), geneNames::Vector{String} = Vector{String}(undef,0),
     perturbationLevels::Vector{Float32} = Vector{Float32}(undef,0),
     enforceMaxPred::Bool = true, maxPrediction::Float32 = 2*maximum(expressionData))`

Function to make future expression predictions using a trained neural ODE
outputs a 3d tensor containing a predicted expression counts matrix for the
cell at each time step
# Required Arguments
* trainedNetwork - the trained neural ODE, from the trainRNAForecaster function
* expressionData - the initial expression states that should be used to make predictions from
* tSteps - how many future time steps should be predicted. (Error will propagate
 with each prediction so predictions will eventually become highly innaccurate at high numbers of time steps)
# Keyword Arguments
* perturbGenes - a vector of gene names that will have their values set to a constant 'perturbed' level.
* geneNames - a vector of gene names in the order of the rows of the expressionData.
Used only when simulating perturbations.
* perturbationLevels - a vector of Float32, corresponding to the level each perturbed
 gene's expression should be set at.
* enforceMaxPred - should a maximum allowed prediction be enforced? This is used
 to represent prior knowledge about what sort of expression values are remotely reasonable predictions.
* maxPrediction - if enforcing a maximum prediction, what should the value be?
2 times the maximum of the input expression data by default (in log space).
"""
function predictCellFutures(trainedNetwork, expressionData::Matrix{Float32}, tSteps::Int;
     perturbGenes::Vector{String} = Vector{String}(undef,0), geneNames::Vector{String} = Vector{String}(undef,0),
     perturbationLevels::Vector{Float32} = Vector{Float32}(undef,0),
     enforceMaxPred::Bool = true, maxPredictionMult::Float32 = 2.0f0,
     useGPU::Bool = false, batchsize::Int = 100)


     if useGPU
         #load data and network onto the gpu
         trainedNetwork = trainedNetwork |> gpu
     end

     if enforceMaxPred
         geneMaxes = Array{Float32}(undef, size(expressionData)[1])
         for i=1:size(expressionData)[1]
             geneMaxes[i] = maximum(expressionData[i,:])
         end
     end

    if length(perturbGenes) == 0
        inputData = copy(expressionData)
        predictions = Array{Float32}(undef, size(expressionData)[1], size(expressionData)[2], tSteps)

        for i=1:tSteps
            #allows the data to be processed by the neural network simultaneously
            if useGPU
                inputData = ([inputData[:,k] for k in partition(1:size(inputData)[2], batchsize)]) |> gpu
            else
                inputData = ([inputData[:,k] for k in partition(1:size(inputData)[2], batchsize)])
            end

            for j=1:length(inputData)
                if j*batchsize > size(expressionData)[2] #deals with batchsize not being an exact multiple of number of cells
                    predictions[:,(1+((j-1)*batchsize)):size(expressionData)[2],i] = cpu(trainedNetwork(inputData[[j]]...)[1])
                else
                    predictions[:,(1+((j-1)*batchsize)):(j*batchsize),i] = cpu(trainedNetwork(inputData[[j]]...)[1])
                end
            end
            #set negative predictions to zero
            predictions[findall(x->x < 0, predictions)] .= 0

            if enforceMaxPred
                for m=1:size(predictions)[1]
                    predictions[m,findall(x->x > (geneMaxes[m] * maxPredictionMult), predictions[m,:,i]),i] .= (geneMaxes[m] * maxPredictionMult)
                end
            end

            inputData = predictions[:,:,i]

        end

    else
        if length(geneNames) != size(expressionData)[1]
            error("Length of gene names is not equal to the number of rows (genes)
            in the input data.")
        end

        if length(perturbGenes) != length(perturbationLevels)
            error("perturbGenes and perturbationLevels must be the same length.")
        end

        inputData = copy(expressionData)
        #set perturbation gene levels
        perturbGeneInds = findall(in(perturbGenes), geneNames)
        inputData[perturbGeneInds,:] .= perturbationLevels[sortperm(perturbGeneInds)]
        predictions = Array{Float32}(undef, size(expressionData)[1], size(expressionData)[2], tSteps)

        for i=1:tSteps
            #allows the data to be processed by the neural network simultaneously
            if useGPU
                inputData = ([inputData[:,k] for k in partition(1:size(inputData)[2], batchsize)]) |> gpu
            else
                inputData = ([inputData[:,k] for k in partition(1:size(inputData)[2], batchsize)])
            end

            for j=1:length(inputData)
                if j*batchsize > size(expressionData)[2] #deals with batchsize not being an exact multiple of number of cells
                    predictions[:,(1+((j-1)*batchsize)):size(expressionData)[2],i] = cpu(trainedNetwork(inputData[[j]]...)[1])
                else
                    predictions[:,(1+((j-1)*batchsize)):(j*batchsize),i] = cpu(trainedNetwork(inputData[[j]]...)[1])
                end
            end
            #set negative predictions to zero
            predictions[findall(x->x < 0, predictions)] .= 0
            #set perturb gene expression levels
            predictions[perturbGeneInds,:,i] .= perturbationLevels[sortperm(perturbGeneInds)]

            if enforceMaxPred
                for m=1:size(predictions)[1]
                    predictions[m,findall(x->x > (geneMaxes[m] * maxPredictionMult), predictions[m,:,i]),i] .= geneMaxes[m]
                end
            end

            inputData = predictions[:,:,i]
        end
    end

    return predictions
end


function ensembleExpressionPredictions(networks, expressionData::Matrix{Float32}, tSteps::Int;
     perturbGenes::Vector{String} = Vector{String}(undef,0), geneNames::Vector{String} = Vector{String}(undef,0),
     perturbationLevels::Vector{Float32} = Vector{Float32}(undef,0),
     enforceMaxPred::Bool = true, maxPredictionMult::Float32 = 2.0f0,
     useGPU::Bool = false, batchsize::Int = 100)


     if useGPU && nprocs() > 1
         error("Creating multiple GPU processes is not currently supported.")
     end

     predictionData = Vector{Any}(undef, length(networks))
     if nprocs() > 1
         for i = 1:length(networks)
             predictionData[i] = @spawn predictCellFutures(networks[i][1], expressionData,
                  tSteps, perturbGenes= perturbGenes, geneNames = geneNames,
                  perturbationLevels = perturbationLevels,
                  enforceMaxPred = enforceMaxPred, maxPredictionMult = maxPredictionMult,
                  useGPU = useGPU, batchsize = batchsize)
        end

        predictionData = fetch.(predictionData)
    else
        for i = 1:length(networks)
            predictionData[i] = predictCellFutures(networks[i][1], expressionData,
                 tSteps, perturbGenes= perturbGenes, geneNames = geneNames,
                 perturbationLevels = perturbationLevels,
                 enforceMaxPred = enforceMaxPred, maxPredictionMult = maxPredictionMult,
                 useGPU = useGPU, batchsize = batchsize)
       end
    end

    #wrangle the data shape and get median predictions from networks
    predData = Array{Float32}(undef, tSteps, length(predictionData), size(expressionData)[1], size(expressionData)[2])
    for i=1:length(predictionData)
        predData[:,i,:,:] = permutedims(predictionData[i], (3,1,2))
    end

    results = permutedims(median(predData, dims=2)[:,1,:,:], (2,3,1))

    return results

end

"""
`mostTimeVariableGenes(cellFutures::AbstractArray{Float32}, geneNames::Vector{String};
     statType = "mean")`

For each cell, takes the predicted expression levels of each gene over time
and finds the variance with respect to predicted time points. Then get the
mean/median for each gene's variance across cells for each gene.

Outputs a sorted DataFrame containing gene names and the variances over predicted time.

# Required Arguments
* cellFutures - a 3D tensor of gene expression over time; the output from predictCellFutures
* geneNames - a vector of gene names corresponding to the order of the genes in cellFutures
# Optional Arguments
* statType - How to summarize the gene variances. Valid options are "mean" or "median"
"""
function mostTimeVariableGenes(cellFutures::AbstractArray{Float32}, geneNames::Vector{String};
     statType = "mean")
    vars = Array{Float32}(undef, size(cellFutures)[1], size(cellFutures)[2])
    for i=1:size(cellFutures)[1]
        for j=1:size(cellFutures)[2]
            vars[i,j] = var(cellFutures[i,j,:])
        end
    end
     if statType == "mean"
         stats = mean(vars, dims=2)
         #put into data frame with gene names
         geneData = DataFrame(GeneNames = geneNames, MeanVariance = vec(stats))
         #sort
         sort!(geneData, [:MeanVariance], rev= true)
     elseif statType == "median"
         stats = median(vars, dims=2)
         #put into data frame with gene names
         geneData = DataFrame(GeneNames = geneNames, MedianVariance = vec(stats))
         #sort
         sort!(geneData, [:MedianVariance], rev= true)
     else
         error("Not a valid statType. Use 'mean' or 'median'")
     end

     return geneData

end
