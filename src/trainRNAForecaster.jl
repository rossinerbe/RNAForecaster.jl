function minZero(num::AbstractArray{<:Number})
    num[findall(x->x < 0, num)] .= 0
    num
end

#function to check model prediction stability for each gene
function checkModelStability(model, initialConditions::Matrix{Float32},
     iterToCheck::Int, expressionThreshold::Float32, useGPU::Bool)

    println("Checking Model Stability...")

    inputData = copy(initialConditions)
    recordedVals = Array{Float32}(undef, size(inputData)[1], size(inputData)[2], iterToCheck)

    for i=1:iterToCheck
        recordedVals[:,:,i] = inputData

        #calculate model predictions
        if useGPU
            exprPreds = Matrix{Float32}(undef, size(inputData)[1], size(inputData)[2]) |> gpu
            inputData = inputData |> gpu
            for x=1:size(exprPreds)[2]
                pred = model(inputData[:,x])[1]
                exprPreds[:,x] = minZero(pred)
            end
            inputData = cpu(copy(exprPreds))
        else
            exprPreds = Matrix{Float32}(undef, size(inputData)[1], size(inputData)[2])
            for x=1:size(exprPreds)[2]
                pred = model(inputData[:,x])[1]
                exprPreds[:,x] = minZero(pred)
            end
            inputData = copy(exprPreds)
        end


    end

    #check expression levels
    if length(findall(x->x > expressionThreshold, recordedVals)) == 0
        println("Model is stable, continuing")
        return(false)
    else
        println("Model is unstable, retraining...")
        return(true)
    end
end

#function to calculate the mean loss across all examples in the training/validation set
function meanLoss(data_loader, model)
    ls = 0.0f0
    num = 0
    for (x, y) in data_loader
        ŷ = model(x)[1]
        ls += mse(ŷ, y)
        num +=  size(x)[end]
    end
    return ls / num
end


#train the neural ODE
function trainNetwork(trainData, nGenes::Int,
     hiddenLayerNodes::Int, learningRate::Float64, nEpochs::Int, useGPU::Bool)

     if useGPU
        nn = Chain(Dense(nGenes, hiddenLayerNodes, relu),
                   Dense(hiddenLayerNodes, nGenes)) |> gpu
        model = NeuralODE(nn, (0.0f0, 1.0f0), Tsit5(),
                           save_everystep = false,
                           reltol = 1e-3, abstol = 1e-3,
                           save_start = false)|> gpu
    else
        nn = Chain(Dense(nGenes, hiddenLayerNodes, relu),
                   Dense(hiddenLayerNodes, nGenes))
        model = NeuralODE(nn, (0.0f0, 1.0f0), Tsit5(),
                           save_everystep = false,
                           reltol = 1e-3, abstol = 1e-3,
                           save_start = false)
    end


    loss(x,y) = mse(model(x)[1], y)
    opt = ADAM(learningRate)

    losses = Vector{Float32}(undef, nEpochs)
    for epoch = 1:nEpochs
      for d in trainData
        gs = gradient(Flux.params(model)) do
          l = loss(d...)
        end
        Flux.Optimise.update!(opt, Flux.params(model), gs)
      end
      losses[epoch]= meanLoss(trainData, model)
    end

    return (model, losses)
end

#train the neural ODE with validation set
function trainNetworkVal(trainData, valData, nGenes::Int, hiddenLayerNodes::Int,
     learningRate::Float64, nEpochs::Int, useGPU::Bool)

     if useGPU
        nn = Chain(Dense(nGenes, hiddenLayerNodes, relu),
                   Dense(hiddenLayerNodes, nGenes)) |> gpu
        model = NeuralODE(nn, (0.0f0, 1.0f0), Tsit5(),
                           save_everystep = false,
                           reltol = 1e-3, abstol = 1e-3,
                           save_start = false)|> gpu
    else
        nn = Chain(Dense(nGenes, hiddenLayerNodes, relu),
                   Dense(hiddenLayerNodes, nGenes))
        model = NeuralODE(nn, (0.0f0, 1.0f0), Tsit5(),
                           save_everystep = false,
                           reltol = 1e-3, abstol = 1e-3,
                           save_start = false)
    end

   opt = ADAM(learningRate)
   loss(x,y) = mse(model(x)[1], y)

   losses = Matrix{Float32}(undef, nEpochs, 2)
   for epoch = 1:nEpochs
     for d in trainData
       gs = gradient(Flux.params(model)) do
         l = loss(d...)
       end
       Flux.Optimise.update!(opt, Flux.params(model), gs)
     end
     # Report on train and validation error
     losses[epoch,1]= meanLoss(trainData, model)
     losses[epoch,2] = meanLoss(valData, model)
   end

   return (model, losses)
end

"""
`trainRNAForecaster(expressionDataT1::Matrix{Float32}, expressionDataT2::Matrix{Float32};
     trainingProp::Float64 = 0.8, hiddenLayerNodes::Int = 2*size(expressionDataT1)[1],
     shuffleData::Bool = true, seed::Int = 123, learningRate::Float64 = 0.005,
     nEpochs::Int = 10, batchsize::Int = 100, checkStability::Bool = true, iterToCheck::Int = 50,
     stabilityThreshold::Float32 = 2*maximum(expressionDataT1), stabilityChecksBeforeFail::Int = 5,
     useGPU::Bool = false)`

Function to train RNAForecaster based on expression data. Main input is two
matrices representing expression data from two different time points in the same cell.
This can be either based on splicing or metabolic labeling currently.
Each should be log normalized and have genes as rows and cells as columns.

# Required Arguments
* expressionDataT1 - Float32 Matrix of log-normalized expression counts in the format of genes x cells
* expressionDataT2 - Float32 Matrix of log-normalized expression counts in the format
 of genes x cells from a time after expressionDataT1
# Keyword Arguments
* trainingProp - proportion of the data to use for training the model, the rest will be
 used for a validation set. If you don't want a validation set, this value can be set to 1.0
* hiddenLayerNodes - number of nodes in the hidden layer of the neural network
* shuffleData - should the cells be randomly shuffled before training
* seed - random seed
* learningRate - learning rate for the neural network during training
* nEpochs - how many times should the neural network be trained on the data.
 Generally yields small gains in performance, can be lowered to speed up the training process
* batchsize - batch size for training
* checkStability - should the stability of the networks future time predictions be checked,
 retraining the network if unstable?
* iterToCheck - when checking stability, how many future time steps should be predicted?
* stabilityThreshold - when checking stability, what is the maximum gene variance allowable across predictions?
* stabilityChecksBeforeFail - when checking stability, how many times should the network
 be allowed to retrain before an error is thrown? Used to prevent an infinite loop.
* useGPU - use a GPU to train the neural network? highly recommended for large data sets, if available

"""

function trainRNAForecaster(expressionDataT1::Matrix{Float32}, expressionDataT2::Matrix{Float32};
     trainingProp::Float64 = 0.8, hiddenLayerNodes::Int = 2*size(expressionDataT1)[1],
     shuffleData::Bool = true, seed::Int = 123, learningRate::Float64 = 0.005,
     nEpochs::Int = 10, batchsize::Int = 100, checkStability::Bool = true, iterToCheck::Int = 50,
     stabilityThreshold::Float32 = 2*maximum(expressionDataT1), stabilityChecksBeforeFail::Int = 5,
     useGPU::Bool = false)

     println("Loading Data...")
      #randomly shuffle the input data cells
      if shuffleData
         Random.seed!(seed)
         shuffling = shuffle(1:size(expressionDataT1)[2])
         expressionDataT1 = expressionDataT1[:,shuffling]
         expressionDataT2 = expressionDataT2[:,shuffling]
     end

     #get the number of genes in the data
     nGenes = size(expressionDataT1)[1]

     if trainingProp < 1.0
         #subset the data into training and validation sets
         #determine how many cells should be in training set
         cellsInTraining = Int(round(size(expressionDataT1)[2]*trainingProp))

         trainX = expressionDataT1[:,1:cellsInTraining]
         trainY = expressionDataT2[:,1:cellsInTraining]
         valX = expressionDataT1[:,cellsInTraining+1:size(expressionDataT1)[2]]
         valY = expressionDataT2[:,cellsInTraining+1:size(expressionDataT1)[2]]

         if useGPU
             trainData = ([(trainX[:,i], trainY[:,i]) for i in partition(1:size(trainX)[2], batchsize)]) |> gpu
             valData = ([(valX[:,i], valY[:,i]) for i in partition(1:size(valX)[2], batchsize)]) |> gpu
         else
             trainData = ([(trainX[:,i], trainY[:,i]) for i in partition(1:size(trainX)[2], batchsize)])
             valData = ([(valX[:,i], valY[:,i]) for i in partition(1:size(valX)[2], batchsize)])
         end

         println("Training model...")
         if checkStability
             iter = 1
             #repeat until model is stable or until user defined break point
             while checkStability
                 if iter > stabilityChecksBeforeFail
                     error("Failed to find a stable solution after " * string(iter-1) * " attempts.
                      Try increasing the stabilityChecksBeforeFail variable or, if a slightly less
                      stable solution is acceptable, increase the stabilityThreshold variable.")
                 end

                 #train the neural ODE
                 trainedNet = trainNetworkVal(trainData, valData,
                      nGenes, hiddenLayerNodes, learningRate, nEpochs, useGPU)

                #check model stability
                checkStability = checkModelStability(trainedNet[1], trainX, iterToCheck, stabilityThreshold, useGPU)
                Random.seed!(seed+iter)
                iter +=1
            end
        else
            #train the neural ODE
            trainedNet = trainNetworkVal(trainData, valData,
                 nGenes, hiddenLayerNodes, learningRate, nEpochs, useGPU)
       end

       return trainedNet

    #in the case where we want to use the entire data set to train the model
    #(e.g. if we have already validated its performance and we now want to
    #train using the full data set)
     else
         trainX = expressionDataT1
         trainY = expressionDataT2

         if useGPU
             trainData = ([(trainX[:,i], trainY[:,i]) for i in partition(1:size(trainX)[2], batchsize)]) |> gpu
         else
             trainData = ([(trainX[:,i], trainY[:,i]) for i in partition(1:size(trainX)[2], batchsize)])
         end

         println("Training model...")
         if checkStability
             iter = 1
             #repeat until model is stable or until user defined break point
             while checkStability
                 if iter > stabilityChecksBeforeFail
                     error("Failed to find a stable solution after " * string(iter-1) * " attempts.
                      Try increasing the stabilityChecksBeforeFail variable or, if a slightly less
                      stable solution is acceptable, increase the stabilityThreshold variable.")
                 end

                 #train the neural ODE
                 trainedNet = trainNetwork(trainData, nGenes, hiddenLayerNodes,
                  learningRate, nEpochs, useGPU)

                #check model stability
                checkStability = checkModelStability(trainedNet[1], trainX,
                 iterToCheck, stabilityThreshold, useGPU)
                Random.seed!(seed+iter)
                iter +=1
            end
        else
            #train the neural ODE
            trainedNet = trainNetwork(trainData, nGenes, hiddenLayerNodes,
             learningRate, nEpochs, useGPU)
       end

       return trainedNet
     end
 end

"""
`createEnsembleForecaster(expressionDataT1::Matrix{Float32}, expressionDataT2::Matrix{Float32};
     nNetworks::Int = 5, trainingProp::Float64 = 0.8,
     hiddenLayerNodes::Int = 2*size(expressionDataT1)[1],
     shuffleData::Bool = true, seed::Int = 123, learningRate::Float64 = 0.005,
     nEpochs::Int = 10, batchsize::Int = 100, checkStability::Bool = true,
     iterToCheck::Int = 50, stabilityThreshold::Float32 = 2*maximum(expressionDataT1),
     stabilityChecksBeforeFail::Int = 5, useGPU::Bool = false)`

Function to train multiple neural ODEs to predict expression, allowing an ensembling of
their predicitons, which tends to yield more accurate results on future predictions.
This is because stochastic gradient descent yields slightly different solutions
when given different random seeds. In the training data these solutions yield almost
identical results, but when generalizing to future predictions, the results can
diverge substantially. To account for this, we can average across multiple forecasters.

It is recommended to run this function on a GPU (useGPU = true) or if a GPU is not
available run in parallel. To train the neural networks on separate processes
call
`using Distributed
addprocs(desiredNumberOfParallelProcesses)
@everywhere using RNAForecaster`

# Required Arguments
* expressionDataT1 - Float32 Matrix of log-normalized expression counts in the format of genes x cells
* expressionDataT2 - Float32 Matrix of log-normalized expression counts in the format
 of genes x cells from a time after expressionDataT1
# Keyword Arguments
* nNetworks - number of networks to train
* trainingProp - proportion of the data to use for training the model, the rest will be
 used for a validation set. If you don't want a validation set, this value can be set to 1.0
* hiddenLayerNodes - number of nodes in the hidden layer of the neural network
* shuffleData - should the cells be randomly shuffled before training
* seed - random seed
* learningRate - learning rate for the neural network during training
* nEpochs - how many times should the neural network be trained on the data.
 Generally yields small gains in performance, can be lowered to speed up the training process
* batchsize - batch size for training
* checkStability - should the stability of the networks future time predictions be checked,
 retraining the network if unstable?
* iterToCheck - when checking stability, how many future time steps should be predicted?
* stabilityThreshold - when checking stability, what is the maximum gene variance allowable across predictions?
* stabilityChecksBeforeFail - when checking stability, how many times should the network
 be allowed to retrain before an error is thrown? Used to prevent an infinite loop.
* useGPU - use a GPU to train the neural network? highly recommended for large data sets, if available
"""
function createEnsembleForecaster(expressionDataT1::Matrix{Float32}, expressionDataT2::Matrix{Float32};
     nNetworks::Int = 5, trainingProp::Float64 = 0.8,
     hiddenLayerNodes::Int = 2*size(expressionDataT1)[1],
     shuffleData::Bool = true, seed::Int = 123, learningRate::Float64 = 0.005,
     nEpochs::Int = 10, batchsize::Int = 100, checkStability::Bool = true,
     iterToCheck::Int = 50, stabilityThreshold::Float32 = 2*maximum(expressionDataT1),
     stabilityChecksBeforeFail::Int = 5, useGPU::Bool = false)

     if nprocs() > 1 && useGPU
         error("Using multiple separate julia processes on the GPU is currently not supported")
     end

     if useGPU
         println("Training " * string(nNetworks) * " networks using on the GPU...")
         networks = Vector{Any}(undef, nNetworks)
         for i=1:nNetworks
             seed = seed + ((i-1)*(stabilityChecksBeforeFail+1))
             networks[i] = trainRNAForecaster(expressionDataT1, expressionDataT2,
             trainingProp = trainingProp, hiddenLayerNodes = hiddenLayerNodes,
             shuffleData = shuffleData, seed = seed, learningRate = learningRate,
             nEpochs = nEpochs, batchsize = batchsize, checkStability = checkStability,
             iterToCheck = iterToCheck, stabilityThreshold = stabilityThreshold,
             stabilityChecksBeforeFail = stabilityChecksBeforeFail, useGPU = useGPU)
         end
         return networks
     else
         println("Training " * string(nNetworks) * " networks using " * string(nprocs()) * " parallel processes...")
         networks = Vector{Any}(undef, nNetworks)
         for i=1:nNetworks
             seed = seed + ((i-1)*(stabilityChecksBeforeFail+1))
             networks[i] = @spawn trainRNAForecaster(expressionDataT1, expressionDataT2,
             trainingProp = trainingProp, hiddenLayerNodes = hiddenLayerNodes,
             shuffleData = shuffleData, seed = seed, learningRate = learningRate,
             nEpochs = nEpochs, batchsize = batchsize, checkStability = checkStability,
             iterToCheck = iterToCheck, stabilityThreshold = stabilityThreshold,
             stabilityChecksBeforeFail = stabilityChecksBeforeFail, useGPU = useGPU)
         end

         networks = fetch.(networks)
         return networks
     end
end
