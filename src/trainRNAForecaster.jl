function minZero(num::AbstractArray{<:Number})
    num[findall(x->x < 0, num)] .= 0
    num
end

#function to check model prediction stability for each gene
function checkModelStability(model, initialConditions::Matrix{Float32}, iterToCheck::Int, expressionThreshold::Float32)

    println("Checking Model Stability...")

    inputData = copy(initialConditions)
    recordedVals = Array{Float32}(undef, size(inputData)[1], size(inputData)[2], iterToCheck)

    for i=1:iterToCheck
        recordedVals[:,:,i] = inputData

        #calculate model predictions
        exprPreds = Matrix{Float32}(undef, size(inputData)[1], size(inputData)[2])
        #suppressing warning messages
        @suppress begin
            for x=1:size(exprPreds)[2]
                pred = model(inputData[:,x])
                exprPreds[:,x] = minZero(pred[1])
            end
        end

        inputData = copy(exprPreds)
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
        ŷ = model(x)
        ls += mse(ŷ, y)
        num +=  size(x)[end]
    end
    return ls / num
end


#train the neural ODE
function trainNetwork(trainData, nGenes::Int,
     hiddenLayerNodes::Int, learningRate::Float64, nEpochs::Int)

    nn = Chain(Dense(nGenes, hiddenLayerNodes, relu),
               Dense(hiddenLayerNodes, nGenes)) |> gpu
    model = NeuralODE(nn, (0.0f0, 1.0f0), Tsit5(),
                       save_everystep = false,
                       reltol = 1e-3, abstol = 1e-3,
                       save_start = false) |> gpu

    opt = ADAM(learningRate)

    ps = Flux.params(model)
    losses = Vector{Float32}(undef, nEpochs)
    @suppress begin
       for epoch in 1:nEpochs
           for (x, y) in trainData
               gs = gradient(() -> mse(model(x), y), ps) # compute gradient
               Flux.Optimise.update!(opt, ps, gs) # update parameters
           end

           # Report on training error
           losses[epoch]= meanLoss(trainData, model)
       end
    end
    return (model, losses)
end

#train the neural ODE with validation set
function trainNetworkVal(trainData, valData,
     nGenes::Int, hiddenLayerNodes::Int, learningRate::Float64, nEpochs::Int)

    nn = Chain(Dense(nGenes, hiddenLayerNodes, relu),
               Dense(hiddenLayerNodes, nGenes)) |> gpu
    model = NeuralODE(nn, (0.0f0, 1.0f0), Tsit5(),
                       save_everystep = false,
                       reltol = 1e-3, abstol = 1e-3,
                       save_start = false)|> gpu

   opt = ADAM(learningRate)

   ps = Flux.params(model)
   losses = Matrix{Float32}(undef, nEpochs, 2)
   @suppress begin
       for epoch in 1:nEpochs
           for (x, y) in trainData
               gs = gradient(() -> mse(model(x), y), ps) # compute gradient
               Flux.Optimise.update!(opt, ps, gs) # update parameters
           end

           # Report on train and validation error
           losses[epoch,1]= meanLoss(trainData, model)
           losses[epoch,2] = meanLoss(valData, model)
       end
   end

   return (model, losses)
end

"""
`trainRNAForecaster(expressionDataT1::Matrix{Float32}, expressionDataT2::Matrix{Float32};
     trainingProp::Float64 = 0.8, hiddenLayerNodes::Int = 2*size(expressionDataT1)[1],
     shuffleData::Bool = true, seed::Int = 123, learningRate::Float64 = 0.005,
     nEpochs::Int = 10, batchsize::Int = 10, checkStability::Bool = true, iterToCheck::Int = 50,
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
     nEpochs::Int = 10, batchsize::Int = 10, checkStability::Bool = true, iterToCheck::Int = 50,
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
             trainData = DataLoader(gpu.((trainX, trainY)), batchsize=batchsize)
             valData = DataLoader(gpu.((valX, valY)), batchsize=batchsize)
         else
             trainData = DataLoader((trainX, trainY), batchsize=batchsize)
             valData = DataLoader((valX, valY), batchsize=batchsize)
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
                      nGenes, hiddenLayerNodes, learningRate, nEpochs)

                #check model stability
                checkStability = checkModelStability(trainedNet[1], trainX, iterToCheck, stabilityThreshold)
                Random.seed!(seed+iter)
                iter +=1
            end
        else
            #train the neural ODE
            trainedNet = trainNetworkVal(trainData, valData,
                 nGenes, hiddenLayerNodes, learningRate, nEpochs)
       end

       return trainedNet

    #in the case where we want to use the entire data set to train the model
    #(e.g. if we have already validated its performance and we now want to
    #train using the full data set)
     else
         trainX = expressionDataT1
         trainY = expressionDataT2

         if useGPU
             trainData = DataLoader(gpu.((trainX, trainY)), batchsize=batchsize)
         else
             trainData = DataLoader((trainX, trainY), batchsize=batchsize)
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
                  learningRate, nEpochs)

                #check model stability
                checkStability = checkModelStability(trainNet[1], trainX,
                 iterToCheck, stabilityThreshold)
                Random.seed!(seed+iter)
                iter +=1
            end
        else
            #train the neural ODE
            trainedNet = trainNetwork(trainData, nGenes, hiddenLayerNodes,
             learningRate, nEpochs)
       end

       return trainedNet
     end
 end
