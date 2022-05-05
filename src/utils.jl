#utility functions for RNAForecaster

#filtering expression count matrices
"""
`filterByZeroProp(t1Counts::Matrix{Float32}, t2Counts::Matrix{Float32},
    zeroProp::Float32)`

Filter by zero proportion for both genes and cells. Very high sparsity prevents
the neural network from achieving a stable solution.

# Required Arguments
* t1Counts - Counts matrix for time 1. Should be genes x cells and Float32.
* t2Counts - Counts matrix for time 2. Should be genes x cells and Float32.

# Keyword Arguments
* zeroProp - proportion of zeroes allowed for a gene or a cell. 0.98f0 by default
"""
function filterByZeroProp(t1Counts::Matrix{Float32}, t2Counts::Matrix{Float32};
    zeroProp::Float32 = 0.98f0)

    zeroPropT1Genes = Array{Float32}(undef, size(t1Counts)[1])
    for i=1:size(t1Counts)[1]
        zeroPropT1Genes[i] = length(findall(x->x == 0, t1Counts[i,:]))/size(t1Counts)[2]
    end
    zeroPropT2Genes = Array{Float32}(undef, size(t2Counts)[1])
    for i=1:size(t2Counts)[1]
        zeroPropT2Genes[i] = length(findall(x->x == 0, t2Counts[i,:]))/size(t2Counts)[2]
    end

    t1Sub = t1Counts[intersect(findall(x->x < zeroProp, zeroPropT1Genes),
     findall(x->x < zeroProp, zeroPropT2Genes)),:]
    t2Sub = t2Counts[intersect(findall(x->x < zeroProp, zeroPropT1Genes),
     findall(x->x < zeroProp, zeroPropT2Genes)),:]

     zeroPropT1Cells = Array{Float32}(undef, size(t1Counts)[2])
     for i=1:size(t1Counts)[2]
         zeroPropT1Cells[i] = length(findall(x->x == 0, t1Counts[:,i]))/size(t1Counts)[1]
     end
     zeroPropT2Cells = Array{Float32}(undef, size(t2Counts)[2])
     for i=1:size(t2Counts)[2]
         zeroPropT2Cells[i] = length(findall(x->x == 0, t2Counts[:,i]))/size(t2Counts)[1]
     end

     t1Sub = t1Sub[:,intersect(findall(x->x < zeroProp, zeroPropT1Cells),
      findall(x->x < zeroProp, zeroPropT2Cells))]
     t2Sub = t2Sub[:,intersect(findall(x->x < zeroProp, zeroPropT1Cells),
      findall(x->x < zeroProp, zeroPropT2Cells))]

     return (t1Sub, t2Sub)
end


"""
`filterByGeneVar(t1Counts::Matrix{Float32}, t2Counts::Matrix{Float32},
    topGenes::Int)`

Filter by gene variance, measured across both count matrices.

# Required Arguments
* t1Counts - Counts matrix for time 1. Should be genes x cells and Float32.
* t2Counts - Counts matrix for time 2. Should be genes x cells and Float32.
* topGenes - the number of top most variable genes to use

# Examples
`mat1 = Float32.(randn(50,50))
mat2 = Float32.(randn(50,50))
hvgFilteredCounts = filterByGeneVar(mat1, mat2, 20)`
"""
function filterByGeneVar(t1Counts::Matrix{Float32}, t2Counts::Matrix{Float32},
    topGenes::Int)

    geneVars = vec(var(t1Counts + t2Counts, dims = 2))
    ordered = sortperm(geneVars)

    t1Counts = t1Counts[ordered[1:topGenes],:]
    t2Counts = t2Counts[ordered[1:topGenes],:]

    return (t1Counts, t2Counts)
end

"""
`saveForecaster(trainedModel, fileName::String)`

Saves the parameters from the neural network after training.

# Required Arguments
* trainedModel - the trained neural network from trainRNAForecaster - just the network
don't include the loss results
* fileName - fileName to save the parameters to. Must end in .jld2
"""
function saveForecaster(trainedModel, fileName::String)
    save_object(fileName, Flux.params(trainedModel))
end

"""
`loadForecaster(fileName::String, inputNodes::Int, hiddenLayerNodes::Int)`

Recreates a previously saved neural network.

# Required Arguments
* fileName - file name where the parameters are saved
* inputNodes - number of input nodes in the network. Should be the same as the number
of genes in the data the network was trained on
* hiddenLayerNodes - number of hidden layer nodes in the network
"""
function loadForecaster(fileName::String, inputNodes::Int, hiddenLayerNodes::Int)
    #recreate neural network structure
    nn = Chain(Dense(inputNodes, hiddenLayerNodes, relu),
               Dense(hiddenLayerNodes, inputNodes))
    model = NeuralODE(nn, (0.0f0, 1.0f0), Tsit5(),
                       save_everystep = false,
                       reltol = 1e-3, abstol = 1e-3,
                       save_start = false)
    #load parameters into the model
    model = loadmodel!(model, load_object(fileName))
    return model
end



function saveEnsembleForecaster(trainedForecaster, fileName::String; gpu::Bool = false)

    if gpu
        trainedNetworksCpu = Vector{Any}(undef, length(trainedForecaster))
        for i=1:length(trainedForecaster)
            trainedNetworksCpu[i] = cpu.(trainedForecaster[i])
        end
    end

    paramVector = Vector{Any}(undef, length(trainedForecaster))
    for i=1:length(trainedNetworksCpu)
        paramVector[i] = Flux.params(trainedNetworksCpu[i][1])
    end

    save_object(fileName * ".jld2", paramVector)

    #losses
    lossVector = Vector{Any}(undef, length(trainedForecaster))
    for i=1:length(trainedNetworksCpu)
        lossVector[i] = trainedNetworksCpu[i][2]
    end

    save_object(fileName * "_Losses.jld2", lossVector)
end



function loadEnsembleForecaster(fileName::String, inputNodes::Int, hiddenLayerNodes::Int)

    paramVec = load_object(fileName * ".jld2")
    lossVec = load_object(fileName * "_Losses.jld2")

    rnaForecaster = Vector{Any}(undef, length(paramVec))
    for i=1:length(paramVec)
        #recreate neural network structure
        nn = Chain(Dense(inputNodes, hiddenLayerNodes, relu),
                   Dense(hiddenLayerNodes, inputNodes))
        model = NeuralODE(nn, (0.0f0, 1.0f0), Tsit5(),
                           save_everystep = false,
                           reltol = 1e-3, abstol = 1e-3,
                           save_start = false)

        model = loadmodel!(model, paramVec[i])

        rnaForecaster[i] = (model, lossVec[i])
    end

    return rnaForecaster
end
