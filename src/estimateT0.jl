function calcDegradationRates(labeledData::Matrix{Float32}, totalData::Matrix{Float32},
     labelingTime::AbstractVector)
    #get slope for each gene's new to total RNA for each time group
    labelingTimeCat = CategoricalArray(labelingTime)
    slopes = Array{Float32}(undef, size(labeledData)[1], length(levels(labelingTimeCat)))
    for i=1:length(levels(labelingTimeCat))
        cellsToUse = findall(x->x == levels(labelingTimeCat)[i], labelingTimeCat)
        for j=1:size(labeledData)[1]
            lr = linregress(totalData[j,cellsToUse], labeledData[j,cellsToUse])
            slopes[j,i] = LinearRegression.slope(lr)[1]
        end
    end

    #if some slopes are very slightly negative, set those to zero
    slopes[findall(x->x < 0, slopes)] .= 0

    degradation = -1 .* log.(1 .- slopes)

    return degradation

end

""""
`estimateT0LabelingData(labeledData::Matrix{Float32}, totalData::Matrix{Float32},
    unlabeledData::Matrix{Float32}, labelingTime::AbstractVector)`

Function to predict total expression level before labeling based on
degradation rate estimates.
Outputs the estimated time 1 counts matrix.

# Required Arguments
* labeledData - Float32 counts matrix of the labeled counts
* totalData - Float32 counts matrix of combined counts
* unlabeledData - Float32 counts matrix of unlabeled counts
* labelingTime - Vector with the amount of time each cell was labeled for
"""
function estimateT0LabelingData(labeledData::Matrix{Float32}, totalData::Matrix{Float32},
    unlabeledData::Matrix{Float32}, labelingTime::AbstractVector)

    degradationRateEstimates = calcDegradationRates(labeledData, totalData, labelingTime)

    labelingTimeCat = CategoricalArray(labelingTime)
    T0 = copy(unlabeledData)
    for i=1:length(levels(labelingTimeCat))
        cellsToUse = findall(x->x == levels(labelingTimeCat)[i], labelingTimeCat)
        T0[:,cellsToUse] = T0[:,cellsToUse] .+ degradationRateEstimates[:,i]
    end

    return T0
end
