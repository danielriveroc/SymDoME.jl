using SymDoME
using Test
using Statistics


@testset "SymDoME.jl" begin

    # Test with a regression problem
    inputs = Float64.(hcat(1:100, 200:-2:1));
    targets = inputs[:,1].*2.4 .- inputs[:,2]./inputs[:,1] .- 10;
    validationIndices = 2:3:100;
    testIndices = 3:3:100;
    (trainingResult, validationResult, testResult, bestTree) = dome(inputs, targets;
        validationIndices = validationIndices ,
        testIndices = testIndices ,
        minimumReductionMSE = 1e-6 ,
        maximumHeight = Inf ,
        maximumNodes = 13 ,
        strategy = StrategySelectiveWithConstantOptimization ,
        useDivisionOperator = true ,
        showText = false
    );

    func = eval(Meta.parse(string("X -> ", vectorString(bestTree))));
    outputs = Base.invokelatest(func, inputs);
    trainingIndices = setdiff(1:length(targets),vcat(validationIndices,testIndices));
    trainResult  = mean((outputs[  trainingIndices] .- targets[  trainingIndices]).^2);
    valResult    = mean((outputs[validationIndices] .- targets[validationIndices]).^2);
    testResult2  = mean((outputs[      testIndices] .- targets[      testIndices]).^2);
    @assert isapprox(trainResult,   trainingResult; atol=1e-7)
    @assert isapprox(valResult  , validationResult; atol=1e-7)
    @assert isapprox(testResult2,       testResult; atol=1e-7)
    @assert isapprox(trainResult, mean( (evaluateTree(bestTree, view(inputs,   trainingIndices, :) ) .- targets[  trainingIndices]).^2 ) )
    @assert isapprox(  valResult, mean( (evaluateTree(bestTree, view(inputs, validationIndices, :) ) .- targets[validationIndices]).^2 ) )
    @assert isapprox( testResult, mean( (evaluateTree(bestTree, view(inputs,       testIndices, :) ) .- targets[      testIndices]).^2 ) )
    @assert isapprox(trainResult, 6.172e-29; atol=1e-7)
    @assert isapprox(valResult  , 5.732e-29; atol=1e-7)
    @assert isapprox(testResult2, 4.312e-29; atol=1e-7)

    # Test with a classification problem
    inputs = Float64.(hcat(1:100, 200:-2:1));
    targets = inputs[:,1].*2.4 .- inputs[:,2]./inputs[:,1] .- 10;
    validationIndices = 2:3:100;
    testIndices = 3:3:100;
    targets = targets.>=0;
    (trainingResult, validationResult, testResult, bestTree) = dome(inputs, targets;
        validationIndices = validationIndices ,
        testIndices = testIndices ,
        minimumReductionMSE = 1e-6 ,
        maximumHeight = Inf ,
        maximumNodes = 13 ,
        strategy = StrategySelectiveWithConstantOptimization ,
        useDivisionOperator = true ,
        showText = false
    );

    func = eval(Meta.parse(string("X -> ", vectorString(bestTree))));
    outputs = Base.invokelatest(func, inputs);
    trainingIndices = setdiff(1:length(targets),vcat(validationIndices,testIndices));
    trainResult  = mean((outputs[  trainingIndices].>=0) .== targets[  trainingIndices]);
    valResult    = mean((outputs[validationIndices].>=0) .== targets[validationIndices]);
    testResult2  = mean((outputs[      testIndices].>=0) .== targets[      testIndices]);
    @assert isapprox(trainResult,   trainingResult; atol=1e-7)
    @assert isapprox(valResult  , validationResult; atol=1e-7)
    @assert isapprox(testResult2,       testResult; atol=1e-7)
    @assert isapprox(trainResult, mean( (evaluateTree(bestTree, view(inputs,   trainingIndices, :) ).>=0) .== targets[  trainingIndices] ) )
    @assert isapprox(  valResult, mean( (evaluateTree(bestTree, view(inputs, validationIndices, :) ).>=0) .== targets[validationIndices] ) )
    @assert isapprox( testResult, mean( (evaluateTree(bestTree, view(inputs,       testIndices, :) ).>=0) .== targets[      testIndices] ) )
    @assert isapprox(trainResult, 0.970588235294117; atol=1e-7)
    @assert isapprox(valResult  , 1                ; atol=1e-7)
    @assert isapprox(testResult , 0.969696969696969; atol=1e-7)

end
