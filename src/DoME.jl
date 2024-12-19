
using Statistics

include("NodePool.jl")

@inline MSE(y::Semantic, t::AbstractArray{<:Real,1}) = mean((y.-t).^2);
@inline Accuracy(y::Semantic, t::AbstractArray{<:Real,1}) = eltype(t)==Bool ?  mean((y .>= 0) .== t) : mean((y.>=0) .== (t.>=0));



#################################################################################################################################################################
#
# Struct used to store the results of the searches
#  Since only the best result of any search is going to be used, this struct could be seen as unnecessary
#  However, because of inaccuracies, the best result might not improve the mse. In this case, we have to check the following one
#  Therefore, all of the search results are stored
#

mutable struct SearchResults
    MSEReduction::Array{<:Real,1}
    numNode::Array{Int,1}
    constant::Array{<:Real,1}
    numVariableNodePool::Array{Int,1}
    operation::Array{Int,1}
    nodePoolVariables::NodePool
    index::Int
    SearchResults(constantType::DataType, nodePoolVariables::NodePool) = new(
        Array{constantType,1}(undef, 0),
        Array{Int,1}(undef, 0),
        Array{constantType,1}(undef, 0),
        Array{Int,1}(undef, 0),
        Array{Int,1}(undef, 0),
        nodePoolVariables, 0
    );
end;

@inbounds function increaseSize!(sr::SearchResults)
    currentLength = length(sr.MSEReduction);
    newNumResults = currentLength + 200;
    constantType = eltype(sr.constant);
    newMSEReduction        = Array{constantType,1}(undef, newNumResults); newMSEReduction[1:sr.index] .= sr.MSEReduction[1:sr.index];               sr.MSEReduction = newMSEReduction
    newNumNode             = Array{Int,1}(undef, newNumResults);               newNumNode[1:sr.index] .= sr.numNode[1:sr.index];                         sr.numNode = newNumNode;
    newNumVariableNodePool = Array{Int,1}(undef, newNumResults);   newNumVariableNodePool[1:sr.index] .= sr.numVariableNodePool[1:sr.index]; sr.numVariableNodePool = newNumVariableNodePool;
    newConstant            = Array{constantType,1}(undef, newNumResults);     newConstant[1:sr.index] .= sr.constant[1:sr.index];                       sr.constant = newConstant;
    newOperation           = Array{Int,1}(undef, newNumResults);             newOperation[1:sr.index] .= sr.operation[1:sr.index];                     sr.operation = newOperation;
end;

resetSearchResults!(sr::SearchResults) = ( sr.index = 0; )


@inbounds function addVariableSearchResult!(sr::SearchResults, MSEReduction::Real, numNode::Int, numVariableNodePool::Int)
    (MSEReduction<0 || isnan(MSEReduction)) && return;
    (sr.index>=length(sr.MSEReduction)) && increaseSize!(sr);
    sr.index += 1;
    sr.MSEReduction[sr.index] = MSEReduction;
    sr.numNode[sr.index] = numNode;
    sr.constant[sr.index] = NaN;
    sr.operation[sr.index] = -1;
    sr.numVariableNodePool[sr.index] = numVariableNodePool;
end;

@inbounds function addConstantSearchResult!(sr::SearchResults, MSEReduction::Real, numNode::Int, constant::Real)
    (MSEReduction<0 || isnan(MSEReduction)) && return;
    (sr.index>=length(sr.MSEReduction)) && increaseSize!(sr);
    sr.index += 1;
    sr.MSEReduction[sr.index] = MSEReduction;
    sr.numNode[sr.index] = numNode;
    sr.constant[sr.index] = constant;
    sr.operation[sr.index] = -1;
    sr.numVariableNodePool[sr.index] = -1;
end;

@inbounds function addConstantVariableSearchResult!(sr::SearchResults, MSEReduction::Real, numNode::Int, constant::Real, operation::Int, numVariableNodePool::Int)
    (MSEReduction<0 || isnan(MSEReduction)) && return;
    (sr.index>=length(sr.MSEReduction)) && increaseSize!(sr);
    sr.index += 1;
    sr.MSEReduction[sr.index] = MSEReduction;
    sr.numNode[sr.index] = numNode;
    sr.constant[sr.index] = constant;
    sr.operation[sr.index] = operation;
    sr.numVariableNodePool[sr.index] = numVariableNodePool;
end;

@inline addConstantVariableSearchResult_Add!(sr::SearchResults, MSEReduction::Real, numNode::Int, constant::Real, numVariableNodePool::Int) = addConstantVariableSearchResult!(sr, MSEReduction, numNode, constant, 1, numVariableNodePool)
@inline addConstantVariableSearchResult_Sub!(sr::SearchResults, MSEReduction::Real, numNode::Int, constant::Real, numVariableNodePool::Int) = addConstantVariableSearchResult!(sr, MSEReduction, numNode, constant, 2, numVariableNodePool)
@inline addConstantVariableSearchResult_Mul!(sr::SearchResults, MSEReduction::Real, numNode::Int, constant::Real, numVariableNodePool::Int) = addConstantVariableSearchResult!(sr, MSEReduction, numNode, constant, 3, numVariableNodePool)
@inline addConstantVariableSearchResult_Div!(sr::SearchResults, MSEReduction::Real, numNode::Int, constant::Real, numVariableNodePool::Int) = addConstantVariableSearchResult!(sr, MSEReduction, numNode, constant, 4, numVariableNodePool)

@inbounds function addConstantExpressionSearchResult!(sr::SearchResults, MSEReduction::Real, numNode::Int, constant::Real, operation::Int)
    (MSEReduction<0 || isnan(MSEReduction)) && return;
    (sr.index>=length(sr.MSEReduction)) && increaseSize!(sr);
    sr.index += 1;
    sr.MSEReduction[sr.index] = MSEReduction;
    sr.numNode[sr.index] = numNode;
    sr.constant[sr.index] = constant;
    sr.operation[sr.index] = operation;
    sr.numVariableNodePool[sr.index] = -1;
end;

@inline addConstantExpressionSearchResult_Add!(sr::SearchResults, MSEReduction::Real, numNode::Int, constant::Real) = addConstantExpressionSearchResult!(sr, MSEReduction, numNode, constant, 1)
@inline addConstantExpressionSearchResult_Mul!(sr::SearchResults, MSEReduction::Real, numNode::Int, constant::Real) = addConstantExpressionSearchResult!(sr, MSEReduction, numNode, constant, 3)


@inbounds function bestSearchResult!(sr::SearchResults, nodeList::Array{Tree,1})

    operationFunction(numOperation::Int) = numOperation==1 ? AddNode : numOperation==2 ? SubNode : numOperation==3 ? MulNode : numOperation==4 ? DivNode : nothing;

    sr.index<=0 && return nothing;
    (bestMSEReduction, index) = findmax(view(sr.MSEReduction, 1:sr.index));
    bestMSEReduction<0 && return nothing;
    numNode = sr.numNode[index];
    constant = sr.constant[index];
    operation = sr.operation[index];
    numVariableNodePool = sr.numVariableNodePool[index];
    if isnan(constant)
        # Variable search
        subTreeInsert = clone(sr.nodePoolVariables.nodes[numVariableNodePool]);
    elseif numVariableNodePool==-1
        if operation==-1
            # Constant search
            subTreeInsert = Constant(constant);
        else
            # Constant-expression search
            subTreeInsert = operationFunction(operation)(Constant(constant), nodeList[numNode]);
        end;
    else
        # Constant-variable search
        subTreeInsert = operationFunction(operation)(Constant(constant), clone(sr.nodePoolVariables.nodes[numVariableNodePool]));
    end;
    sr.MSEReduction[index] = -Inf; # Not to choose this combination again
    return (bestMSEReduction, numNode, subTreeInsert)
end;



#################################################################################################################################################################
#
# The DoME struct
#

mutable struct DoME
    tree::Tree
    mse::Real
    goalMSE::Real
    nodePoolVariables::NodePool
    inputs::Array{<:Real,2}
    dataInRows::Bool
    targets::Array{<:Real,1}
    initialEquation::NodeEquation
    minimumReductionMSE::Real
    maximumHeight::Number
    maximumNodes::Number
    strategy::Function
    useDivisionOperator::Bool
    searchResults::SearchResults
    checkForErrors::Bool
    function DoME(inputs::AbstractArray{<:Real,2}, targets::AbstractArray{<:Real,1};
        # Each instance in inputs is in a row or in a column
        dataInRows          ::Bool                     = true,
        # Hyperparameters of the algorithm
        minimumReductionMSE ::Real                     = 1e-6,
        maximumNodes        ::Int64                    = 50 ,
        strategy            ::Function                 = StrategySelectiveWithConstantOptimization ,
        # Other hyperparameter that the user might find useful
        maximumHeight       ::Real                     = Inf ,
        # Another stopping criteria
        goalMSE             ::Real                     = 0 ,
        # Whether to use the division operator or not
        useDivisionOperator ::Bool                     = true ,
        # Initial tree with its MSE
        initialTree::Union{Nothing,Tree,Tuple{Tree,Real}} = nothing ,
        # This parameter was used only for development. If it is set to true, the execution becomes much slower
        checkForErrors      ::Bool                     = false
    )

        @assert(!any(isnan.(inputs)));
        @assert(!any(isinf.(inputs)));
        @assert(eltype(inputs)==Float32 || eltype(inputs)==Float64);
        @assert(length(targets)==size(inputs, dataInRows ? 1 : 2), "Inputs and targets have different number of instances");

        uniqueTargets = sort(unique(targets));
        # @assert(length(uniqueTargets)>1);
        classificationProblem = eltype(targets)==Bool;
        if classificationProblem
            # if (eltype(targets)==Bool) || uniqueTargets==[0, 1]
                targets = (eltype(inputs).(targets) .* 2) .- 1;
            # end;
            # @assert(length(targets)>=2)
        else
            @assert(eltype(inputs)==eltype(targets));
        end;

        initialEquation = NodeEquation(eltype(inputs)(1.), targets, eltype(inputs)(0.), eltype(inputs)(-1.));

        @assert(!anyNaN(targets));
        @assert(!anyInf(targets));

        nodePoolVariables = NodePoolVariables(inputs; dataInRows=dataInRows);
        # if (length(nodePoolVariables.nodes)==0) && length(uniqueTargets)>1
        #     error("No variables have been added, they all have the same values for all of the instances, with different targets values");
        # end;

        # obj = new(tree, mse, eltype(inputs)(0), nodePoolVariables, inputs, targets, initialEquation, eltype(inputs)(1e-6), maximumHeight, maximumNodes, strategy, true, SearchResults(eltype(inputs), nodePoolVariables), checkForErrors);
        obj = new(Constant(0.), -1, eltype(inputs)(goalMSE), nodePoolVariables, inputs, dataInRows, targets, initialEquation, eltype(inputs)(minimumReductionMSE), maximumHeight, maximumNodes, strategy, useDivisionOperator, SearchResults(eltype(inputs), nodePoolVariables), checkForErrors);
        if isnothing(initialTree)
            # Build initial tree
            updateTree!(obj, Constant(mean(targets)); overwriteOnlyIfBetter=false);
        elseif isa(initialTree, Tree)
            updateTree!(obj, initialTree; overwriteOnlyIfBetter=false);
        elseif isa(initialTree, Tuple)
            updateTree!(obj, initialTree...; overwriteOnlyIfBetter=false);
        end;

        return obj;
    end;
end;


function updateTree!(obj::DoME, tree::Tree; mse::Real=NaN, overwriteOnlyIfBetter::Bool=false)
    tree = clone(tree; resetSemantic=true);
    if !setVariableValues!(tree, obj.nodePoolVariables)
        error("The tree has a number of variables higher than the dataset");
    end;
    if isnan(mse)
        mse = MSE(evaluateTree(tree), obj.targets);
    end;
    @assert(!isnan(mse) && !isinf(mse));

    if (!overwriteOnlyIfBetter) || (mse<obj.mse)
        obj.tree = tree;
        obj.mse = mse;
    end;
end;



function updateDataset!(obj::DoME, inputs::AbstractArray{<:Real,2}, targets::AbstractArray{<:Real,1}; dataInRows::Bool=true)
    maxVariableIndex(node::Constant) = 0;
    maxVariableIndex(node::Variable) = node.variableNumber;
    maxVariableIndex(node::BinaryNode) = max(maxVariableIndex(node.child1), maxVariableIndex(node.child2));
    maxVariableIndex(node::NonBinaryNode) = maximum([maxVariableIndex(node.child) for child in node.children]);
    if size(inputs, dataInRows ? 2 : 1) < maxVariableIndex(tree)
        error("The tree has a number of attributes higher than the dataset");
    end;
    newNodePoolVariables = NodePoolVariables(inputs; dataInRows=dataInRows);
    setVariableValues!(obj.tree, newNodePoolVariables)
    obj.inputs = inputs;
    obj.targets = targets;
    obj.tree = newTree;
    obj.nodePoolVariables = newNodePoolVariables;
    obj.mse = MSE(evaluateTree(tree), obj.targets);
end;







#################################################################################################################################################################
#
# The four searches and the function to optimize constants
#


function variableSearch!(sr::SearchResults, nodes::Array{Tree,1}, nodePoolVariables::NodePool, mse::Real, targets::Array{<:Real,1}, whichNodesPerformVariableSearch::Union{DataType,Union}; checkForErrors=false)

    if (checkForErrors)
        @assert(unique(nodePoolVariables.heights)==[1]);
        @assert(unique(nodePoolVariables.numNodes)==[1]);
        @assert(unique(isa.(nodePoolVariables.nodes,Variable))==[true]);
    end;

    for numNode in 1:length(nodes)

        node = nodes[numNode];

        # if isa(node, whichNodesPerformVariableSearch) && !isNaN(node.equation.S)
        (!isa(node, whichNodesPerformVariableSearch) || isNaN(node.equation.S)) && continue;

        for indexNodePool in 1:length(nodePoolVariables.nodes)

            # If looking into the variables of the tree, do not look into the same variables
            isa(node,Variable) && (node.variableNumber==nodePoolVariables.nodes[indexNodePool].variableNumber) && continue;

            semantic = nodePoolVariables.semantics[indexNodePool];

            isa(semantic, Real) && continue;

            # If the semantic of the variable is not in the domain, do not check it
            semanticNotInDomain(semantic, node.equation) && continue;

            newMSE = MSE(semantic, targets);
            reduction = mse - newMSE;
            addVariableSearchResult!(sr, reduction, numNode, indexNodePool);

        end;

    end;

end;


function constantSearch!(sr::SearchResults, nodes::Array{Tree,1}, mse::Real, numNodes::Array{UInt,1}, targets::Array{<:Real,1}, whichNodesPerformConstantSearch::Union{DataType,Union}; checkForErrors=false)

    for numNode in 2:length(nodes)
        node = nodes[numNode];
        if isa(node, whichNodesPerformConstantSearch) && !isNaN(node.equation.S)
            (constant, reduction) = calculateConstantMinimizeEquation(node.equation, mse; checkForErrors=checkForErrors);
            addConstantSearchResult!(sr, reduction, numNode, constant);
        end;
    end;
end;


function constantVariableSearch!(sr::SearchResults, nodes::Array{Tree,1}, nodePool::NodePool, mse::Real, targets::Array{<:Real,1}, useDivisionOperator::Bool, maximumHeights, maximumNodes, whichNodesPerformConstantVariableSearch::Union{DataType,Union}; checkForErrors=false)

    for numNode in 1:length(nodes)

        # (maximumNodes[numNode]<3 || maximumHeights[numNode]<2) && continue;

        node = nodes[numNode];

        # if isa(node, whichNodesPerformConstantVariableSearch) && !isNaN(node.equation.S)
        (!isa(node, whichNodesPerformConstantVariableSearch) || isNaN(node.equation.S)) && continue;

        # (a,b,c,d) = node.equation;
        eq=node.equation; a=eq.a; b=eq.b; c=eq.c; d=eq.d; S=eq.S;

        for indexNodePool in findall(((nodePool.numNodes.+2).<=maximumNodes[numNode]) .& (nodePool.heights.<=(maximumHeights[numNode]-1)))
        # for indexNodePool in Base.OneTo(length(nodePool.nodes))

            checkForErrors && (@assert(nodePool.numNodes[indexNodePool]==1 && nodePool.heights[indexNodePool]==1))

            nodeVariable = nodePool.nodes[indexNodePool];
            checkForErrors && @assert(nodePool.nodes[indexNodePool].variableNumber==indexNodePool)

            semantic = nodeVariable.semantic;

            isa(semantic, Real) && continue;

            # if ( (!isa(node,BinaryNode)) || (node.name!="+") || (!isa(node.child1,Constant)) || (!isa(node.child2,Variable)) || (node.child2.variableNumber!=numVar) )
            if ( (!isa(node,BinaryNode)) || (node.name!="+") || (!isa(node.child1,Constant)) || (!isa(node.child2,Variable)) || (node.child2.variableNumber!=nodeVariable.variableNumber) )
                equationConstant = equationAddChild1(semantic, a, b, c, d, S);
                if (checkForErrors) checkEquation(equationConstant); end;
                # If there are possible values in the domain
                if (!isNaN(equationConstant.S))
                    (constant, reductionMSE) = calculateConstantMinimizeEquation(equationConstant, mse; checkForErrors=checkForErrors);
                    addConstantVariableSearchResult_Add!(sr, reductionMSE, numNode, constant, indexNodePool);
                end;
            end;

            # if ( (!isa(node,BinaryNode)) || (node.name!="-") || (!isa(node.child1,Constant)) || (!isa(node.child2,Variable)) || (node.child2.variableNumber!=numVar) )
            if ( (!isa(node,BinaryNode)) || (node.name!="-") || (!isa(node.child1,Constant)) || (!isa(node.child2,Variable)) || (node.child2.variableNumber!=nodeVariable.variableNumber) )
                equationConstant = equationSubChild1(semantic, a, b, c, d, S);
                if (checkForErrors) checkEquation(equationConstant); end;
                # If there are possible values in the domain
                if (!isNaN(equationConstant.S))
                    (constant, reductionMSE) = calculateConstantMinimizeEquation(equationConstant, mse; checkForErrors=checkForErrors);
                    addConstantVariableSearchResult_Sub!(sr, reductionMSE, numNode, constant, indexNodePool);
                end;
            end;

            # if ( (!isa(node,BinaryNode)) || (node.name!="*") || (!isa(node.child1,Constant)) || (!isa(node.child2,Variable)) || (node.child2.variableNumber!=numVar) )
            if ( (!isa(node,BinaryNode)) || (node.name!="*") || (!isa(node.child1,Constant)) || (!isa(node.child2,Variable)) || (node.child2.variableNumber!=nodeVariable.variableNumber) )
                equationConstant = equationMulChild1(semantic, a, b, c, d, S);
                if (checkForErrors) checkEquation(equationConstant) end;
                # If there are possible values in the domain
                if (!isNaN(equationConstant.S))
                    (constant, reductionMSE) = calculateConstantMinimizeEquation(equationConstant, mse; checkForErrors=checkForErrors);
                    addConstantVariableSearchResult_Mul!(sr, reductionMSE, numNode, constant, indexNodePool);
                end;
            end;

            # if useDivisionOperator && ( (!isa(node,BinaryNode)) || (node.name!="/") || (!isa(node.child1,Constant)) || (!isa(node.child2,Variable)) || (node.child2.variableNumber!=numVar) )
            if useDivisionOperator && ( (!isa(node,BinaryNode)) || (node.name!="/") || (!isa(node.child1,Constant)) || (!isa(node.child2,Variable)) || (node.child2.variableNumber!=nodeVariable.variableNumber) )
                if !(nodePool.hasZerosInSemantics[indexNodePool])
                    equationConstant = equationDivChild1(semantic, a, b, c, d, S);
                    if (checkForErrors) checkEquation(equationConstant); end;
                    # If there are possible values in the domain
                    if (!isNaN(equationConstant.S))
                        (constant, reductionMSE) = calculateConstantMinimizeEquation(equationConstant, mse; checkForErrors=checkForErrors);
                        addConstantVariableSearchResult_Div!(sr, reductionMSE, numNode, constant, indexNodePool);
                    end;
                end;
            end;

        end;

    end;
end;



function constantExpressionSearch!(sr::SearchResults, nodes::Array{Tree,1}, mse::Real, targets::Array{<:Real,1}, heights, maximumHeights, numNodes, maximumNodes; checkForErrors=false)

    # Search only on non terminal nodes
    # for numNode in findall(isa.(nodes,NonTerminal) .& (heights.<maximumHeights) .& ((numNodes.+2).<=maximumNodes))
    for numNode in 1:length(nodes)

        (((numNodes[numNode]+2)>maximumNodes[numNode]) || (heights[numNode]>=maximumHeights[numNode])) && continue;

        node = nodes[numNode];

        (!isa(node,NonTerminal) || isNaN(node.equation.S)) && continue;

        semantic = node.semantic;
        eq=node.equation; a=eq.a; b=eq.b; c=eq.c; d=eq.d; S=eq.S;

        # if !isNaN(S)

        if ( ((node.name!="+") && (node.name!="-")) || ((!isa(node.child1,Constant)) || (!isa(node.child2,Constant))) )
            equationConstant = equationAddChild1(node.semantic, a, b, c, d, S);
            if (checkForErrors) checkEquation(equationConstant); end;
            # If there are possible values in the domain
            if (!isNaN(equationConstant.S))
                (constant, reductionMSE) = calculateConstantMinimizeEquation(equationConstant, mse; checkForErrors=checkForErrors);
                addConstantExpressionSearchResult_Add!(sr, reductionMSE, numNode, constant);
            end;
        end

        if ( ((node.name!="*") && (node.name!="/")) || ((!isa(node.child1,Constant)) || (!isa(node.child2,Constant))) )
            equationConstant = equationMulChild1(node.semantic, a, b, c, d, S);
            if (checkForErrors) checkEquation(equationConstant); end;
            # If there are possible values in the domain
            if (!isNaN(equationConstant.S))
                (constant, reductionMSE) = calculateConstantMinimizeEquation(equationConstant, mse; checkForErrors=checkForErrors);
                addConstantExpressionSearchResult_Mul!(sr, reductionMSE, numNode, constant);
            end;
        end;

        # end;
    end;

end;




function OptimizeConstants!(obj::DoME)

    nodes, = iterateTree(obj.tree);

    constantNodes = [isa(node,Constant) for node in nodes];
    nodes = nodes[constantNodes];
    isempty(nodes) && return nothing;
    pathsToConstantNodes = findNode.([obj.tree], nodes);

    if (length(nodes)==1)
        calculateEquations!(obj.tree, obj.initialEquation, path=pathsToConstantNodes[1]; checkForErrors=obj.checkForErrors);
        node = nodes[1];
        minimumReduction = obj.minimumReductionMSE * obj.mse;
        (constant, reduction) = calculateConstantMinimizeEquation(node.equation, obj.mse; checkForErrors=obj.checkForErrors);

        clearEquations!(obj.tree);
        if (reduction>minimumReduction) && (constant!=nodes[1].semantic)
            node.semantic = constant;
            mse = MSE(reevaluatePath(obj.tree, pathsToConstantNodes[1]; checkForErrors=obj.checkForErrors), obj.targets);
            if (obj.checkForErrors)
                @assert(isa(node, Constant))
                @assert(!isnan(mse));
                @assert(equal(mse+reduction, obj.mse));
            end;
            obj.mse = mse;
        end;
        return nothing;
    end;

    numNode = 0;
    numIterationsWithNoImprovement = 0;
    while (true)

        numNode = (numNode>=length(nodes)) ? 1 : numNode+1;

        calculateEquations!(obj.tree, obj.initialEquation; path=pathsToConstantNodes[numNode], checkForErrors=obj.checkForErrors)

        if (obj.checkForErrors)
            checkEquations(obj.tree, obj.mse, obj.targets);
        end;

        minimumReduction = obj.minimumReductionMSE * obj.mse;
        (constant, reduction) = calculateConstantMinimizeEquation(nodes[numNode].equation, obj.mse; checkForErrors=obj.checkForErrors);

        clearEquations!(obj.tree);
        if (reduction>minimumReduction) && (constant!=nodes[numNode].semantic)

            oldConstant = nodes[numNode].semantic;
            nodes[numNode].semantic = constant;

            mse = MSE(reevaluatePath(obj.tree, pathsToConstantNodes[numNode]; checkForErrors=obj.checkForErrors), obj.targets);

            # Check that the resulting MSE is lower than the previous MSE in obj.minimumReductionMSE
            # This is expected to be. However, precision errors may happen
            if (mse >= (obj.mse - (obj.minimumReductionMSE * obj.mse)))

                # Undo the change in the tree
                nodes[numNode].semantic = oldConstant;
                obj.mse = MSE(reevaluatePath(obj.tree, pathsToConstantNodes[numNode]; checkForErrors=obj.checkForErrors), obj.targets);
                # obj.checkForErrors && @assert(mse==obj.mse);
                numIterationsWithNoImprovement += 1;
                if (numIterationsWithNoImprovement>=length(nodes))
                    return;
                end;

            else

                if (obj.checkForErrors)
                    @assert(isa(nodes[numNode], Constant))
                    @assert(!isnan(mse));
                    # @assert(equal(mse+reduction, obj.mse));
                end;
                obj.mse = mse;
                numIterationsWithNoImprovement = 0;

            end;
        else
            numIterationsWithNoImprovement += 1;
            if (numIterationsWithNoImprovement>=length(nodes))
                return;
            end;
        end;
    end;

end;



#################################################################################################################################################################
#
# The function that performs the searches, called by a strategy
#

function PerformSearches!(obj::DoME;
    whichNodesPerformConstantSearch        ::Union{DataType,Union} = Nothing ,
    whichNodesPerformVariableSearch        ::Union{DataType,Union} = Nothing ,
    whichNodesPerformConstantVariableSearch::Union{DataType,Union} = Nothing ,
    performConstantExpressionSearch::Bool = false)

    (obj.mse<=obj.goalMSE) && return false;

    (nodes, heights, depths, _, numNodesEach) = iterateTree(obj.tree);

    # If the equations of the nodes were already calculated (the last search was unsuccessful), do not calculate them again
    if isnothing(obj.tree.equation)
        calculateEquations!(obj.tree, obj.initialEquation)
    end;

    if (obj.checkForErrors)
        checkEquations(obj.tree, obj.mse, obj.targets);
        # The equation must be correct: none of the semantic sets must be NaN (meaning that there are possible values in the domain)
        for node in nodes
            @assert(!isNaN(node.equation.S))
        end;
    end;

    subTreeInsert = nothing;
    indexNode = -1;
    maximumReduction = -Inf;
    maximumReduction = obj.minimumReductionMSE * obj.mse;

    resetSearchResults!(obj.searchResults);

    #########################################################################################
    # Constant search
    if (whichNodesPerformConstantSearch!=Nothing)
        # Check if a constant can be inserted anywhere in the tree
        constantSearch!(obj.searchResults, nodes, obj.mse, numNodesEach, obj.targets, whichNodesPerformConstantSearch; checkForErrors=obj.checkForErrors);
    end;

    #########################################################################################
    # Variable search
    if (whichNodesPerformVariableSearch!=Nothing)
        variableSearch!(obj.searchResults, nodes, obj.nodePoolVariables, obj.mse, obj.targets, whichNodesPerformVariableSearch; checkForErrors=obj.checkForErrors);
    end;

    if (whichNodesPerformConstantVariableSearch!=Nothing) || (performConstantExpressionSearch)
        maximumHeightSubTree = (obj.maximumHeight+1).-depths;
        maximumNodesSubTree = (obj.maximumNodes - numNodesEach[1]) .+ numNodesEach;
    end;

    #########################################################################################
    # Constant-Variable search
    if (whichNodesPerformConstantVariableSearch!=Nothing)
        constantVariableSearch!(obj.searchResults, nodes, obj.nodePoolVariables, obj.mse, obj.targets, obj.useDivisionOperator, maximumHeightSubTree, maximumNodesSubTree, whichNodesPerformConstantVariableSearch; checkForErrors=obj.checkForErrors);
    end;

    #########################################################################################
    # Constant-Expression search
    if (performConstantExpressionSearch)
        constantExpressionSearch!(obj.searchResults, nodes, obj.mse, obj.targets, heights, maximumHeightSubTree, numNodesEach, maximumNodesSubTree; checkForErrors=obj.checkForErrors);
    end;

    # Do not clear equations if the search was unsuccessful: they can be used for the next search

    while (true)

        bestResult = bestSearchResult!(obj.searchResults, nodes);
        isnothing(bestResult) && return false;
        (maximumReduction, indexNode, subTreeInsert) = bestResult;

        clearEquations!(obj.tree);

        if (indexNode==1)
            obj.tree = subTreeInsert;
        else
            evaluateTree(subTreeInsert; checkForErrors=obj.checkForErrors);
            replaceSubtree!(obj.tree, nodes[indexNode], subTreeInsert);
        end;

        mse = MSE(evaluateTree(obj.tree; checkForErrors=obj.checkForErrors), obj.targets);
        obj.checkForErrors && @assert(equal(obj.mse,mse+maximumReduction; tolerance=1e-4));

        # Check that the resulting MSE is lower than the previous MSE in obj.minimumReductionMSE
        # This is expected to be. However, precision errors may happen
        if (mse < (obj.mse - (obj.minimumReductionMSE * obj.mse)))
            obj.mse = mse;
            return true;
        end;

        # Undo the change in the tree
        if (indexNode==1)
            obj.tree = nodes[1];
        else
            replaceSubtree!(obj.tree, subTreeInsert, nodes[indexNode]);
        end;
        obj.checkForErrors && @assert(obj.mse == MSE(evaluateTree(obj.tree; checkForErrors=obj.checkForErrors), obj.targets));

    end;

end;








##################################################################################################
#
# Strategies
#    Return true if change has been done or false otherwise
#

function StrategyExhaustive(obj::DoME)
    changeDone = PerformSearches!(obj;
        whichNodesPerformConstantSearch=Any ,
        whichNodesPerformVariableSearch=Any ,
        whichNodesPerformConstantVariableSearch=Any ,
        performConstantExpressionSearch=true);
    return changeDone;
end;

function StrategyExhaustiveWithConstantOptimization(obj::DoME)
    changeDone = PerformSearches!(obj;
        whichNodesPerformConstantSearch=Union{Variable,NonTerminal} ,
        whichNodesPerformVariableSearch=Any ,
        whichNodesPerformConstantVariableSearch=Any ,
        performConstantExpressionSearch=true);
    changeDone && OptimizeConstants!(obj);
    return changeDone;
end;

function StrategySelectiveWithConstantOptimization(obj::DoME)
    # Variable search only on constants
    changeDone =               PerformSearches!(obj; whichNodesPerformVariableSearch=Constant);
    changeDone = changeDone || PerformSearches!(obj; performConstantExpressionSearch=true);
    # Constant-variable search only on terminals
    changeDone = changeDone || PerformSearches!(obj; whichNodesPerformConstantVariableSearch=Union{Constant,Variable});
    if (!changeDone)
        # Constant search on variables and non-terminals, variable seach on variables and non-terminals, and constant-variable search on non-terminals
        changeDone = PerformSearches!(obj;
            whichNodesPerformConstantSearch=Union{Variable,NonTerminal} ,
            whichNodesPerformVariableSearch=Union{Variable,NonTerminal} ,
            whichNodesPerformConstantVariableSearch=NonTerminal );
    end;
    changeDone && OptimizeConstants!(obj);
    return changeDone;
end;

function StrategySelective(obj::DoME)
    changeDone =               PerformSearches!(obj; whichNodesPerformConstantSearch=Constant);
    # Variable search only on constants
    changeDone = changeDone || PerformSearches!(obj; whichNodesPerformVariableSearch=Constant);
    changeDone = changeDone || PerformSearches!(obj; performConstantExpressionSearch=true);
    # Constant-variable search only on terminals
    changeDone = changeDone || PerformSearches!(obj; whichNodesPerformConstantVariableSearch=Union{Constant,Variable});
    if (!changeDone)
        # Constant search on variables and non-terminals, variable seach on variables and non-terminals, and constant-variable search on non-terminals
        changeDone = PerformSearches!(obj;
            whichNodesPerformConstantSearch=Union{Variable,NonTerminal} ,
            whichNodesPerformVariableSearch=Union{Variable,NonTerminal} ,
            whichNodesPerformConstantVariableSearch=NonTerminal );
    end;
    return changeDone;
end;

Strategy1 = StrategyExhaustive;
Strategy2 = StrategyExhaustiveWithConstantOptimization;
Strategy3 = StrategySelectiveWithConstantOptimization;
Strategy4 = StrategySelective;




#################################################################################################################################################################
#
# The function Step!, that performs one iteration of the algorithm
#

function Step!(obj::DoME)
    obj.mse<=obj.goalMSE && return false;
    obj.strategy(obj) && return true;

    # No change was done with a small tree, try to expand the tree with a higher order
    isempty(obj.nodePoolVariables.nodes) && return false;
    originalTree = obj.tree;
    mseCopy = obj.mse;
    numNodesCurrentTree = numNodes(originalTree);
    variableArrayList = convert(Array{Variable,1},obj.nodePoolVariables.nodes);
    floatTypeTree = floatType(originalTree);
    order = 1;
    while true
        numNodesNextTree = numNodesCurrentTree + numNodesTreeOrder(order, length(variableArrayList)) + 1;
        if numNodesNextTree>obj.maximumNodes
            obj.tree = originalTree;
            obj.mse = mseCopy;
            return false;
        end;
        obj.tree = AddNode( clone(originalTree), buildTreeOrder(order, variableArrayList, floatTypeTree) );
        @assert(numNodes(obj.tree) == numNodesNextTree);
        obj.mse = MSE(evaluateTree(obj.tree), obj.targets);
        obj.strategy(obj) && return true;
        order += 1;
    end;
end;




#################################################################################################################################################################
#
# The function is an interface that creates the DoME object and calls the function Step! on each iteration
#

function dome(inputs::AbstractArray{<:Real,2}, targets::AbstractArray{<:Real,1};
    # Each instance in inputs is in a row or in a column
    dataInRows          ::Bool                     = true,
    # Hyperparameters of the algorithm
    minimumReductionMSE ::Real                     = 1e-6,
    maximumNodes        ::Int64                    = 50 ,
    strategy            ::Function                 = StrategySelectiveWithConstantOptimization ,
    # Other hyperparameter that the user might find useful
    maximumHeight       ::Real                     = Inf ,
    # Stopping criteria
    goalMSE             ::Real                     = 0 ,
    maxIterations       ::Real                     = Inf ,
    executionTime       ::Real                     = Inf ,
    # Whether to use the division operator or not
    useDivisionOperator ::Bool                     = true ,
    # Indices of the instances used for validation and test
    validationIndices   ::AbstractVector{Int64}    = Int64[],
    testIndices         ::AbstractVector{Int64}    = Int64[],
    # Initial tree with its MSE
    initialTree::Union{Nothing,Tree,Tuple{Tree,Real}} = nothing ,
    # Function to be called at the end of each iteration
    callFunction        ::Union{Nothing, Function} = nothing ,
    # If you want to see the iterations on screen. This makes the execution slower
    showText            ::Bool                     = false ,
    # This parameter was used only for development. If it is set to true, the execution becomes much slower
    checkForErrors      ::Bool                     = false
    )

    elapsedTime = 0;

    # Validation and test ratios (these parameters have not been tested and may fail)
    # validationRatio     ::Real                     = 0. ,
    # testRatio           ::Real                     = 0. ,
    validationRatio                          = 0.
    testRatio                                = 0.

    @assert(length(targets)==size(inputs, dataInRows ? 1 : 2), "Inputs and targets have different number of instances");

    @assert(isempty(validationIndices) || (validationRatio==0.))
    @assert(isempty(testIndices) || (testRatio==0.))
    @assert(validationRatio+testRatio<1.);
    if ((!isempty(validationIndices)) && (!isempty(testIndices)))
        @assert(isempty(intersect(validationIndices,testIndices)));
    end;

    # Hold-out
    # randomIndices = randperm(length(targets));
    # if (isempty(validationIndices))
    #     numValidation = Int(round(length(targets)*validationRatio));
    #     validationIndices = randomIndices[1:numValidation];
    #     indices = sort(randomIndices[(numValidation+1):end]);
    # else
    #     @assert(maximum(validationIndices)<=length(targets));
    #     numValidation = length(validationIndices);
    #     randomIndices = setdiff(randomIndices, validationIndices);
    # end;
    # if (isempty(testIndices))
    #     numTest = Int(round(length(targets)*testRatio));
    #     testIndices = sort(randomIndices[1:numTest]);
    #     randomIndices = randomIndices[(numTest+1):end];
    # else
    #     @assert(maximum(testIndices)<=length(targets));
    #     numTest = length(testIndices);
    #     randomIndices = setdiff(randomIndices, testIndices);
    # end;
    # trainingIndices = sort(randomIndices);

    if !isempty(validationIndices)
        @assert(minimum(validationIndices)>0 && maximum(validationIndices)<=length(targets));
        numValidation = length(validationIndices);
    else
        numValidation = 0;
    end;
    if !isempty(testIndices)
        @assert(minimum(testIndices)>0 && maximum(testIndices)<=length(targets));
        numTest = length(testIndices);
    else
        numTest = 0;
    end;
    trainingIndices = setdiff(1:length(targets), vcat(validationIndices, testIndices));

    numTraining = length(targets) - numValidation - numTest;
    @assert(numTraining == length(trainingIndices));
    @assert(isempty(intersect(trainingIndices,validationIndices)) && isempty(intersect(trainingIndices,testIndices)) && isempty(intersect(validationIndices,testIndices)));
    @assert(!isempty(trainingIndices));

    sr = DoME(dataInRows ? inputs[trainingIndices,:] : inputs[:,trainingIndices], targets[trainingIndices];
        dataInRows = dataInRows ,
        minimumReductionMSE = eltype(inputs)(minimumReductionMSE),
        maximumNodes = maximumNodes,
        strategy = strategy ,
        maximumHeight = maximumHeight ,
        goalMSE = eltype(inputs)(goalMSE) ,
        useDivisionOperator = useDivisionOperator ,
        initialTree = initialTree ,
        checkForErrors = checkForErrors
    );

    classificationProblem = eltype(targets)==Bool;

    iteration = 0;

    BestTrainingMSE      =  Inf; BestValidationMSE      =  Inf; BestTestMSE      =  Inf;
    BestTrainingAccuracy = -Inf; BestValidationAccuracy = -Inf; BestTestAccuracy = -Inf;
    BestExpression = ""; BestTree =  nothing; # BestM = NaN; BestM0 = NaN;


    function evaluateTreeIndices(tree, indices; evaluateAsString::Bool=false)
        if evaluateAsString
            func = eval(Meta.parse(string("X -> ", vectorString(tree; dataInRows=dataInRows))));
            outputs = Base.invokelatest(func, inputs);
            if classificationProblem
                return ( MSE(isa(outputs,Real) ? outputs : outputs[indices], (targets[indices].*2).-1 ), Accuracy(isa(outputs,Real) ? outputs : outputs[indices], (targets[indices].*2).-1 ) );
            else
                return ( MSE(isa(outputs,Real) ? outputs : outputs[indices],  targets[indices]        ), NaN );
            end;
        else
            outputs = evaluateTree(tree, dataInRows ? view(inputs, indices, :) : view(inputs, :, indices))
            if classificationProblem
                return ( MSE(outputs , (view(targets, indices).*2).-1 ), Accuracy( outputs , (view(targets, indices).*2).-1 ) );
            else
                return ( MSE(outputs ,  view(targets, indices)        ), NaN );
            end;
        end;
    end;

    function evaluateIteration()
        if (showText)
            BestTrainingMSE = sr.mse;
            if checkForErrors
                @assert( BestTrainingMSE == evaluateTreeIndices(sr.tree, validationIndices; evaluateAsString=false)[1] )
                @assert( BestTrainingMSE == evaluateTreeIndices(sr.tree, validationIndices; evaluateAsString=true )[1] )
            end;
            println(" Iteration ", iteration);
            println(" Expression: ", string(sr.tree));
            println("   Nuber of nodes: ", numNodes(sr.tree), " - Height: ", height(sr.tree));
            println("   Current training:        MSE: ", BestTrainingMSE);
            if classificationProblem
                BestTrainingAccuracy = Accuracy(evaluateTree(sr.tree), sr.targets);
                println("                       Accuracy: ", BestTrainingAccuracy);
            end;
        end;

        # If there is validation set, the expression improves if it improves in the validation set
        if !isempty(validationIndices)
            mseValidation, validationAccuracy = evaluateTreeIndices(sr.tree, validationIndices);
            if classificationProblem
                improves = (validationAccuracy>BestValidationAccuracy) || ((validationAccuracy==BestValidationAccuracy) && (mseValidation<BestValidationMSE))
            else
                improves =                                                                                                 (mseValidation<BestValidationMSE)
            end;

            # If this expression is improved, save the tree and the results
            if improves
                BestTree = clone(sr.tree; resetSemantic=true);
                BestTrainingMSE = sr.mse;
                checkForErrors && @assert(BestTrainingMSE == evaluateTreeIndices(BestTree, trainingIndices)[1])
                if classificationProblem
                    BestTrainingAccuracy = Accuracy(evaluateTree(sr.tree), sr.targets);
                    checkForErrors && @assert(BestTrainingAccuracy == evaluateTreeIndices(BestTree, trainingIndices)[2])
                end;
                BestValidationMSE      = mseValidation;
                BestValidationAccuracy = validationAccuracy;
                if !isempty(testIndices)
                    BestTestMSE, BestTestAccuracy = evaluateTreeIndices(BestTree, testIndices);
                end;
            end;

            # Print the overall results
            if showText
                println("   Overall training:        MSE: ", BestTrainingMSE);
                if classificationProblem
                    println("                       Accuracy: ", BestTrainingAccuracy);
                end;
                println("           validation:      MSE: ", BestValidationMSE);
                if classificationProblem
                    println("                       Accuracy: ", BestValidationAccuracy);
                end;
                if !isempty(testIndices)
                    println("           test:            MSE: ", BestTestMSE);
                    if classificationProblem
                        println("                       Accuracy: ", BestTestAccuracy);
                    end;
                end;
            end;

        else
            # No validation indices

            # Print the overall results in test
            if showText && !isempty(testIndices)
                BestTestMSE, BestTestAccuracy = evaluateTreeIndices(sr.tree, testIndices);
                println("           test:            MSE: ", BestTestMSE);
                if classificationProblem
                    println("                       Accuracy: ", BestTestAccuracy);
                end;
            end;
        end;


        if showText
            println("           Nuber of nodes: ", numNodes(isnothing(BestTree) ? sr.tree : BestTree), " - Height: ", height(isnothing(BestTree) ? sr.tree : BestTree));
            println("-------------------------------------------------------------------------------------")
        end;

        !isnothing(callFunction) && callFunction(sr);

    end;

    # Evaluate iteration 0
    evaluateIteration();

    stoppingCriteria = false;
    while !stoppingCriteria

        elapsedTime += @elapsed begin

            stoppingCriteria = !Step!(sr);

            if !stoppingCriteria

                iteration += 1;

                # Evaluate each iteration
                evaluateIteration();

                # Stopping criteria
                if (sr.mse<=goalMSE) || (iteration>=maxIterations)
                    stoppingCriteria = true;
                end;

            end;

        end;

        if elapsedTime>executionTime
            stoppingCriteria = true;
        end;

    end;

    if isnothing(BestTree)
        @assert(isempty(validationIndices))
        BestTree = clone(sr.tree; resetSemantic=true);
        BestTrainingMSE = sr.mse;
        BestTrainingAccuracy = isinf(BestTrainingAccuracy) ? Accuracy(evaluateTree(sr.tree), sr.targets) : BestTrainingAccuracy;
        checkForErrors && @assert(BestTrainingAccuracy == evaluateTreeIndices(BestTree, trainingIndices)[2])
        if isinf(BestTestMSE)
            BestTestMSE, BestTestAccuracy = evaluateTreeIndices(BestTree, testIndices);
        end;
    end;

    if showText
        println("Best expression found: ", string(BestTree));
    end;

    return classificationProblem ? (BestTrainingAccuracy, BestValidationAccuracy, BestTestAccuracy, BestTree) : (BestTrainingMSE, BestValidationMSE, BestTestMSE, BestTree);

end;
