
include("Tree.jl")

mutable struct NodePool
    nodes::Array{Tree,1}
    numNodes::Array{UInt,1}
    heights::Array{UInt,1}
    semantics::Array{Semantic}
    hasZerosInSemantics::Array{Bool,1}
    toleranceValue::Real
    NodePool() = new([],[],[],[],[], 0.);
end


function NodePoolVariables(inputs::AbstractArray{<:Real,2}; dataInRows::Bool=true)
    # Create variables, add them to the node pool and to the prototypes
    if (dataInRows)
        variableValues = collect.(eachcol(inputs));
        # inputValues = [inputs[:,numInput] for numInput in 1:size(inputs,2)];
    else
        variableValues = collect.(eachrow(inputs));
        # inputValues = [inputs[numInput,:] for numInput in 1:size(inputs,1)];
    end;
    nodePoolVariables = NodePool();
    for numVariable in 1:length(variableValues)
        # if !allequal(variableValues[numVariable])
        #     nodeVariable = Variable(UInt(numVariable); semantic=variableValues[numVariable]);
        #     addNodeToPool!(nodePoolVariables, nodeVariable, UInt(1), UInt(1));
        # end;
        if allequal(variableValues[numVariable])
            nodeVariable = Variable(UInt(numVariable); semantic=mean(variableValues[numVariable]));
        else
            nodeVariable = Variable(UInt(numVariable); semantic=variableValues[numVariable]);
        end;
        addNodeToPool!(nodePoolVariables, nodeVariable, UInt(1), UInt(1));
        @assert(length(nodePoolVariables.nodes)==numVariable)
    end;
    return nodePoolVariables;
end;

function setVariableValues!(tree::Tree, nodePoolVariables::NodePool, inputs::AbstractArray{<:Real,2}; dataInRows::Bool=true)
    function findVariable(numVariable::UInt)
        for index in Base.OneTo(nodePoolVariables.length(nodes))
            (nodePoolVariables.nodes[index].variableNumber == numVariable) && return index;
        end;
        return -1;
    end;

    setVariableValues(node::Constant) = nothing;
    function setVariableValues(node::Variable)
        index = findVariable(node.variableNumber);
        if index!=-1
            node.semantic = nodePoolVariables.semantics[index];
        else
            numVariable = node.variableNumber;
            semantic = dataInRows ? inputs[:,numVariable] : inputs[numVariable,:];
            nodeVariable = Variable(UInt(numVariable); semantic=semantic);
            addNodeToPool!(nodePoolVariables, nodeVariable, UInt(1), UInt(1));
            node.semantic = semantic;
        end;
    end;
    setVariableValues(node::BinaryNode) = (setVariableValues(node.child1); setVariableValues(node.child2););
    setVariableValues(node::NonBinaryNode) = (for child in node.children setVariableValues(node.child); end;);

    setVariableValues(tree);
end;


function setVariableValues!(tree::Tree, nodePoolVariables::NodePool)
    setVariableValues(node::Constant) = (node.equation = nothing; return true;);
    function setVariableValues(node::Variable)
        index = node.variableNumber;
        if (index>length(nodePoolVariables.nodes))
            return false;
        end;
        numVariable = node.variableNumber;
        @assert(numVariable==index);
        node.semantic = nodePoolVariables.semantics[index];
        node.equation = nothing;
        return true;
    end;
    setVariableValues(node::BinaryNode) = (node.equation = nothing; return setVariableValues(node.child1) && setVariableValues(node.child2););
    setVariableValues(node::NonBinaryNode) = (node.equation = nothing; for child in node.children !setVariableValues(node.child) && return false; end; return true;);

    return setVariableValues(tree);
end;


function checkIntegrity(sol::NodePool)
    @assert (length(sol.nodes)==length(sol.numNodes));
    @assert (length(sol.nodes)==length(sol.heights));
    @assert (length(sol.nodes)==length(sol.semantics));
    for numNode = 1:length(sol.nodes)
        @assert (numNodes(sol.nodes[numNode])==sol.numNodes[numNode]);
        @assert (height(sol.nodes[numNode])==sol.heights[numNode]);
    end;
end;

function removeNodesFromPool!(sol::NodePool, indexNodes::Array{Int,1})
    numberOfNodes = length(sol.nodes);
    indexRemaining = setdiff(1:numberOfNodes, indexNodes);

    sol.nodes = sol.nodes[indexRemaining];
    sol.numNodes = sol.numNodes[indexRemaining];
    sol.heights = sol.heights[indexRemaining];
    sol.semantics = sol.semantics[indexRemaining];
    return indexRemaining;
end;

function addNodeToPool!(sol::NodePool, node::Tree, numNodes::UInt, height::UInt)
    semantic = node.semantic;
    @assert (!isnothing(semantic));
    # if (sol.toleranceValue!=0.)
    #     add = all( sqrt.( (x->sum(x.^2)).([semantic] .- sol.semantics)) .> sol.toleranceValue );
    # else
    #     add = all( [semantic] .!= sol.semantics );
    # end;

    # if (add)
        push!(sol.nodes, node);
        push!(sol.numNodes, numNodes);
        push!(sol.heights, height);
        push!(sol.semantics, semantic);
        # push!(sol.hasZerosInSemantics, any(semantic.==0));
        push!(sol.hasZerosInSemantics, any0(semantic));
        return true;
    # else
    #     return false;
    # end;
end;

