
# using Random

include("Equation.jl")

macro def(name, definition)
    return quote
        macro $(esc(name))()
            esc($(Expr(:quote, definition)))
        end
    end
end

@def addEquationField begin
    equation::Union{Nothing,NodeEquation}
end



abstract type Tree end

abstract type Terminal <: Tree end

mutable struct Variable <: Terminal
    @addEquationField
    variableNumber::UInt
    # semantic::Array{<:Real,1}
    semantic::Semantic
    Variable(num::UInt; semantic::Semantic=Float64[]) = new(nothing, num, semantic)
end

mutable struct Constant <: Terminal
    @addEquationField
    semantic::AbstractFloat
    Constant(value::AbstractFloat) = new(nothing, value)
end

# mutable struct RandomConstant <: Terminal
#     @addEquationField
#     lowerLimit::Real
#     upperLimit::Real
#     RandomConstant(lowerLimit::Real, upperLimit::Real) = new(nothing, lowerLimit, upperLimit)
# end


@def addNonTerminalFields begin
    name::String
    semantic::Union{Nothing,Semantic}
    evalFunction::Function
end

abstract type NonTerminal <: Tree end

mutable struct BinaryNode <: NonTerminal
    @addEquationField
    @addNonTerminalFields
    child1::Tree
    child2::Tree
    functionEquationChild1::Function
    functionEquationChild2::Function
    function BinaryNode(name::String, evalFunction::Function, child1::Tree, child2::Tree, functionEquationChild1::Function, functionEquationChild2::Function; semantic=nothing)
        return new(nothing, name, semantic, evalFunction, child1, child2, functionEquationChild1, functionEquationChild2);
    end;
end

mutable struct NonBinaryNode <: NonTerminal
    @addEquationField
    @addNonTerminalFields
    children::Array{Tree,1}
    functionEquationChildren::Array{Function,1}
    function NonBinaryNode(name::String, evalFunction::Function, functionEquationChildren::Array{Function,1}; semantic=nothing, children::Array{Tree,1}=[])
        return new(nothing, name, semantic, evalFunction, children, functionEquationChildren);
    end;
    function NonBinaryNode(type::Int, name::String, evalFunction::Function, functionEquationChildren::Array{Function,1}, children::Array{Tree,1}; semantic=nothing)
        return new(nothing, name, semantic, evalFunction, children, functionEquationChildren);
    end;
end


AddNode(child1::Tree, child2::Tree) = BinaryNode("+", addSemantics, child1, child2, equationAddChild1, equationAddChild2);
SubNode(child1::Tree, child2::Tree) = BinaryNode("-", subSemantics, child1, child2, equationSubChild1, equationSubChild2);
MulNode(child1::Tree, child2::Tree) = BinaryNode("*", mulSemantics, child1, child2, equationMulChild1, equationMulChild2);
DivNode(child1::Tree, child2::Tree) = BinaryNode("/", divSemantics, child1, child2, equationDivChild1, equationDivChild2);



clearEvaluationValues!(node::Constant) = nothing;
# clearEvaluationValues!(node::RandomConstant) = error("");
clearEvaluationValues!(node::Variable) = nothing;
function clearEvaluationValues!(node::BinaryNode)
    node.semantic = nothing;
    clearEvaluationValues!(node.child1);
    clearEvaluationValues!(node.child2);
    return nothing;
end
function clearEvaluationValues!(node::NonBinaryNode)
    node.semantic = nothing;
    for childNode in node.children
        clearEvaluationValues!(childNode);
    end;
    return nothing;
end


# These 3 functions clear the equations except in the given path
clearEquations!(node::Terminal) = ( node.equation = nothing; )
# function clearEquations!(node::BinaryNode; exceptInPath=nothing)
#     if isnothing(exceptInPath)
#         node.equation = nothing;
#         clearEquations!(node.child1);
#         clearEquations!(node.child2);
#     else
#         if (isempty(exceptInPath))
#             clearEquations!(node.child1);
#             clearEquations!(node.child2);
#         elseif (exceptInPath[1]==1)
#             clearEquations!(node.child1, exceptInPath[2:end]);
#             clearEquations!(node.child2);
#         else
#             clearEquations!(node.child1);
#             clearEquations!(node.child2, exceptInPath[2:end]);
#         end;
#     end;
# end
function clearEquations!(node::NonBinaryNode; exceptInPath=nothing)
    if isnothing(exceptInPath)
        node.equation = nothing;
        for child in node.children
            clearEquations!(child);
        end;
    else
        if (isempty(exceptInPath))
            for child in node.children
                clearEquations!(child);
            end;
        else
            nextStep = exceptInPath[1];
            for numChild in 1:length(node.children)
                if (numChild==nextStep)
                    clearEvaluationValues!(node.children[numChild], exceptInPath[2:end]);
                else
                    clearEvaluationValues!(node.children[numChild]);
                end;
            end;
        end;
    end;
end

function clearEquations!(node::BinaryNode; exceptInPaths=nothing)
    if isnothing(exceptInPaths)
        node.equation = nothing;
        clearEquations!(node.child1);
        clearEquations!(node.child2);
    else
        if (isempty(exceptInPaths))
            clearEquations!(node.child1);
            clearEquations!(node.child2);
        else
            firstSteps = [path[1] for path in exceptInPaths];
            has1 = anyequal(1,firstSteps);
            has2 = anyequal(2,firstSteps);
            if (has1) && (has2)
                followingSteps = exceptInPaths[firstSteps.==1];
                followingSteps = [path[2:end] for path in followingSteps];
                followingSteps = followingSteps[!isempty.(followingSteps)];
                clearEquations!(node.child1, exceptInPaths=followingSteps);
                followingSteps = exceptInPaths[firstSteps.==2];
                followingSteps = [path[2:end] for path in followingSteps];
                followingSteps = followingSteps[!isempty.(followingSteps)];
                clearEquations!(node.child1, exceptInPaths=followingSteps);
            elseif (has1)
                followingSteps = [path[2:end] for path in exceptInPaths];
                followingSteps = followingSteps[!isempty.(followingSteps)];
                clearEquations!(node.child1, exceptInPaths=followingSteps);
                clearEquations!(node.child2);
            elseif (has2)
                followingSteps = [path[2:end] for path in exceptInPaths];
                followingSteps = followingSteps[!isempty.(followingSteps)];
                clearEquations!(node.child1);
                clearEquations!(node.child2, exceptInPaths=followingSteps);
            else
                error("Don't know which path to go")
            end;
        end;
    end;
end




import Base.string
string(node::Constant) = node.semantic>=0 ? string(node.semantic) : string("(",node.semantic,")");
# string(node::RandomConstant) = error("");
string(node::Variable) = string("X", node.variableNumber);
string(node::BinaryNode) = string("(",string(node.child1),node.name,string(node.child2),")")
function string(node::NonBinaryNode)
    if (length(node.children)==2)
        text = string("(",string(node.children[1]),node.name,string(node.children[2]),")")
    else
        text = string(node.name, "(");
        for numChildren = 1:length(node.children)
            text = string(text, string(node.children[numChildren]));
            if (numChildren!=length(node.children))
                text = string(text, ",");
            end;
        end;
        text = string(text, ")");
    end;
    return text;
end;

# function string(node::Constant)
#     numDecimals = 30;
#     v = node.semantic
#     iszero(v) && return "0"
#     # !contains(str, ".")
#     function number_to_string(x)
#         # # Primero multiplicamos por 10^decimales y truncamos
#         # factor = 10^numDecimals
#         # truncated = trunc(x * factor) / factor
        
#         # # Convertimos a string y aseguramos que tenga todos los decimales
#         # str = string(truncated)
        
#         # # Si no tiene punto decimal, añadimos .0
#         # if !contains(str, ".")
#         #     str = str * ".0"
#         # end
        
#         # # Añadimos ceros si faltan decimales
#         # parts = split(str, ".")
#         # if length(parts) == 2 && length(parts[2]) < numDecimals
#         #     str = str * "0"^(numDecimals - length(parts[2]))
#         # end
        
#         # return str
#         x_integer = trunc(x);
#         x==x_integer && return string(Int(x_integer));
#         str = string(x_integer)*".";
#         x_decimal = abs(x-x_integer)
#         str_decimals = Vector{UInt}(undef, numDecimals)
#         for numDecimal in Base.OneTo(numDecimals)
#             x_decimal = x_decimal*10;
#             x_decimal_int = trunc(x_decimal);
#             str_decimals[numDecimal] = UInt(x_decimal_int);
#             x_decimal = x_decimal-x_decimal_int;
#         end;
#         # println(string(string.(str_decimals)...))
#         return string(string.(str_decimals)...)
#     end
#     d = floor(Int, log10(abs(v)));
#     if abs(d)<=3
#         s = number_to_string(v);
#         # s = string(BigFloat(v))
#         return v>0 ? s : "("*s*")"
#     end;
#     mantissa = v * ((10.)^(-d));
#     mantissa = BigFloat(mantissa)
#     # s = @sprintf("%.3f", v);
#     # s = s*"\\cdot 10^{$d}"
#     return "($(number_to_string(mantissa))e$(d)"
# end;


vectorString(node::Constant; dataInRows=true) = node.semantic>=0 ? string(node.semantic) : string("(",node.semantic,")");
# vectorString(node::RandomConstant; dataInRows=true) = error("");
vectorString(node::Variable; dataInRows=true) = dataInRows ? string("X[:,", node.variableNumber,"]") : string("X[", node.variableNumber,",:]");
vectorString(node::BinaryNode; dataInRows=true) = string("(", vectorString(node.child1; dataInRows=dataInRows), " .", node.name, " ", vectorString(node.child2; dataInRows=dataInRows), ")")
function vectorString(node::NonBinaryNode; dataInRows=true)
    if (length(node.children)==2)
        text = string("(", vectorString(node.children[1]; dataInRows=dataInRows), " .", node.name, " ", vectorString(node.children[2]; dataInRows=dataInRows), ")")
    else
        text = string(node.name, ".(");
        for numChildren = 1:length(node.children)
            text = string(text, vectorString(node.children[numChildren]; dataInRows=dataInRows));
            if (numChildren!=length(node.children))
                text = string(text, ",");
            end;
        end;
        text = string(text, ")");
    end;
    return text;
end;


latexString(node::Constant) = string(node);
# latexString(node::RandomConstant) = string(node);
latexString(node::Variable) = string("X_{", node.variableNumber,"}");
latexString(node::BinaryNode) = (node.name=="/") ? string("\\frac{",latexString(node.child1),"}{",latexString(node.child2),"}") : string("\\left(",latexString(node.child1),(node.name=="*") ? "\\cdot " : node.name,latexString(node.child2),"\\right)")
# latexString(node::NonBinaryNode) = string(node);
function latexString(node::NonBinaryNode)
    if (length(node.children)==2)
        text = string("\\left(",string(node.children[1]),node.name,string(node.children[2]),"\\right)")
    else
        text = string(node.name, "\\left(");
        for numChildren = 1:length(node.children)
            text = string(text, string(node.children[numChildren]));
            if (numChildren!=length(node.children))
                text = string(text, ',');
            end;
        end;
        text = string(text, "\\right)");
    end;
    return text;
end;




function spreadsheetString(tree::Tree)
    ssString(node::Constant) = replace( node.semantic>=0 ? string(node.semantic) : string("(",node.semantic,")") , "." => "," );
    # spreadsheetString(node::RandomConstant) = error("");
    function ssString(node::Variable)
        letters = ['A':'Z';]
        base = length(letters)
        result = []
        n = node.variableNumber;
        while n > 0
            remainder = (n - 1) % base + 1
            push!(result, letters[remainder])
            n = div(n - remainder, base)
        end
        columnName = join(reverse(result));
        return columnName*"2";
    end;
    ssString(node::BinaryNode) = string("(",ssString(node.child1),node.name,ssString(node.child2),")")
    function ssString(node::NonBinaryNode)
        if (length(node.children)==2)
            text = string("(",ssString(node.children[1]),node.name,ssString(node.children[2]),")")
        else
            text = string(node.name, "(");
            for numChildren = 1:length(node.children)
                text = string(text, ssString(node.children[numChildren]));
                if (numChildren!=length(node.children))
                    text = string(text, ",");
                end;
            end;
            text = string(text, ")");
        end;
        return text;
    end;

    return "="*ssString(tree);
end;

# Tuple(node::Constant) = node.semantic;
# Tuple(node::Variable) = node.variableNumber;
# Tuple(node::BinaryNode) = (node.name, Tuple(node.child1), Tuple(node.child2));

asTuple(node::Constant) = node.semantic;
asTuple(node::Variable) = node.variableNumber;
asTuple(node::BinaryNode) = (node.name, asTuple(node.child1), asTuple(node.child2));

asVector(node::Constant) = node.semantic;
asVector(node::Variable) = node.variableNumber;
asVector(node::BinaryNode) = [node.name, asVector(node.child1), asVector(node.child2)];

evaluateTree(nodeDescription::AbstractFloat, dataset::AbstractArray{<:AbstractFloat,2}; dataInRows=true) = nodeDescription;
function evaluateTree(nodeDescription::UInt, dataset::AbstractArray{<:AbstractFloat,2}; dataInRows=true)
    if dataInRows
        nodeDescription>size(dataset,2) && error("Not enough columns in the dataset");
        return view(dataset, :, nodeDescription);
    else
        nodeDescription>size(dataset,1) && error("Not enough rows in the dataset");
        return view(dataset, nodeDescription, :);
    end;
end;
function evaluateTree(nodeDescription::Tuple, dataset::AbstractArray{<:AbstractFloat,2}; dataInRows=true)
    @assert(length(nodeDescription)==3)
    op = nodeDescription[1];
    if op=="+"
        return evaluateTree(nodeDescription[2], dataset; dataInRows=dataInRows) .+ evaluateTree(nodeDescription[3], dataset; dataInRows=dataInRows);
    elseif op=="-"
        return evaluateTree(nodeDescription[2], dataset; dataInRows=dataInRows) .- evaluateTree(nodeDescription[3], dataset; dataInRows=dataInRows);
    elseif op=="*"
        return evaluateTree(nodeDescription[2], dataset; dataInRows=dataInRows) .* evaluateTree(nodeDescription[3], dataset; dataInRows=dataInRows);
    elseif op=="/"
        return evaluateTree(nodeDescription[2], dataset; dataInRows=dataInRows) ./ evaluateTree(nodeDescription[3], dataset; dataInRows=dataInRows);
    else
        error("Unknown operator")
    end;
end;
function evaluateTree(nodeDescription::Vector, dataset::AbstractArray{<:AbstractFloat,2}; dataInRows=true)
    @assert(length(nodeDescription)==3)
    op = nodeDescription[1];
    if op=="+"
        return evaluateTree(nodeDescription[2], dataset; dataInRows=dataInRows) .+ evaluateTree(nodeDescription[3], dataset; dataInRows=dataInRows);
    elseif op=="-"
        return evaluateTree(nodeDescription[2], dataset; dataInRows=dataInRows) .- evaluateTree(nodeDescription[3], dataset; dataInRows=dataInRows);
    elseif op=="*"
        return evaluateTree(nodeDescription[2], dataset; dataInRows=dataInRows) .* evaluateTree(nodeDescription[3], dataset; dataInRows=dataInRows);
    elseif op=="/"
        return evaluateTree(nodeDescription[2], dataset; dataInRows=dataInRows) ./ evaluateTree(nodeDescription[3], dataset; dataInRows=dataInRows);
    else
        error("Unknown operator")
    end;
end;
evaluateTree(tree, instance::AbstractVector{<:AbstractFloat}) = evaluateTree(tree, reshape(instance, 1, :); dataInRows=true)[1];


function evaluateTree(expression::String, dataset::AbstractArray{<:AbstractFloat,2}; dataInRows=true)

    numVariables = size(dataset, dataInRows ? 2 : 1);

    if occursin("X[",expression)
        if dataInRows && occursin(",:]",expression)
            for numInput in numVariables:-1:1
                expression = replace(expression, "X[$numInput,:]" => "X[:,$numInput]");
            end;
        elseif (!dataInRows) && occursin("X[:,",expression)
            for numInput in numVariables:-1:1
                expression = replace(expression, "X[:,$numInput]" => "X[$numInput,:]");
            end;
        end;
    elseif occursin("X",expression)
        expression = replace(expression, "+" => " .+ ");
        expression = replace(expression, "e-" => "R_R");
        expression = replace(expression, "-" => " .- ");
        expression = replace(expression, "R_R" => "e-");
        expression = replace(expression, "*" => " .* ");
        expression = replace(expression, "/" => " ./ ");
        for numInput in numVariables:-1:1
            expression = replace(expression, "X$numInput" => dataInRows ? "X[:,$numInput]" : "X[$numInput,:]");
        end;
    end;
    func = eval(Meta.parse(string("X -> ", expression)));
    return Base.invokelatest(func, dataset);

end;


writeIndentSpaces(n) = ( for i=1:n print(" "); end; )
writeAsTree(node::Constant; n=0) = ( writeIndentSpaces(n); println(node.semantic); )
# writeAsTree(node::RandomConstant; n=0) = error("");
writeAsTree(node::Variable; n=0) = ( writeIndentSpaces(n); println("X", node.variableNumber); )
# writeAsTree(node::BinaryNode; n=0) = ( writeIndentSpaces(n); println(node.name); writeAsTree(node.child1; n=n+3); writeAsTree(node.child2; n=n+3); )
writeAsTree(node::BinaryNode; n=0) = ( writeAsTree(node.child1; n=n+3); writeIndentSpaces(n); println(node.name); writeAsTree(node.child2; n=n+3); )




evaluateTree(tree::Terminal; checkForErrors=false) = tree.semantic;
# evaluateTree(tree::RandomConstant; checkForErrors=false) = error("");
function evaluateTree(tree::BinaryNode; checkForErrors=false)
    if isnothing(tree.semantic)
        # tree.semantic = tree.evalFunction( evaluateTree(tree.child1;checkForErrors=checkForErrors), evaluateTree(tree.child2;checkForErrors=checkForErrors), nothing );
        tree.semantic = tree.evalFunction( evaluateTree(tree.child1;checkForErrors=checkForErrors), evaluateTree(tree.child2;checkForErrors=checkForErrors), tree.semantic );
        # tree.semantic = tree.evalFunction( evaluateTree(tree.child1;checkForErrors=checkForErrors), evaluateTree(tree.child2;checkForErrors=checkForErrors) );
    end;
    return tree.semantic;
end;
function evaluateTree(tree::NonBinaryNode; checkForErrors=false)
    if isnothing(tree.semantic)
        evaluationChildren = [evaluateTree(child; checkForErrors=checkForErrors) for child in tree.children];
        tree.semantic = tree.evalFunction( evaluationChildren... );
    end;
    if (checkForErrors)
        @assert(!any(isinf, tree.semantic));
        @assert(!any(isnan, tree.semantic));
    end;
    return tree.semantic;
end;


evaluateTree(node::Constant, dataset::AbstractArray{<:AbstractFloat,2}; dataInRows=true) = node.semantic;
function evaluateTree(node::Variable, dataset::AbstractArray{<:AbstractFloat,2}; dataInRows=true)
    if dataInRows
        node.variableNumber>size(dataset,2) && error("Not enough columns in the dataset");
        return view(dataset, :, node.variableNumber);
    else
        node.variableNumber>size(dataset,1) && error("Not enough rows in the dataset");
        return view(dataset, node.variableNumber, :);
    end;
end;
# evaluateTree(node::RandomConstant, dataset::AbstractArray{<:Real,2}; dataInRows=true) = error("");
evaluateTree(node::BinaryNode, dataset::AbstractArray{<:AbstractFloat,2}; dataInRows=true) = node.evalFunction( evaluateTree(node.child1, dataset; dataInRows=dataInRows), evaluateTree(node.child2, dataset; dataInRows=dataInRows) );
function evaluateTree(node::NonBinaryNode, dataset::AbstractArray{<:AbstractFloat,2}; dataInRows=true)
    evaluationChildren = [evaluateTree(child, dataset; dataInRows=dataInRows) for child in node.children];
    return node.evalFunction( evaluationChildren... );
end;
evaluateTree(tree::Tree, instance::AbstractVector{<:AbstractFloat}) = evaluateTree(tree, reshape(instance, 1, :); dataInRows=true)[1];


reevaluatePath(tree::Terminal, path::Array{Int64,1}, indexPath::Int64; checkForErrors=false) = tree.semantic;
# reevaluatePath(tree::RandomConstant, path::Array{Int64,1}, indexPath::Int64; checkForErrors=false) = error("");
function reevaluatePath(tree::BinaryNode, path::Array{Int64,1}, indexPath::Int64=1; checkForErrors=false)
    if (indexPath>length(path))
        semanticChild1 =   evaluateTree(tree.child1;            checkForErrors=checkForErrors);
        semanticChild2 =   evaluateTree(tree.child2;            checkForErrors=checkForErrors);
    elseif (path[indexPath]==1)
        semanticChild1 = reevaluatePath(tree.child1, path, indexPath+1; checkForErrors=checkForErrors);
        semanticChild2 =   evaluateTree(tree.child2;            checkForErrors=checkForErrors);
    elseif (path[indexPath]==2)
        semanticChild1 =   evaluateTree(tree.child1;            checkForErrors=checkForErrors);
        semanticChild2 = reevaluatePath(tree.child2, path, indexPath+1; checkForErrors=checkForErrors);
    else
        error("Don't know which path to go")
    end;
    tree.semantic = tree.evalFunction( semanticChild1, semanticChild2, tree.semantic );
    if (checkForErrors)
        @assert(!any(isinf.(tree.semantic)));
        @assert(!any(isnan.(tree.semantic)));
    end;
    return tree.semantic;
end;




numNodes(node::Terminal) = UInt(1);
numNodes(node::BinaryNode) = UInt(1 + numNodes(node.child1) + numNodes(node.child2));
numNodes(node::NonBinaryNode) = UInt(1 + sum([numNodes(child) for child in node.children]));

height(node::Terminal) = UInt(1);
height(node::BinaryNode) = UInt(1) + max(height(node.child1), height(node.child2));
height(node::NonBinaryNode) = UInt(1) + maximum([height(child) for child in node.children]);

clone(node::Constant; resetSemantic::Bool=false) = Constant(node.semantic);
# clone(node::RandomConstant) = Constant(rand()*(node.upperLimit-node.lowerLimit) + node.lowerLimit, node.type);
# clone(node::Variable) = Variable(node.variableNumber; semantic=(isnothing(node.semantic) ? node.semantic : copy(node.semantic)));
clone(node::Variable;      resetSemantic::Bool=false) = Variable(     node.variableNumber; semantic= resetSemantic ? Float32[] : node.semantic); # All of the variables share the reference to the semantics
# clone(node::BinaryNode;    resetSemantic::Bool=false) = BinaryNode(   node.name, node.evalFunction, clone(node.child1), clone(node.child2), node.functionEquationChild1, node.functionEquationChild2; semantic=(isnothing(node.semantic) ? node.semantic : copy(node.semantic)));
clone(node::BinaryNode;    resetSemantic::Bool=false) = BinaryNode(   node.name, node.evalFunction, clone(node.child1; resetSemantic=resetSemantic), clone(node.child2; resetSemantic=resetSemantic), node.functionEquationChild1, node.functionEquationChild2; semantic=(resetSemantic || isnothing(node.semantic)) ? nothing : copy(node.semantic));
# clone(node::NonBinaryNode; resetSemantic::Bool=false) = NonBinaryNode(node.name, node.evalFunction, node.equationsFunction; semantic=(isnothing(node.semantic) ? node.semantic : copy(node.semantic)), children=convert(Array{Tree,1},[clone(child) for child in node.children]));
clone(node::NonBinaryNode; resetSemantic::Bool=false) = NonBinaryNode(node.name, node.evalFunction, node.equationsFunction; semantic=((resetSemantic || isnothing(node.semantic)) ? nothing : copy(node.semantic)), children=convert(Array{Tree,1},[clone(child; resetSemantic=resetSemantic) for child in node.children]));



iterateTree(tree::Terminal) = (convert(Array{Tree,1},[tree]), [0x0000000000000001], [0x0000000000000001], [0x0000000000000000], [0x0000000000000001]);
function iterateTree(tree::BinaryNode)
    (nodesChild1, heightsChild1, depthsChild1, numChildrenChild1, numNodesChild1) = iterateTree(tree.child1);
    (nodesChild2, heightsChild2, depthsChild2, numChildrenChild2, numNodesChild2) = iterateTree(tree.child2);
    nodes = [nodesChild1; nodesChild2]; pushfirst!(nodes, tree);
    heights = [heightsChild1; heightsChild2]; pushfirst!(heights, 0x0000000000000000);
    depths = [depthsChild1; depthsChild2]; pushfirst!(depths, 0x0000000000000001);
    numChildren = [numChildrenChild1; numChildrenChild2]; pushfirst!(numChildren, 0x0000000000000001);
    numNodes = [numNodesChild1; numNodesChild2]; pushfirst!(numNodes, 0x0000000000000000);
    heights[1] = maximum(heights[2:end])+1;
    numNodes[1] = length(numNodes);
    depths[2:end] .+= 1;
    return (nodes, heights, depths, numChildren, numNodes);
end
function iterateTree(tree::NonBinaryNode)
    nodes = [tree];
    heights = [0x0000000000000000];
    depths = [0x0000000000000001];
    numChildren = [length(tree.children)];
    numNodes = [0x0000000000000000];
    for child in tree.children
        (nodesThisChild, tiposThisChild, heightsThisChild, depthsThisChild, numChildrenThisChild, numNodesThisChild) = iterateTree(child);
        nodes = [nodes; nodesThisChild];
        # hcat(tipos, tiposEsteHijo);
        heights = [heights; heightsThisChild];
        depths = [depths; depthsThisChild.+1];
        numChildren = [numChildren; numChildrenThisChild];
        numNodes = [numNodes; numNodesThisChild];
    end;
    heights[1] = maximum(heights[2:end])+1;
    numNodes[1] = length(numNodes);
    depths[2:end] .+= 1;
    return (nodes, heights, depths, numChildren, numNodes);
end



calculateEquations!(tree::Terminal, equation::NodeEquation; path=nothing, indexPath=1, checkForErrors=false) = ( tree.equation = equation; );
function calculateEquations!(tree::BinaryNode, equation::NodeEquation; path=nothing, indexPath=1, checkForErrors=false)
    if (checkForErrors)
        @assert(!isnothing(tree.semantic));
        checkEquation(equation);
    end;
    tree.equation = equation;

    if isnothing(path)
        # if isnothing(tree.child1.equation)
        #     calculateEquations!(tree.child1, tree.functionEquationChild1( evaluateTree(tree.child2; checkForErrors=checkForErrors), equation); checkForErrors=checkForErrors );
        # else
        #     calculateEquations!(tree.child1, tree.child1.equation                                                                            ; checkForErrors=checkForErrors );
        # end;
        # if isnothing(tree.child2.equation)
        #     calculateEquations!(tree.child2, tree.functionEquationChild2( evaluateTree(tree.child1; checkForErrors=checkForErrors), equation); checkForErrors=checkForErrors );
        # else
        #     calculateEquations!(tree.child2, tree.child2.equation                                                                            ; checkForErrors=checkForErrors );
        # end;
        # calculateEquations!(tree.child1, tree.functionEquationChild1( evaluateTree(tree.child2; checkForErrors=checkForErrors), equation); checkForErrors=checkForErrors );
        # calculateEquations!(tree.child2, tree.functionEquationChild2( evaluateTree(tree.child1; checkForErrors=checkForErrors), equation); checkForErrors=checkForErrors );
        calculateEquations!(tree.child1, !isnothing(tree.child1.equation) ? tree.child1.equation : tree.functionEquationChild1( evaluateTree(tree.child2; checkForErrors=checkForErrors), equation); checkForErrors=checkForErrors );
        calculateEquations!(tree.child2, !isnothing(tree.child2.equation) ? tree.child2.equation : tree.functionEquationChild2( evaluateTree(tree.child1; checkForErrors=checkForErrors), equation); checkForErrors=checkForErrors );
    else
        (indexPath>length(path)) && return
        if (path[indexPath]==1)
            calculateEquations!(tree.child1, tree.functionEquationChild1( evaluateTree(tree.child2; checkForErrors=checkForErrors), equation); path=path, indexPath=indexPath+1, checkForErrors=checkForErrors );
        elseif (path[indexPath]==2)
            calculateEquations!(tree.child2, tree.functionEquationChild2( evaluateTree(tree.child1; checkForErrors=checkForErrors), equation); path=path, indexPath=indexPath+1, checkForErrors=checkForErrors );
        else
            error("Don't know which path to go");
        end;
    end;
end;







findNode(tree::Terminal, node::Tree) = (tree==node) ? Array{Int,1}([]) : nothing
function findNode(tree::BinaryNode, node::Tree)
    (tree==node) && return Array{Int,1}([]);
    path = findNode(tree.child1, node);
    if (path!=nothing)
        pushfirst!(path, 1);
        return path;
    end;
    path = findNode(tree.child2, node);
    if (path!=nothing)
        pushfirst!(path, 2);
        return path;
    end;
    return nothing;
end;
function findNode(tree::NonBinaryNode, node::Tree)
    (tree==node) && return Array{Int,1}([]);
	for numChild in 1:length(tree.children)
        path = findNode(tree.children[numChild], node);
        if (path!=nothing)
            pushfirst!(path, numChild);
            return path;
        end;
    end;
    return nothing;
end;



# replaceSubtree!(tree::Terminal, oldSubTree::Tree, newSubTree::Tree) = false;
replaceSubtree!(tree::Terminal, oldSubTree::Tree, newSubTree::Tree) = false;
function replaceSubtree!(tree::BinaryNode, oldSubTree::Tree, newSubTree::Tree)
    replaced = false;
    if (tree.child1===oldSubTree)
        newSubTree.equation = tree.child1.equation;
        tree.child1 = newSubTree;
        clearEquations!(tree.child2);
# clearEquations!(tree.child1);
        replaced = true;
    elseif (tree.child2===oldSubTree)
        newSubTree.equation = tree.child2.equation;
        tree.child2 = newSubTree;
        clearEquations!(tree.child1);
# clearEquations!(tree.child2);
        replaced = true;
    else
        replaced = replaceSubtree!(tree.child1, oldSubTree, newSubTree);
        if replaced
            clearEquations!(tree.child2);
        else
            replaced = replaceSubtree!(tree.child2, oldSubTree, newSubTree);
            replaced && clearEquations!(tree.child1);
        end;
    end;
    if (replaced)
        # tree.semantic = nothing; evaluateTree(tree);
        tree.semantic = reevaluatePath(tree, Int64[]);
        return true;
    end;
# clearEquations!(tree);
    return false;
end
# function replaceSubtree!(tree::BinaryNode, oldSubTree::Tree, newSubTree::Tree)
#     replaced = false;
#     if (tree.child1==oldSubTree)
#         tree.child1 = newSubTree;
#         replaced = true;
#     elseif (tree.child2==oldSubTree)
#         tree.child2 = newSubTree;
#         replaced = true;
#     else
#         replaced = replaceSubtree!(tree.child1, oldSubTree, newSubTree);
#         if !replaced
#             replaced = replaceSubtree!(tree.child2, oldSubTree, newSubTree);
#         end;
#     end;
#     if (replaced)
#         tree.semantic = reevaluatePath(tree, Int64[]);
#         return true;
#     end;
#     return false;
# end
# function replaceSubtree!(tree::NonBinaryNode, oldSubTree::Tree, newSubTree::Tree)
# 	for numChild in 1:length(tree.children)
#         child = tree.children[numChild]
#         if (child==oldSubTree)
#             tree.children[numChild] = newSubTree;
#             tree.semantic = reevaluatePath(tree, Int64[]);
#             return true;
#         else
#             if (replaceSubtree!(child, oldSubTree, newSubTree))
#                 tree.semantic = reevaluatePath(tree, Int64[]);
#                 return true;
#             end;
#         end;
#     end;
#     return false;
# end



# function checkEquations(tree::Tree, mse::Real, targets::Semantic)
function checkEquations(tree::Tree, mse::AbstractFloat)
    @assert(!any(isnan, tree.semantic));
    @assert(!any(isinf, tree.semantic));
    if (tree.equation!=nothing)
        checkEquation(tree.equation);
# println( (mse, MSE(calculateOutputsFromEquation(tree.semantic, tree.equation; checkForErrors=true), targets, M, M0)) );
# if !isapprox_DoME(mse, calculateMSEFromEquation(tree.semantic, tree.equation; checkForErrors=true))
        # @assert(isapprox_DoME(mse, calculateMSEFromEquation(tree.semantic, tree.equation; checkForErrors=true)));
    end;
    if (isa(tree,BinaryNode))
        checkEquations(tree.child1, mse)
        checkEquations(tree.child2, mse)
    elseif (isa(tree,NonBinaryNode))
        for numChild in 1:length(tree.children)
            checkEquations(tree.children[numChild], mse)
        end;
    end;
end;



floatType(node::Constant) = eltype(node.semantic);
floatType(node::Variable) = eltype(node.semantic);
function floatType(node::BinaryNode)
    !isnothing(node.semantic) && return eltype(node.semantic);
    floatTypeChild1 = floatType(node.child1);
    !isnothing(floatTypeChild1) && return floatTypeChild1;
    return floatType(node.child2);
end;


function buildTreeOrder(order::Integer, variables::AbstractArray{Variable,1}, floatType::DataType)

    function buildVariableCombinationTreeList(thisOrder::Integer, variablesThisOrder::AbstractArray{Variable,1})
        (thisOrder==0) && return [Constant(floatType(0.))];
        treeListThisOrder = [];
        for numVariable in 1:length(variablesThisOrder)
            subTrees = buildVariableCombinationTreeList(thisOrder-1, view(variablesThisOrder, numVariable:length(variablesThisOrder)));
            for subTree in subTrees
                push!(treeListThisOrder, MulNode(subTree, clone(variablesThisOrder[numVariable])) );
            end;
        end;
        return treeListThisOrder;
    end;

    order<=0 && return nothing;

    treeList = buildVariableCombinationTreeList(order, variables);
    tree = treeList[1];
    for subTree in view(treeList, 2:length(treeList))
        tree = AddNode( tree, subTree );
    end;
    return tree;
end;


# Only calculates the number of nodes in a tree of a specific order, not with preceding orders
function numNodesTreeOrder(order::Integer, numVariables::Integer)
    numCombinations = binomial(numVariables + order - 1, order);
    numNodesEachCombination = order + order + 1; # Variables + Multiplications + Constant 0.0
    # println("Number of combinations: ", numCombinations, "     numNodesEachCombination: ", numNodesEachCombination)
    return numCombinations*numNodesEachCombination + numCombinations - 1; # Add the number of sum operations - 1
end;
