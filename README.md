# DoME

This library contains the source code of the DoME algorithm for Symbolic Regression. This algorithm is described in the paper available at https://doi.org/10.1016/j.eswa.2022.116712

This library is fully functional, feel free to use it to perform your experiments. However, if any publication is generated through this system, please add a citation to that paper. Also, if you need any more explanations on how to run DoME, or there is any issue with this repository, please let me know.

To run DoME, only the packages Statistics is needed.

# How to use DoME

The easiest way to wun DoME is by calling the function dome, which performs all of the operations. This function gets two parameters: inputs, as a real-number matrix, and targets, as a real-number vector. Also, it can receive hyperparameter values.

Here is an example of use, in which only the main hyperparameters are set:

	using FileIO
	using DelimitedFiles
	
	# Load the dataset and create a matrix with the inputs and a vector for the targets
	dataset = DelimitedFiles.readdlm("561_cpu.tsv");
	inputs  = Float64.(dataset[2:end, 1:end-1]);
	targets = Float64.(dataset[2:end, end]);

	# Load the DoME system
	using SymDoME
 
	# Run DoME with these parameters
	(trainingMSE, validationMSE, testMSE, bestTree) = dome(inputs, targets;
	   minimumReductionMSE = 1e-6,
	   maximumNodes = 30,
	   strategy = StrategyExhaustive,
	   showText = true
	);

	# Write the expression on screen
	println("Best expression found: ", string(bestTree));
	println("Best expression found (written in Latex): ", latexString(bestTree));
 
	# If you want to rename the variable names, one of the easiest way is to do something like:
	expression = string(bestTree);
	expression = replace(expression, "X1" => "vendor");
	expression = replace(expression, "X2" => "MYCT");
	expression = replace(expression, "X3" => "MMIN");
	expression = replace(expression, "X4" => "MMAX");
	expression = replace(expression, "X5" => "CACH");
	expression = replace(expression, "X6" => "CHMIN");
	expression = replace(expression, "X7" => "CHMAX");
	println("Best expression found (with the real names of the variables): ", expression);

When calling the function dome, inputs is a NxP matrix of real numbers, and targets is a N-length vector or real numbers (N: number of instances, P: number of attributes). Inputs and targets can have Float32 or Float64 values; however, since many constants are generated during the run of the algorithm, it is recommended to use Float64 to have the highest precision. Also, the elements of both inputs and targets must have the same type (Float32 or Float64). The parameters minimumReductionMSE, maximumNodes and strategy are the 3 hyperparameters described in the paper.

The declaration of this function is the following, with the whole set of parameters and their default values:

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
	    # Whether to use the division operator or not
	    useDivisionOperator ::Bool                     = true ,
	    # Indices of the instances used for validation and test
	    validationIndices   ::AbstractVector{Int64}    = Int64[],
	    testIndices         ::AbstractVector{Int64}    = Int64[],
	    # Function to be called at the end of each iteration
	    callFunction        ::Union{Nothing, Function} = nothing ,
	    # If you want to see the iterations on screen. This makes the execution slower
	    showText            ::Bool                     = false ,
	    )
    
The description of these parameters is the following:

	dataInRows -> allows the input matrix to have dimensions NxP when it is set to true (by default) or PxN when it is false (N: number of instances).
	minimumReductionMSE -> A search is found to be successful if the reduction in MSE is positive and higher than the previous MSE value multiplied by this parameter.
	maximumNodes -> maximum number of nodes in the tree.
	strategy -> the strategy used to select which searches are going to be performed on which nodes. The 4 strategies described in the paper are available, with names StrategyExhaustive (by default), StrategyExhaustiveWithConstantOptimization, StrategySelectiveWithConstantOptimization and StrategySelective. They are also called Strategy1, Strategy2, Strategy3, Strategy4 respectively as used in the paper.
	maximumHeight -> maximum height of the tree. As explained in the paper, this parameter is not recommended to be used in the experiments.
	goalMSE -> if the algorithm reaches this MSE value in training, the iterative process is stopped.
	maxIterations -> maximum number of iterations to be performed.
	validationIndices -> allows to split the dataset by separating some instances to perform the validation, specifying which ones will be used for validation.
	testIndices -> allows to split the dataset by separating some instances to perform the test, specifying which ones will be used for test.
	showText -> if it is set to true, on each iteration some text (iteration number, best tree, MSE in training and test) is shown.

As it can be sen, this function allows the definition of a validation set.

Once the dome function has been executed, it returns 4 values: training, validation and test results, and the tree found. To convert this tree into a text equation, 5 functions are provided:
- string. This function receives the tree and returns a String with the equation.
- vectorString. This function receives the tree and returns a String with the equation, but it is written to perform vector operations in Julia.
- latexString. This function received the tree and returns a String with the equation as a text in LaTeX, ready to use in your documents
- spreadsheetString. This function received the tree and returns a String with the equation ready to use in a spreadsheet. In this case, the attributes in the equation are making reference to the columns in the second row (the first one is supposed to have the names of the attributes), so you only have to paste and drag down the equation. 
- writeAsTree. This function receives the tree and does not return text, but instead of it, it writes the expression on screen as a tree, which can be useful if you want to analyse it.

An alternative way to run DoME is by creating a DoME struct and calling the function Step! for each iteration. This is automatically done by the previous way to run DoME.

# How to define your own strategy

Strategies are based on calling the functions PerformSearches! and OptimizeConstants!

The function PerformSearches! allows to specify in the call which nodes are going to be used on each search. To do this, this function has keyword parameters for each search, and in each one the user can specify type of the nodes in which this search will take place. These types are Terminal, Variable, Constant, and NonTerminal. Also, the types Any and Nothing can be used to specify that a search will be performed on all of the nodes, or in none of them respectively. The declaration of this function is the following:

	function PerformSearches!(obj::DoME;
	   whichNodesPerformConstantSearch        ::Union{DataType,Union} = Nothing ,
	   whichNodesPerformVariableSearch        ::Union{DataType,Union} = Nothing ,
	   whichNodesPerformConstantVariableSearch::Union{DataType,Union} = Nothing ,
	   performConstantExpressionSearch        ::Bool = false)

Note that constant-expression search receives a boolean value, because this search is only performed on non-terminal nodes.

This function returns a Boolean value: if it is true, a search has been succesful, otherwise no search was succesful. The strategy function to be defined should also return a Boolean value, with the same interpretation.

An example is the Exhaustive strategy, in which the searches are performed on all of the nodes of the tree:

	function StrategyExhaustive(obj::DoME)
	   changeDone = PerformSearches!(obj;
	      whichNodesPerformConstantSearch=Any ,
	      whichNodesPerformVariableSearch=Any ,
	      whichNodesPerformConstantVariableSearch=Any ,
	      performConstantExpressionSearch=true);
	   return changeDone;
	end;

Another example is the Selective strategy, that performs the searches performs searches sequentially, moving on to the next one only if the previous one has been unsuccessful:

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

Also, inside the strategy the function OptimizeConstants! can be called. This function performs the constant optimization process as described in the paper, and returns a boolean value with the same interpretation. The strategies corresponding to Exhaustive and Selective but with constant optimisation are as follows:

	function StrategyExhaustiveWithConstantOptimization(obj::DoME)
	   changeDone = PerformSearches!(obj;
	      whichNodesPerformConstantSearch=Union{Variable,NonTerminal} ,
	      whichNodesPerformVariableSearch=Any ,
	      whichNodesPerformConstantVariableSearch=Any ,
	      performConstantExpressionSearch=true);
	   changeDone && OptimizeConstants(obj);
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
	   changeDone && OptimizeConstants(obj);
	   return changeDone;
	end;

These four strategies are provided in the library and are available for use. They can be specified as a hyperparameter whan calling the funcion dome.
