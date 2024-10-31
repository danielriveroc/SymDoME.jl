module DoME

using Statistics

# The basic function with the interface
export dome

# For writing the DoME expressions
export latexString
export vectorString
export spreadsheetString
export writeAsTree

# If you want to write your own loop
export DOME
export Step!

# Basic strategies
export StrategyExhaustive
export StrategyExhaustiveWithConstantOptimization
export StrategySelectiveWithConstantOptimization
export StrategySelective

export Strategy1
export Strategy2
export Strategy3
export Strategy4

# If you want to write your own strategy, you need these:
export PerformSearches!
export Constant
export Variable
export Terminal
export NonTerminal
export OptimizeConstants!


include("main.jl")

end
