
using Statistics

# DoME_Tolerance0 = 1e-10;
DoME_Tolerance0 = 1e-15;
DoME_ToleranceApprox = 1e-8;
const DoME_UseSSet = false;
# const DoME_UseSSet = true;


Semantic = Union{<:AbstractFloat,AbstractArray{<:AbstractFloat,1}}
SemanticSet = Union{<:AbstractFloat,Array{Semantic,1}}

@inline valueSemantic(semantic::AbstractFloat, index::Int) = semantic;
@inline valueSemantic(semantic::AbstractArray{<:AbstractFloat,1}, index::Int) = @inbounds semantic[index];

# Use "mutable" here so this struct is allocated in the heap, and the garbage collector can free the memory stored on a,b,c,d,S
mutable struct NodeEquation
    a::Semantic
    b::Semantic
    c::Semantic
    d::Semantic
    S::SemanticSet
    function NodeEquation(a,b,c,d,S)
        a = cleanVector(a);
        b = cleanVector(b);
        c = cleanVector(c);
        d = cleanVector(d);
        return new(a,b,c,d,S)
    end;
    NodeEquation(a,b,c,d) = NodeEquation(a,b,c,d,Semantic[])
end

#############################################################################################################
#
# Useful functions for evaluating the semantics
#

# if iszero(DoME_ToleranceApprox)
#     @inline isapprox_DoME(x,y,radius) = x==y
#     @inline isapprox_DoME(x,y) = x==y
# else
    @inline isapprox_DoME(x,y,radius) = abs(x-y)<=radius
    # @inline isapprox_DoME(x,y) = isapprox_DoME(x,y,DoME_ToleranceApprox*max(abs(x),abs(y)))
    @inline isapprox_DoME(x,y) = (isfinite(x) && isfinite(y)) ? isapprox_DoME(x,y,DoME_ToleranceApprox*max(abs(x),abs(y))) : false
# end;

# if iszero(DoME_Tolerance0)
#     @inline isapprox0(x::Real) = iszero(x);
# else
    # @inline isapprox0(x::Real; tolerance=Tolerance0) = iszero(x) || abs(x)<tolerance;
    @inline isapprox0(x::AbstractFloat) = abs(x)<=DoME_Tolerance0;
# end;

# @inline isapproxDoME(x,y) = x==y

@inline allequal(x::AbstractFloat) = true
# @inline function allequal(x::AbstractArray{<:Real,1})
#     length(x) < 2 && return true
#     v = x[1]
#     @inbounds for i=2:length(x)
#         isapprox(x[i], v) || return false
#     end
#     return true
# end
# @inline allequal(x::AbstractArray{<:Real,1}, v::Real) = all(a -> isapprox_DoME(a,v), x)
# @inline allequal(x::AbstractArray{<:Real,1}) = allequal(x, mean(x))
@inline allequal(x::AbstractArray{<:AbstractFloat,1}) = isapprox_DoME(extrema(x)...)

# @inline allequal(x::AbstractArray{<:Real,1}) = allequal(x, x[1])

# @inline valueAllEqual(x::AbstractArray{<:Real,1}) = ( kx = x[1]; allequal(x, kx) ? kx : NaN; )
# @inline valueAllEqual(x::AbstractArray{<:Real,1}) = ( kx = mean(x); allequal(x, kx) ? kx : NaN; )
# @inline function valueAllEqual(v::AbstractArray{<:Real,1})
#     m,M = extrema(v);
#     !isapprox_DoME(m,M) && return NaN;
#     return (m+M)/2;
# end;
# @inline function valueAllEqual(v::AbstractArray{<:Real,1})
#     mean_val = zero(eltype(v));
#     for (i, elem) in enumerate(v)
#         mean_val += (elem - mean_val) / i
#         !isapprox_DoME(elem, mean_val) && return NaN
#     end
#     return mean_val;
# end
# @inline valueAllEqual(x::Real) = x
# @inline valueAllEqual(x::Real) = isapprox0(x) ? zero(x) : x

@inline any0(x::AbstractFloat) = isapprox0(x)
@inline any0(x::AbstractArray{<:AbstractFloat,1}) = any(isapprox0, x)

# If vectors x and y have any zero in the same position
@inline any0(x::AbstractFloat,y::AbstractFloat) = isapprox0(x) && isapprox0(y)
@inline any0(x::AbstractFloat,y::AbstractArray{<:AbstractFloat,1}) = isapprox0(x) ? any0(y) : false
@inline any0(x::AbstractArray{<:AbstractFloat,1},y::AbstractFloat) = any0(y,x)
@inline function any0(x::AbstractArray{<:AbstractFloat,1},y::AbstractArray{<:AbstractFloat,1})
    # @inbounds for i=1:length(x)
    #     isapprox0(x[i]) && isapprox0(y[i]) && return true
    # end
    @inbounds for (i,j) in zip(x,y)
        isapprox0(i) && isapprox0(j) && return true
    end;
    return false
end

@inline anyInf(x::AbstractFloat) = isinf(x)
@inline anyInf(x::AbstractArray{<:AbstractFloat,1}) = any(isinf, x)

@inline anyNaN(x::AbstractFloat) = isnan(x)
@inline anyNaN(x::AbstractArray{<:AbstractFloat,1}) = any(isnan, x)

@inline all0(x::AbstractFloat) = isapprox0(x)
@inline all0(x::AbstractArray{<:AbstractFloat,1}) = all(isapprox0, x)

@inline all1(x::AbstractFloat) = isone(x)
@inline all1(x::AbstractArray{<:AbstractFloat,1}) = all(isone, x)
# @inline function all1(x::AbstractArray{<:Real,1})
#     @inbounds for i=1:length(x)
#         x[i]!=1 && return false
#     end
#     return true
# end

# @inline anyequal(x::Real, y::Real) = (x==y)
# @inline function anyequal(x::Real, y::AbstractArray{<:Real,1})
#     @inbounds for i=1:length(y)
#         y[i]==x && return true
#     end
#     return false
# end
# @inline anyequal(x::AbstractArray{<:Real,1}, y::Real) = anyequal(y,x)
# @inline function anyequal(x::AbstractArray{<:Real,1}, y::AbstractArray{<:Real,1})
#     @inbounds for (i,j) in zip(x,y)
#         isapprox(i,j) && return true;
#     end;
#     return false
#     # @inbounds for i=1:length(x)
#     #     x[i]==y[i] && return true
#     # end
#     # return false
# end

@inline anyequal(x::AbstractFloat, y::AbstractFloat) = isapprox_DoME(x,y)
# @inline anyequal(x::AbstractArray{<:Real,1}, y::Real) = any(a -> isapprox_DoME(a,y), x)
# @inline anyequal(x::Real, y::AbstractArray{<:Real,1}) = anyequal(y,x)
# @inline function anyequal(x::AbstractArray{<:Real,1}, y::Real)
#     @inbounds for i in x
#         i==y && return true
#         # abs(i-y)<(1e-10)*max(i,y) && return true;
#         # isapprox(x[i],y) && return true
#     end
#     return false
# end
# @inline function anyequal(x::AbstractArray{<:Real,1}, y::Real)
#     v = (1e-10)*y;
#     @inbounds for i in x
#         # i==y && return true
#         # abs(i-y)<(1e-10)*max(i,y) && return true;
#         isapprox_DoME(i,y,v) && return true;
#         # abs(i-y)<v && return true;
#         # isapprox(x[i],y) && return true
#     end
#     return false
# end
@inline anyequal(x::AbstractFloat, y::AbstractFloat, radius::AbstractFloat) = isapprox_DoME(x,y,radius)
@inline anyequal(x::AbstractArray{<:AbstractFloat,1}, y::AbstractFloat, radius::AbstractFloat) = any(arg -> isapprox_DoME(arg,y,radius), x);
@inline anyequal(x::AbstractFloat, y::AbstractArray{<:AbstractFloat,1}, radius::AbstractFloat) = anyequal(y, x, radius);
# @inline anyequal(x::AbstractArray{<:Real,1}, y::Real) = ( radius = (1e-10)*y; any(arg -> isapprox_DoME(arg,y,radius), x); )
# @inline anyequal(x::AbstractArray{<:Real,1}, y::Real) = ( radius = (1e-10)*y; any(arg -> isapprox_DoME(arg,y,radius), x); )
@inline anyequal(x::AbstractArray{<:AbstractFloat,1}, y::AbstractFloat) = anyequal(x, y, DoME_ToleranceApprox*y);
@inline anyequal(x::AbstractFloat, y::AbstractArray{<:AbstractFloat,1}) = anyequal(y, x)
@inline function anyequal(x::AbstractArray{<:AbstractFloat,1}, y::AbstractArray{<:AbstractFloat,1})
    @inbounds for (i,j) in zip(x,y)
        isapprox_DoME(i,j) && return true;
    end;
#     @inbounds for i=1:length(x)
#         isapprox(x[i], y[i]) && return true
#     end
    return false
end



# equal(x,y; tolerance=NaN) = (x_y = abs.(x.-y); indNot0 = isa(x_y,Number) ? ((x_y!=0) ? 1 : []) : findall(x_y.!=0); if isnan(tolerance) tolerance = eltype(x)==Float32 ? 1e-3 : 1e-5; end; isempty(indNot0) ? true : all( ((x_y.<tolerance) .| (x_y.<abs.(x.*tolerance)) .| (x_y.<abs.(y.*tolerance)))[indNot0]) );
# @inline toleranceFromSemantic(x::Semantic) = eltype(x)==Float32 ? 1e-7 : 1e-10;
# @inline function equal(x::Real, y::Real; tolerance=NaN)
#     x == y && return true;
#     iszero(tolerance) && return false;
# 	if iszero(x) || iszero(y)
# 		if isnan(tolerance)
# 			tolerance = toleranceFromSemantic(x);
# 		end;
# 		iszero(x) && return abs(y)<tolerance;
# 		return abs(x)<tolerance;
# 	end;
# 	x_y = abs.(x.-y);
# 	if isnan(tolerance)
# 		tolerance = toleranceFromSemantic(x);
# 	end;
# 	return (x_y<abs(x*tolerance)) || (x_y<abs(y*tolerance));
# end;

# @inline cleanVector(v::AbstractFloat) = isapprox0(v) ? zero(v) : v;
@inline cleanVector(v::AbstractFloat) = isapprox0(v) ? zero(eltype(v)) : v;
# @inline function cleanVector(v::AbstractArray{<:Real,1})
#     # all0(v) && return zero(eltype(v));
#     index0 = isapprox0.(v);
#     zero_ = zero(eltype(v));
#     all(index0) && return zero_;
#     v[index0] .= zero_;
#     m = valueAllEqual(v);
#     return isnan(m) ? v : m;
# end;
# @inline function cleanVector(v::AbstractArray{<:Real,1})
#     n = length(v)
#     zero_ = zero(eltype(v));
#     mean_val = zero_;
#     allApprox0 = true;
#     allApproxValues = true;
#     @inbounds for (i, elem) in enumerate(v)
#         if isapprox0(elem)
#             @inbounds v[i] = zero_;
#             elem = zero_;
#             if allApproxValues
#                 if !isapprox0(mean_val)
#                     allApproxValues = false;
#                 else
#                     mean_val -= mean_val / i;
#                 end;
#             end;
#         else
#             if allApprox0 allApprox0 = false; end;
#             if allApproxValues
#                 mean_val += (elem - mean_val) / i
#                 # if i > 1 && !isapprox_DoME(elem, mean_val)
#                 if !isapprox_DoME(elem, mean_val)
#                     allApproxValues = false;
#                 end
#             end;
#         end;
#     end
#     return allApprox0 ? zero_ : allApproxValues ? mean_val : v;
# end
# @inline cleanVector(v::AbstractArray{<:Real,1}) = (allequal(v)) ? cleanVector(v[1]) : v
# @inline function cleanVector(v::AbstractArray{<:Real,1})
#     m = valueAllEqual(v);
#     # return isnan(m) ? v : m;
#     !isnan(m) && return cleanVector(m);
#     # v[isapprox0.(v)] .= 0;
#     return v;
# end;
# @inline function cleanVector(v::AbstractArray{<:Real,1})
#     m,M = extrema(v);
#     isapprox0(m) && isapprox0(M) && return zero(m)
#     isapprox_DoME(m,M) && return mean(v);
#     # isapprox_DoME(m,M) && return (m+M)/2;
#     # if sign(m)!=sign(M)
#     #     v[isapprox0.(v)] .= 0;
#     # end;
#     return v;
# end;
@inline function cleanVector(v::AbstractArray{<:AbstractFloat,1})
    # if anyInf(v)
    #     !all(isinf, v) && return false;
    #     all(x -> x > 0) && return Inf;
    #     all(x -> x < 0) && return -Inf;
    #     return false;
    # end;
    # if anyNaN(x)
    #     return all(isnan, v);
    # end;

    
    m,M = extrema(v);
    # isnan(m) && isnan(M) && return all(isnan,v) ? v[1] : v;
    isnan(m) && isnan(M) && return all(isnan,v) ? eltype(v)(NaN) : v;
    isinf(m) && isinf(M) && return m==M ? m : v;
    isapprox0(m) && isapprox0(M) && return zero(m)
    # if isapprox0(m)
    #     isapprox0(M) && return zero(m)
    # else
    #     isapprox0(M) && return false
    # end;
    isapprox_DoME(m,M) && return mean(v);
    # isapprox_DoME(m,M) && return (m+M)/2;
    # if sign(m)!=sign(M)
    #     v[isapprox0.(v)] .= 0;
    # end;
    return v;
end;

#############################################################################################################
#
# Functions for the operators
#

# These equations make the same computations, but allow to specify a buffer to write the results
# This avoids memory allocations and thus are faster
@inline addSemantics(x, y, buffer::Array{<:AbstractFloat,1}) = (isa(x,Number) && isa(y,Number)) ? x+y : ( buffer .= x; buffer .+= y; cleanVector(buffer); )
@inline subSemantics(x, y, buffer::Array{<:AbstractFloat,1}) = (isa(x,Number) && isa(y,Number)) ? x-y : ( buffer .= x; buffer .-= y; cleanVector(buffer); )
@inline mulSemantics(x, y, buffer::Array{<:AbstractFloat,1}) = (isa(x,Number) && isa(y,Number)) ? x*y : ( buffer .= x; buffer .*= y; cleanVector(buffer); )
@inline divSemantics(x, y, buffer::Array{<:AbstractFloat,1}) = (isa(x,Number) && isa(y,Number)) ? x/y : ( buffer .= x; buffer ./= y; cleanVector(buffer); )
@inline addSemantics(x, y, buffer::Nothing) = addSemantics(x, y);
@inline subSemantics(x, y, buffer::Nothing) = subSemantics(x, y);
@inline mulSemantics(x, y, buffer::Nothing) = mulSemantics(x, y);
@inline divSemantics(x, y, buffer::Nothing) = divSemantics(x, y);
@inline addSemantics(x, y, buffer::AbstractFloat) = addSemantics(x, y);
@inline subSemantics(x, y, buffer::AbstractFloat) = subSemantics(x, y);
@inline mulSemantics(x, y, buffer::AbstractFloat) = mulSemantics(x, y);
@inline divSemantics(x, y, buffer::AbstractFloat) = divSemantics(x, y);
# @inline addSemantics(x, y, buffer::Nothing) = cleanVector(x.+y);
# @inline subSemantics(x, y, buffer::Nothing) = cleanVector(x.-y);
# @inline mulSemantics(x, y, buffer::Nothing) = cleanVector(x.*y);
# @inline divSemantics(x, y, buffer::Nothing) = cleanVector(x./y);
# @inline addSemantics(x, y, buffer::AbstractFloat) = cleanVector(x.+y);
# @inline subSemantics(x, y, buffer::AbstractFloat) = cleanVector(x.-y);
# @inline mulSemantics(x, y, buffer::AbstractFloat) = cleanVector(x.*y);
# @inline divSemantics(x, y, buffer::AbstractFloat) = cleanVector(x./y);
@inline addSemantics(x, y) = cleanVector(x.+y);
@inline subSemantics(x, y) = cleanVector(x.-y);
@inline mulSemantics(x, y) = cleanVector(x.*y);
@inline divSemantics(x, y) = cleanVector(x./y);


# @inline addSemantics(x::AbstractFloat,            y::AbstractFloat,            buffer::Array{<:AbstractFloat,1}) = x+y;
# @inline addSemantics(x::Array{<:AbstractFloat,1}, y::AbstractFloat,            buffer::Array{<:AbstractFloat,1}) = ( buffer .= y; buffer .+= x; cleanVector(buffer); )
# @inline addSemantics(x::AbstractFloat,            y::Array{<:AbstractFloat,1}, buffer::Array{<:AbstractFloat,1}) = ( buffer .= x; buffer .+= y; cleanVector(buffer); )
# @inline addSemantics(x::Array{<:AbstractFloat,1}, y::Array{<:AbstractFloat,1}, buffer::Array{<:AbstractFloat,1}) = ( buffer .= x; buffer .+= y; cleanVector(buffer); )


# @inline addSemantics(x, y, buffer) = cleanVector(x.+y);
# @inline subSemantics(x, y, buffer) = cleanVector(x.-y);
# @inline mulSemantics(x, y, buffer) = cleanVector(x.*y);
# @inline divSemantics(x, y, buffer) = cleanVector(x./y);
# @inline addSemantics(x, y) = cleanVector(x.+y);
# @inline subSemantics(x, y) = cleanVector(x.-y);
# @inline mulSemantics(x, y) = cleanVector(x.*y);
# @inline divSemantics(x, y) = cleanVector(x./y);

#############################################################################################################
#
# Functions for calculating the equation coefficients a,b,c,d, and S
#

isNaN(S::SemanticSet) = isa(S,Number) && isnan(S)

if DoME_UseSSet

    semanticSetOpSemantic(S::SemanticSet, f::Function, semantic::Semantic) = (isempty(S) || isNaN(S)) ? S : Semantic[cleanVector(f.(s,semantic)) for s in S]
    semanticOpSemanticSet(semantic::Semantic, f::Function, S::SemanticSet) = (isempty(S) || isNaN(S)) ? S : Semantic[cleanVector(f.(semantic,s)) for s in S]
    function semanticSetDivSemantic(S::SemanticSet, semantic::Semantic)
        (isempty(S) || isNaN(S)) && return S;
        # for s in S
        #     any0(s,semantic) && return NaN;
        # end;
        any(s -> any0(s,semantic), S) && return NaN;
        return Semantic[cleanVector(s./semantic) for s in S];
    end;
    function semanticDivSemanticSet(semantic::Semantic, S::SemanticSet)
        isempty(S) && return Semantic[zero(eltype(semantic))];
        isNaN(S) && return S;
        # for s in S
        #     any0(s,semantic) && return NaN;
        # end;
        any(s -> any0(s,semantic), S) && return NaN;
        newSemanticSet = Semantic[cleanVector(semantic./s) for s in S];
        for s in newSemanticSet
            s==0 && return newSemanticSet
        end;
        pushfirst!(newSemanticSet,zero(eltype(semantic)));
        return newSemanticSet
    end;

else

    semanticSetOpSemantic(S::SemanticSet, f::Function, semantic::Semantic) = S;
    semanticOpSemanticSet(semantic::Semantic, f::Function, S::SemanticSet) = S;
    semanticSetDivSemantic(S::SemanticSet, semantic::Semantic) = S;
    semanticDivSemanticSet(semantic::Semantic, S::SemanticSet) = S;

end;

@inline equationAddChild1(y, a, b, c, d, S) = NodeEquation(                  a , all1(a) ? b.-y : b.-(a.*y) ,                  c , all0(c) ?  d : d.-(c.*y) , semanticSetOpSemantic(S,-,y) );
@inline equationAddChild2(x, a, b, c, d, S) = NodeEquation(                  a , all1(a) ? b.-x : b.-(a.*x) ,                  c , all0(c) ?  d : d.-(c.*x) , semanticSetOpSemantic(S,-,x) );
@inline equationSubChild1(y, a, b, c, d, S) = NodeEquation(                  a , all1(a) ? b.+y : b.+(a.*y) ,                  c , all0(c) ?  d : d.+(c.*y) , semanticSetOpSemantic(S,+,y) );
@inline equationSubChild2(x, a, b, c, d, S) = NodeEquation(                  a , all1(a) ? x.-b : (a.*x).-b ,                  c , all0(c) ? -d : (c.*x).-d , semanticOpSemanticSet(x,-,S) );
@inline equationMulChild1(y, a, b, c, d, S) = NodeEquation( all1(a) ? y : a.*y ,                          b , all0(c) ? c : c.*y ,                        d , semanticSetDivSemantic(S,y) );
@inline equationMulChild2(x, a, b, c, d, S) = NodeEquation( all1(a) ? x : a.*x ,                          b , all0(c) ? c : c.*x ,                        d , semanticSetDivSemantic(S,x) );
@inline equationDivChild1(y, a, b, c, d, S) = NodeEquation(                  a ,                       b.*y ,                  c , all0(d) ?  d :      d.*y , semanticSetOpSemantic(S,*,y) );
@inline equationDivChild2(x, a, b, c, d, S) = NodeEquation(                  b , all1(a) ?    x :      a.*x ,                  d , all0(c) ?  c :      c.*x , semanticDivSemanticSet(x,S) );

@inline equationAddChild1(y, eq::NodeEquation) = equationAddChild1(y, eq.a, eq.b, eq.c, eq.d, eq.S)
@inline equationAddChild2(x, eq::NodeEquation) = equationAddChild2(x, eq.a, eq.b, eq.c, eq.d, eq.S)
@inline equationSubChild1(y, eq::NodeEquation) = equationSubChild1(y, eq.a, eq.b, eq.c, eq.d, eq.S)
@inline equationSubChild2(x, eq::NodeEquation) = equationSubChild2(x, eq.a, eq.b, eq.c, eq.d, eq.S)
@inline equationMulChild1(y, eq::NodeEquation) = equationMulChild1(y, eq.a, eq.b, eq.c, eq.d, eq.S)
@inline equationMulChild2(x, eq::NodeEquation) = equationMulChild2(x, eq.a, eq.b, eq.c, eq.d, eq.S)
@inline equationDivChild1(y, eq::NodeEquation) = equationDivChild1(y, eq.a, eq.b, eq.c, eq.d, eq.S)
@inline equationDivChild2(x, eq::NodeEquation) = equationDivChild2(x, eq.a, eq.b, eq.c, eq.d, eq.S)


# function semanticNotInDomain(semantic::Semantic, S::SemanticSet)
#     isempty(S) && return false;
#     isNaN(S) && return true;
#     for s in S
#         # anynan(s) && return false;
#         anyequal(s,semantic) && return true;
#     end;
#     return false;
# end;
@inline semanticNotInDomain(semantic::Semantic, S::SemanticSet) = isempty(S) ? false : isNaN(S) ? true : any(x -> anyequal(x,semantic), S);
# function semanticNotInDomain(semantic::Real, S::SemanticSet)
#     isempty(S) && return false;
#     isNaN(S) && return true;
#     radius = (1e-10)*semantic
#     for s in S
#         # anynan(s) && return false;
#         anyequal(s,semantic,radius) && return true;
#     end;
#     return false;
# end;
@inline semanticNotInDomain(semantic::Semantic, eq::NodeEquation) = semanticNotInDomain(semantic, eq.S)


function showEquation(a::Semantic, b::Semantic, c::Semantic, d::Semantic, S::SemanticSet, semantic::Union{Semantic,Nothing}, targets::Union{Semantic,Nothing})
    println("a= ", eltype(a), ".(", a, ");");
    println("b= ", eltype(b), ".(", b, ");");
    println("c= ", eltype(c), ".(", c, ");");
    println("d= ", eltype(d), ".(", d, ");");
    # println("S= ", isempty(eq.S) ? "∅" : eq.S , ";");
    if isempty(S)
        println("S= ", isempty(S) ? "∅" : S , ";");
    elseif isa(S, Real) && isnan(S)
        println("S= NaN;");
    else
        text = string(S[1]);
        for i = 2:length(S)
            text = string(text, ", ", string(S[i]));
        end;

        println("S= {", text , "}");
    end;
    !isnothing(semantic) && println("semantic= ", semantic, ";");
    !isnothing(targets)  && println("t= ", targets, ";");
end;
showEquation(a::Semantic, b::Semantic, c::Semantic, d::Semantic) = showEquation(a, b, c, d, Semantic[], nothing, nothing)
showEquation(eq::NodeEquation, semantic::Union{Semantic,Nothing}) = showEquation(eq.a, eq.b, eq.c, eq.d, eq.S, semantic, nothing)
showEquation(eq::NodeEquation)                                    = showEquation(eq.a, eq.b, eq.c, eq.d, eq.S, nothing, nothing)



function checkEquation(eq::NodeEquation)
    @assert(isa(eq.a,Vector) || isa(eq.b,Vector));
    @assert(eltype(eq.a)==eltype(eq.b)==eltype(eq.c)==eltype(eq.d));
    # checkVector(v) = (isa(v,Number) || !allequal(v)) && !anyNaN(v) && !anyInf(v);
    checkVector(v) = !anyNaN(v) && !anyInf(v);
    @assert(checkVector(eq.a));
    @assert(checkVector(eq.b));
    @assert(checkVector(eq.c));
    @assert(checkVector(eq.d));
    if (isempty(eq.S))
        DoME_UseSSet && @assert(all0(eq.c))
    else
        if (isa(eq.S,Number))
            @assert(isnan(eq.S));
        else
            @assert(isa(eq.S,Array));
            # @assert(sum(eq.S .== 0)<=1);
            for s in eq.S
                @assert(!anyNaN(s));
            end;
            poles = eq.S[end];
            if (isa(poles,Number) && isinf(poles))
                @assert(all0(eq.c));
            elseif (isa(poles,Number) && (poles==0))
                @assert(all0(eq.d));
            else
                if anyInf(poles)
                    poles = poles[.!isinf.(poles)]
                end;
                if !isempty(poles)
                    poles = cleanVector(poles);
                end;
                @assert(!anyNaN(poles));

                # @assert(!any0(eq.c,eq.d));
                poles2 = eq.d./eq.c;

                if isa(poles2,Vector)
                    if anyInf(poles2)
                        poles2 = poles2[.!isinf.(poles2)]
                    end;
                    if !isempty(poles2)
                        poles2 = cleanVector(poles2);
                    end;
                end;
                @assert(!anyNaN(poles2));
                # # @assert(equal(poles,poles2));
                # if maximum(abs.(poles))>1e-8 && maximum(abs.(poles2))>1e-8
                #     # @assert( (all0(poles) && all0(poles2)) || all(isapprox_DoME.(poles,poles2)) );
                #     @assert( (all0(poles) && all0(poles2)) || mean(isapprox_DoME.(poles,poles2))>0.85 );
                # end;
            end;
        end;
    end;
end;



function horizontalAsymptote(a,b,c,d,S; checkForErrors=false)
    if (isa(c,Number))
        checkForErrors && @assert(c!=0)
        return mean(a.^2)/(c^2);
    elseif (isa(a,Number))
        if (a==0)
            asymptote = b./d;
            if (isa(asymptote,Number))
                return (asymptote^2)*mean(c.==0)
            else
                asymptote[c.!=0] .= 0;
                return mean(asymptote.^2);
            end;
        else
            any0(c) && return NaN;
            return mean((a./c).^2);
        end;
    else
        # a and c are vectors
        asymptote = a./c;
        for i in 1:length(c)
            if (c[i]==0)
                (a[i]!=0) && return NaN
                asymptote[i] = (isa(b,Vector) ? b[i] : b) / (isa(d,Vector) ? d[i] : d);
            else
                if (a[i]==0)
                    asymptote[i] = 0;
                end;
            end;
        end;
    end;
    if (checkForErrors)
        @assert(!anyNaN(asymptote))
        @assert(!anyInf(asymptote))
    end;
    return mean(asymptote.^2);
end;
horizontalAsymptote(eq::NodeEquation; checkForErrors=false) = horizontalAsymptote(eq.a,eq.b,eq.c,eq.d,eq.S; checkForErrors=checkForErrors);


#############################################################################################################
#
# Function for calculating the MSE from the equation values and the semantics of this node
#

function calculateMSEFromEquation(semantic::Semantic, a::Semantic, b::Semantic, c::Semantic, d::Semantic, S::SemanticSet; checkForErrors=false)

    isa(semantic, Number) && !isfinite(semantic) && return Inf;

    # calculateMSEFromEquationVectorial(semantic::Semantic,a::Semantic,b::Semantic,c::Semantic,d::Semantic) = mean(( ((semantic.*a).-b)./((semantic.*c).-d) ).^2);
    calculateMSEFromEquationVectorial() = mean(( ((semantic.*a).-b)./((semantic.*c).-d) ).^2);

    # function calculateMSEFromEquationVectorial()
    #     numerator = ((semantic.*a).-b);
    #     !all(isfinite, numerator) && return Inf;
    #     denominator = ((semantic.*c).-d);
    #     !all(isfinite, denominator) && return Inf;
    #     return mean(( numerator./denominator ).^2);
    # end;

    # function calculateMSEFromEquationEfficient(a,b,c,d)
    function calculateMSEFromEquationParticularCases()
        if all0(c)
            numerator = (semantic.*a).-b;
            if any0(d)
                any0(numerator,d) && return NaN;
                return Inf;
            end;
            if (isa(d,Number))
                 # d is a constant
                if (checkForErrors) @assert(d!=0); end;
                result = mean(numerator.^2)/(d^2);
                # result = mean((mulFunction(semantic,a) .- b).^2) ./ (d^2)
            else
                # d is a vector
                result = mean((numerator./d).^2);
                # result = mean(((mulFunction(semantic,a).-b)./d).^2);
            end;
        elseif all1(c) && all0(d)
            if any0(semantic)
                any0(semantic,b) && return NaN;
                return Inf;
            end;
            # result = mean(((a.*semantic .- b)./semantic).^2);
            result = mean((a .- (b./semantic)).^2);
        elseif all0(d)
            if any0(semantic)
                any0(semantic,b) && return NaN;
                return Inf;
            end;
            any0(c) && return Inf;
            result = mean(((a.*semantic .- b)./(c.*semantic)).^2);
        else
            numerator = (semantic.*a).-b;
            denominator = (semantic.*c).-d;
            # any0(numerator,denominator) && return NaN;
            # any0(denominator) && return Inf;
            result = mean( (numerator./denominator).^2 );
        end;
        return result;
    end;


    function calculateMSEFromEquationParticularCasesLoops()
        if all0(c)
            (any0(d)) && return NaN;
            if isa(d,Number)
                # d is a constant

                # return mean(((semantic.*a).-b).^2)/(d^2);

                n = max(length(a),length(b)); result = zero(eltype(a));
                @inbounds for i in 1:n
                    result += (valueSemantic(semantic,i)*valueSemantic(a,i) - valueSemantic(b,i))^2;
                end;
                result /= n*(d^2);

                if (checkForErrors) @assert(isapprox_DoME(result, mean(((semantic.*a).-b).^2)/(d^2))); end;
                return result;

                # result = mean((mulFunction(semantic,a) .- b).^2) ./ (d^2)
            else
                # d is a vector
                # return mean((((semantic.*a).-b)./d).^2);

                n = length(b); result = eltype(a)(0.);
                @inbounds for i in 1:n
                    result += ((valueSemantic(semantic,i)*valueSemantic(a,i) - valueSemantic(b,i)) / valueSemantic(d,i)) ^2;
                end;
                result /= n;

                if (checkForErrors) @assert(isapprox_DoME(result, mean((((semantic.*a).-b)./d).^2))); end;
                return result;

                # result = mean(((mulFunction(semantic,a).-b)./d).^2);
            end;
        elseif all1(c) && all0(d)
            # any0(semantic) && return NaN;
            # result = mean(((a.*semantic .- b)./semantic).^2);
            # return mean((a .- (b./semantic)).^2);

            n = max(length(a),length(b)); result = eltype(a)(0.);
            @inbounds for i in 1:n
                semantic_i = valueSemantic(semantic,i);
                (semantic_i==0) && return NaN;
                result += (valueSemantic(a,i) - valueSemantic(b,i)/semantic_i) ^2;
            end;
            result /= n;

            if (checkForErrors) @assert(isapprox_DoME(result, mean((a .- (b./semantic)).^2))); end;
            return result;

        elseif all0(d)
            # any0(semantic) && return NaN;
            # any0(c) && return NaN;
            # return mean((((a.*semantic) .- b)./(c.*semantic)).^2);

            n = max(length(a),length(b)); result = eltype(a)(0.);
            @inbounds for i in 1:n
                semantic_i = valueSemantic(semantic,i);
                (semantic_i==0) && return NaN;
                c_i = valueSemantic(c,i);
                (c_i==0) && return NaN;
                result += ((valueSemantic(a,i)*semantic_i - valueSemantic(b,i))/(c_i*semantic_i) ) ^2;
            end;
            result /= n;

            if (checkForErrors) @assert(isapprox_DoME(result, mean((((a.*semantic) .- b)./(c.*semantic)).^2))); end;
            return result;
        else
            # denominator = (semantic.*c).-d;
            # any0(denominator) && return NaN;
            # return mean( (((semantic.*a).-b)./denominator).^2 );

            n = max(length(a),length(b),length(c),length(d)); result = zero(eltype(a));
            @inbounds for i in 1:n
                denominator = (valueSemantic(c,i)*valueSemantic(semantic,i) - valueSemantic(d,i));
                (denominator==0) && return NaN;
                result += ((valueSemantic(a,i)*valueSemantic(semantic,i) - valueSemantic(b,i))/denominator)^2;
            end;
            result /= n;

            denominator = (semantic.*c).-d;
            any0(denominator) && return NaN;
            if (checkForErrors)
                denominator = (semantic.*c).-d;
                @assert(!any0(denominator))
                @assert(isapprox_DoME(result, mean( (((semantic.*a).-b)./denominator).^2 )));
            end;
            return result;
        end;
        return nothing;
    end;


    mse = calculateMSEFromEquationVectorial()
    # mse = calculateMSEFromEquationParticularCases();
    # mse = calculateMSEFromEquationParticularCasesLoops();

    # @assert(eltype(semantic)==eltype(a)==eltype(b)==eltype(c)==eltype(d)==eltype(constant)==eltype(mse))

    if checkForErrors
        mseV = calculateMSEFromEquationParticularCases();
        DoME_UseSSet && @assert(isnan(mse) == isnan(mseV));
        DoME_UseSSet && @assert(isinf(mse) == isinf(mseV));
        if isinf(mse) && isinf(mseV)
            @assert(sign(mse)==sign(mseV))
        elseif isfinite(mse) && isfinite(mseV)
            @assert(isapprox_DoME(mse,mseV));
        end;
        # mseE = calculateMSEFromEquationParticularCasesLoops();
        # @assert(!isinf(mseE));
        # DoME_UseSSet && @assert(isnan(mseE) == (isnan(mse) || isinf(mse)))
        # if isfinite(mseE) && isfinite(mseE)
        #     @assert(isapprox_DoME(mse,mseE));
        # end;
    end;

    return mse;
end;

calculateMSEFromEquation(semantic::Semantic, eq::NodeEquation; checkForErrors=false) = calculateMSEFromEquation(semantic, eq.a, eq.b, eq.c, eq.d, eq.S; checkForErrors=checkForErrors);

#############################################################################################################
#
# Function for calculating the derivative of the equation on a value x
#

function derivativeEquation(eq::NodeEquation, x::AbstractFloat)
    # return 2 .* mean( (b.*c .- a.*d) .* (a.*x .- b ) ./ ((c.*x .- d).^3) );
    a=eq.a; b=eq.b; c=eq.c; d=eq.d; S=eq.S;

    if anyDenominatorHas0Coefficients(c,d)
        return Inf;
    end;
    return 2 .* mean( (b.*c .- a.*d) .* (a.*x .- b ) ./ ((c.*x .- d).^3) );

    (a,b,c,d) = equation;
    bc_ad = (b.*c .- a.*d);
    result = bc_ad .* (a.*x .- b ) ./ ((c.*x .- d).^3);
    result[bc_ad.==0] .= 0.;
    return 2*mean(result);
end;

#############################################################################################################
#
# Function for calculating the reduction in MSE
#

# function calculateMSEReduction(semantic::Semantic, equation::NodeEquation, mse::Real; checkForErrors=false)
#     if (checkForErrors)
#         @assert(!all0(equation.c) || !all0(equation.d));
#         @assert(!anyNaN(semantic));
#         @assert(!anyInf(semantic));
#         @assert(!isnan(mse) && !isinf(mse));
#     end;
#     return mse - calculateMSEFromEquation(semantic, equation; checkForErrors=checkForErrors);
# end;


#############################################################################################################
#
# Functions for calculating the best constant for a node
#  Return the constant and the reduction in MSE
#

function calculateConstantMinimizeEquation(equation::NodeEquation; checkForErrors=false)

    function calculateRoots(x,y; checkForErrors=false) # x*k - y => k = y/x
        all0(x) && return Array{eltype(x),1}();
        roots = y./x;
        # if any0(x)
        #     # x must be a vector
        #     (checkForErrors) && @assert(isa(x,Vector));
        #     # roots=roots[x.!=0];
        #     roots=filter(x -> !isapprox0(x), roots);
        # end;
        # roots = unique(roots);

        roots[findall(isapprox0, roots)] .= 0;
        roots = unique(Iterators.filter(root -> isfinite(root), roots));

        # N = length(roots);
        # indices = trues(N);
        # @inbounds for (i,root) in enumerate(roots)
        #     !indices[i] && continue;
        #     if isapprox0(root) root=0; end;
        #     radius = DoME_ToleranceApprox*root;
        #     @inbounds for j in (i+1):N
        #         !indices[j] && continue;
        #         newRoot = roots[j];
        #         if isapprox0(newRoot) newRoot=roots[j]=0; end;
        #         isapprox_DoME(newRoot, root, radius) && (indices[j]=false);
        #     end;
        # end;
        # roots = roots[indices];

        # N = length(roots);
        # newRoots = Array{eltype(roots),1}()
        # @inbounds for root in roots
        #     if isapprox0(root) root = 0; end;
        #     if isfinite(root) && !anyequal(newRoots, root)
        #         push!(newRoots, root);
        #     end;
        # end;
        # roots = newRoots;

        if (checkForErrors)
            @assert(!anyNaN(roots));
            @assert(!anyInf(roots));
            @assert(isa(roots,Vector));
        end;
        return roots;
    end;



    function calculateConstantGeneralCase(a,b,c,d,S; checkForErrors=false)

        if (checkForErrors)
            DoME_UseSSet && @assert(!isempty(S))
            @assert(!isNaN(S))
            DoME_UseSSet && @assert(!anyNaN(S[end]))
        end;

        if !DoME_UseSSet
            zeros = b./a;
            # zeros = unique(zeros);
            yZeros = calculateMSEFromEquation.(zeros, [a],[b],[c],[d],[S]; checkForErrors=checkForErrors);
            # yZeros = [calculateMSEFromEquation(thisZero, a,b,c,d,S; checkForErrors=checkForErrors) for thisZero in zeros];
            return zeros, yZeros
        end;


        zeros = calculateRoots(a,b; checkForErrors=checkForErrors);
        # These are not zeros of the whole equation, but zeros in each term

        isempty(zeros) && return NaN, NaN;
        if DoME_UseSSet
            zeros = zeros[.!semanticNotInDomain.(zeros, [S])];
        end;

        if (checkForErrors)
            @assert(!anyNaN(zeros));
        end;

        isempty(zeros) && return NaN, NaN;


        # Avoid those zeros which are too close to a root of the denominator
        # The roots of the denominator should already be calculated in the last term of S
        if DoME_UseSSet
            poles = S[end];
        else
            poles = calculateRoots(c,d; checkForErrors=checkForErrors);
        end;

# poles = calculateRoots(c,d; checkForErrors=checkForErrors);
        if (isa(poles,Number) && isinf(poles))
            # In this case, any value is valid (there are no roots in the denominator)
            if (checkForErrors)
                @assert(all0(c));
                # @assert(isempty(cleanEquation(calculateRoots(c,d; checkForErrors=checkForErrors))));
                @assert(isempty(calculateRoots(c,d; checkForErrors=checkForErrors)));
            end;
        else

            # However, in order to avoid possible rounding errors, calculate the roots of the denominator instead of S[end]
            if DoME_UseSSet
                poles = calculateRoots(c,d; checkForErrors=checkForErrors);
            end;

            if checkForErrors
                @assert(!isempty(poles));
                @assert(!anyInf(poles));
                @assert(!anyNaN(poles));
            end;

            # poles = cleanVector(poles);

            function isNotClose(zero, poles, tolerance)
                for pole in poles
                    limit = abs(pole*tolerance);
                    # (zero>=pole-limit) && (zero<=pole+limit) && return false;
                    ((pole-limit)<=zero<=(pole+limit)) && return false;
                end;
                return true;
            end;
            # zeros = zeros[isNotClose.(zeros,[poles],[eltype(a)(1e-2)])];
            zeros = filter(zero -> isNotClose(zero,poles,eltype(zero)(1e-2)), zeros);
            isempty(zeros) && return NaN, NaN;

        end;

        # yZeros = calculateMSEFromEquation.(zeros, [a],[b],[c],[d],[S]; checkForErrors=checkForErrors);
        yZeros = [calculateMSEFromEquation(thisZero, a,b,c,d,S; checkForErrors=checkForErrors) for thisZero in zeros];

        if checkForErrors && DoME_UseSSet
            @assert(!anyNaN(yZeros));
            @assert(!anyInf(yZeros));
        end;

        if !DoME_UseSSet
            return zeros, yZeros
        else
            (mse,index) = findmin(yZeros);
            zero = zeros[index];
            return zero, mse;
        end;

    end;


    function calculateConstantFullVectors(a,b,c,d,S; checkForErrors=false)
        any0(c,d) && return NaN;
        lengthArray = maximum([length(a), length(b), length(c), length(d)]);
        # anyDenominatorHas0Coefficients(c,d) && return NaN;
        createVector(v::Vector) = v;
        createVector(x::Number) = ( v = Array{eltype(a),1}(undef,lengthArray); v.=x; return v; )
        a = createVector(a);
        b = createVector(b);
        c = createVector(c);
        d = createVector(d);

        if all0(c)
            any0(d) && return NaN;
            if (isa(a,Vector))
                index0 = (a.==0);
                num = (b.*a)./(d.^2);
                num[index0] .= eltype(a)(0.);
                den = ((a./d).^2);
                den[index0] .= eltype(a)(0.);
                k = sum(num)./sum(den);
            else
                k = sum((b.*a)./(d.^2))/sum((a./d).^2);
            end;
            semanticNotInDomain(k, S) && return NaN;
        elseif all0(d)
            any0(c) && return NaN;
            if (isa(b,Vector))
                index0 = (b.==0);
                num = ((b./c).^2);
                num[index0] .= eltype(a)(0.);
                den = ((a.*b)./(c.^2));
                den[index0] .= eltype(a)(0.);
                k = sum(num)/sum(den);
            else
                k = sum((b./c).^2) ./ sum(a.*b./(c.^2));
            end;
            semanticNotInDomain(k, S) && return NaN;
        elseif allequal(c) && allequal(d)
            mult = (b.*c .- a.*d);
            k = sum(mult.*b)/sum(mult.*a);
            semanticNotInDomain(k, S) && return NaN;
        else
            k = calculateConstantGeneralCase(a,b,c,d,S; checkForErrors=checkForErrors)[1];
        end;
        return isinf(k) ? NaN : k;
    end;
    calculateConstantFullVectors(eq::NodeEquation; checkForErrors=false) = calculateConstantFullVectors(eq.a,eq.b,eq.c,eq.d,eq.S; checkForErrors=checkForErrors);


    function calculateConstantLoops(a,b,c,d,S; checkForErrors=false)
        # Vectorial operations have been replaced by loops, so no memory allocation is needed and all the operations are much faster
        if all0(c)
        # if c==0

            all0(a) && return NaN;
            any0(d) && return NaN;
    # constant = mean((b.*a)./(d.^2))/mean((a.^2)./(d.^2));
    # return semanticNotInDomain(constant, S) ? NaN : constant;
            if allequal(a)
                # a is constant
                a = a[1];
                if (checkForErrors) @assert(isa(b,Vector)); end;
                # if isa(d,Number)
                if allequal(d)
                    # d is constant
                    d = d[1];
                    # constant = mean(b)/a;
                    constant = eltype(a)(0.);
                    @inbounds for i in 1:length(b)
                        constant += b[i];
                    end;
                    constant /= (length(b)*a);

                    if (checkForErrors) @assert(isapprox_DoME(constant,mean((b.*a)./(d.^2))/(mean((a./d).^2)))); end;
                else
                    # d is vector
                    # inv_d = (1. ./ d).^2; constant = sum(b.*inv_d)/ (a*sum(inv_d));

                    numerator = eltype(a)(0.); denominator = eltype(a)(0.);
                    @inbounds for i in 1:length(d)
                        v = (1/d[i])^2;
                        numerator += valueSemantic(b,i)*v;
                        denominator += v;
                    end;
                    constant = numerator / (a*denominator);

                    if (checkForErrors) @assert(isapprox_DoME(constant,sum((b.*a)./(d.^2))/sum((a./d).^2))); end;
                end;
            else
                # a is vector
                if allequal(d)
                    # d is constant
                    d = d[1];
                    # constant = sum(b.*a)/sum(a.^2);

                    numerator = eltype(a)(0.); denominator = eltype(a)(0.);
                    @inbounds for i in 1:length(a)
                        a_i = valueSemantic(a,i);
                        numerator += a_i*valueSemantic(b,i);
                        denominator += (a_i^2);
                    end;
                    constant = numerator / denominator;

                    if (checkForErrors) @assert(isapprox_DoME(constant,sum((b.*a)./(d.^2))/sum((a./d).^2))); end;
                else
                    # d is vector
                    # constant = sum((b.*a)./(d.^2))/sum((a./d).^2);

                    numerator = eltype(a)(0.); denominator = eltype(a)(0.);
                    @inbounds for i in 1:length(a)
                        a_i = valueSemantic(a,i);
                        d_i = valueSemantic(d,i);
                        numerator += a_i*valueSemantic(b,i)/(d_i^2);
                        denominator += (a_i/d_i)^2;
                    end;
                    constant = numerator / denominator;

                    if (checkForErrors) @assert(isapprox_DoME(constant,sum((b.*a)./(d.^2))/sum((a./d).^2))); end;

                end;
            end;
    # constant2 = mean((b.*a)./(d.^2))/mean((a.^2)./(d.^2));
    # @assert(equal(constant, constant2));

            semanticNotInDomain(constant, S) && return NaN;

        elseif all0(d)
            any0(c) && return NaN;
            if allequal(b)
                # b is constant
                b = b[1];
                if (checkForErrors) @assert(isa(a,Vector)); end;
                # if isa(c,Number)
                if allequal(c)
                    # c is constant
                    c = c[1];
                    # constant = b./mean(a);

                    constant = eltype(a)(0.);
                    @inbounds for i in 1:length(a)
                        constant += a[i];
                    end;
                    constant = length(a)*b/constant;
                    if (checkForErrors) @assert(isapprox_DoME(constant,length(a)*((b./c).^2) ./ sum(a.*b./(c.^2)))); end;
                else
                    # c is vector
                    # inv_c = (1. ./ c).^2;
                    # constant = b.*sum(inv_c)/sum(a.*inv_c);

                    numerator = eltype(a)(0.); denominator = eltype(a)(0.);
                    @inbounds for i in 1:length(c)
                        inv_c_i = (1/c[i])^2;
                        numerator += inv_c_i;
                        denominator += (valueSemantic(a,i)*inv_c_i);
                    end;
                    constant = b * numerator / denominator;

                    if (checkForErrors) @assert(isapprox_DoME(constant,sum((b./c).^2) ./ sum(a.*b./(c.^2)))); end;
                end;
                semanticNotInDomain(constant, S) && return NaN;
            else
                # b is vector
                if allequal(c)
                    # c is constant
                    c = c[1];
                    # constant = sum(b.^2)/sum(a.*b);

                    numerator = eltype(a)(0.); denominator = eltype(a)(0.);
                    @inbounds for i in 1:length(b)
                        b_i = b[i];
                        numerator += b_i^2;
                        denominator += (valueSemantic(a,i)*b_i);
                    end;
                    constant = numerator / denominator;

                    if (checkForErrors) @assert(isapprox_DoME(constant,sum((b./c).^2) ./ sum(a.*b./(c.^2)))); end;
                else
                    # c is vector
                    # constant = sum((b./c).^2) ./ sum(a.*b./(c.^2));

                    numerator = eltype(a)(0.); denominator = eltype(a)(0.);
                    @inbounds for i in 1:length(c)
                        b_i = valueSemantic(b,i);
                        numerator += (b_i/c[i])^2;
                        denominator += (valueSemantic(a,i)*b_i/(c[i]^2));
                    end;
                    constant = numerator / denominator;

                    if (checkForErrors) @assert(isapprox_DoME(constant,sum((b./c).^2) ./ sum(a.*b./(c.^2)))); end;
                end;
                semanticNotInDomain(constant, S) && return NaN;
            end;

        elseif allequal(c) && allequal(d)
            # constant = sum(b.*b.*c .- a.*b.*d)/sum(a.*b.*c .- a.*a.*d);

            # mult = (b.*c .- a.*d);
            # constant = sum(mult.*b)/sum(mult.*a);

            c = c[1]; d = d[1];
            numerator = eltype(a)(0.); denominator = eltype(a)(0.);
            @inbounds for i in 1:max(length(a),length(b))
                a_i = valueSemantic(a,i);
                b_i = valueSemantic(b,i);
                mult = b_i*c - a_i*d;
                numerator += b_i*mult;
                denominator += a_i*mult;
            end;
            constant = numerator / denominator;

            if (checkForErrors)
                mult = (b.*c .- a.*d);
                @assert(!all0(a));
                @assert(!all0(mult));
                @assert(isapprox_DoME(constant,sum(mult.*b)/sum(mult.*a)));
            end;

            semanticNotInDomain(constant, S) && return NaN;

        else
            any0(c,d) && return NaN;
            return calculateConstantGeneralCase(a,b,c,d,S; checkForErrors=checkForErrors)[1];
        end;

        if (checkForErrors)
            @assert(!isnan(constant));
            @assert(!isinf(constant));
        end;
        return constant;
    end;
    calculateConstantLoops(eq::NodeEquation; checkForErrors=false) = calculateConstantLoops(eq.a,eq.b,eq.c,eq.d,eq.S; checkForErrors=checkForErrors);




    function calculateConstantNoLoops()

        # if all0(c)
        if c==0

            # constant = mean((b.*a)./(d.^2))/mean((a.^2)./(d.^2));

            # all0(a) && return NaN;
            # any0(d) && return NaN;
            # kd = valueAllEqual(d);
            # if allequal(kd) # d=kd ?
            if isa(d, Number)
                # constant = mean(a.*b)/mean(a.^2);
                # mse = (mean(a.^2)*mean(b.^2) - mean(a.*b)^2)/(d^2*mean(a.^2))
                mean_a2 = mean(a.^2);
                mean_ab = mean(a.*b);
                constant = mean_ab/mean_a2;
                mse = (mean_a2*mean(b.^2) - mean_ab^2)/(d^2*mean_a2)
            else # d!=kd
                # constant = mean((b.*a)./(d.^2))/mean((a.^2)./(d.^2));
                # mse = mean((b.^2)./(d.^2)) - mean((a.*b)./(d.^2))^2/mean((a.^2)./(d.^2))
                mean_ab_d2 = mean((a.*b)./(d.^2))
                mean_a2_d2 = mean((a.^2)./(d.^2))
                constant = mean_ab_d2/mean_a2_d2
                mse = mean((b.^2)./(d.^2)) - mean_ab_d2^2/mean_a2_d2
            end;

            semanticNotInDomain(constant, S) && return NaN, NaN;

        # elseif all0(d)
        elseif d==0

            # any0(c) && return NaN;
            # if allequal(c) # c=kc ?
            if isa(c, Number) # c=kc ?
                # constant = mean(b.^2)/mean(a.*b)
                # mse = (mean(a.^2)*mean(b.^2) + mean(a.*b)^2)/((c^2)*mean(b.^2))
                mean_b2 = mean(b.^2);
                mean_ab = mean(a.*b);
                constant = mean_b2/mean(a.*b)
                mse = (mean(a.^2)*mean_b2 + mean_ab^2)/((c^2)*mean_b2)
            else
                # constant = mean((b.^2)./(c.^2))/mean((a.*b)./(c.^2))
                # mse = mean((a.^2)./(c.^2)) - (mean((a.*b)./(c.^2))^2)/mean((b.^2)./(c.^2))
                mean_b2_c2 = mean((b.^2)./(c.^2));
                mean_ab_c2 = mean((a.*b)./(c.^2));
                constant = mean_b2_c2/mean_ab_c2
                mse = mean((a.^2)./(c.^2)) - mean_ab_c2/mean_b2_c2
            end;

            semanticNotInDomain(constant, S) && return NaN, NaN;

        # elseif allequal(c) && allequal(d)
        elseif isa(c, Number) && isa(d, Number)
            # constant = sum(b.*b.*c .- a.*b.*d)/sum(a.*b.*c .- a.*a.*d);

            # mult = (b.*c .- a.*d);
            # constant = sum(mult.*b)/sum(mult.*a);

            # constant = (c*mean(b.^2) - d*mean(a.*b))/(c*mean(a.*b) - d*mean(a.^2))
            # mse = (mean(a.^2)*mean(b.^2) - mean(a.*b).^2 )/((c^2)*mean(b.^2) + (d^2)*mean(a.^2) - 2*c*d*mean(a.*b))
            mean_a2 = mean(a.^2);
            mean_b2 = mean(b.^2);
            mean_ab = mean(a.*b);
            constant = (c*mean_b2 - d*mean_ab)/(c*mean_ab - d*mean_a2)
            mse = (mean_a2*mean_b2 - mean_ab.^2 )/((c^2)*mean_b2 + (d^2)*mean_a2 - 2*c*d*mean_ab)

            if (checkForErrors)
                mult = (b.*c .- a.*d);
                @assert(!all0(a));
                @assert(!all0(mult));
                @assert(isapprox_DoME(constant,sum(mult.*b)/sum(mult.*a)));
            end;

            semanticNotInDomain(constant, S) && return NaN, NaN;

        else
            any0(c,d) && return NaN, NaN;
            return calculateConstantGeneralCase(a,b,c,d,S; checkForErrors=checkForErrors)[1]
        end;

        if (checkForErrors)
            # @assert(!isnan(constant));
            @assert(!isinf(constant));
        end;
        return constant, mse;
    end;



    function calculateConstantFunctions()

        @inline mean_xy(x::AbstractFloat,                    y::AbstractFloat                   ) = x*y;
        @inline mean_xy(x::AbstractArray{<:AbstractFloat,1}, y::AbstractFloat                   ) = y*mean(x);
        @inline mean_xy(x::AbstractFloat,                    y::AbstractArray{<:AbstractFloat,1}) = x*mean(y);
        # @inline mean_xy(x::AbstractArray{<:Real,1}, y::AbstractArray{<:Real,1}) = dot(x,y)/length(x);
        @inline mean_xy(x::AbstractArray{<:AbstractFloat,1}, y::AbstractArray{<:AbstractFloat,1}) = ( result = zero(eltype(x)); @inbounds for (i,j) in zip(x,y) result += i*j; end; return result/length(x); );
        # @inline mean_xy(x::AbstractArray{<:Real,1}, y::AbstractArray{<:Real,1}) = mean(x.*y);

        @inline mean_x2(x::AbstractFloat                   ) = x*x;
        # @inline mean_x2(x::AbstractArray{<:Real,1}) = dot(a,a)/length(a);
        @inline mean_x2(x::AbstractArray{<:AbstractFloat,1}) = ( result = zero(eltype(x)); @inbounds for i in x result += i*i; end; return result/length(x); );

        @inline mean_1_x2(x::AbstractArray{<:AbstractFloat,1}) = ( result = zero(eltype(x)); @inbounds for i in x result += 1/(i*i) end; return result/length(x); );

        @inline mean_x2_y2(x::AbstractFloat,                    y::AbstractFloat                   ) = (x*x)/(y*y);
        @inline mean_x2_y2(x::AbstractFloat,                    y::AbstractArray{<:AbstractFloat,1}) = (x*x)*mean_1_x2(y);
        @inline mean_x2_y2(x::AbstractArray{<:AbstractFloat,1}, y::AbstractFloat                   ) = mean_x2(x)/(y*y);
        # @inline mean_x2_y2(x::AbstractArray{<:Real,1}, y::AbstractArray{<:Real,1}) = mean((x.*x)./(y.*y));
        @inline mean_x2_y2(x::AbstractArray{<:AbstractFloat,1}, y::AbstractArray{<:AbstractFloat,1}) = ( result = zero(eltype(x)); for (i,j) in zip(x,y) result += (i*i)/(j*j); end; return result/length(x); );
        # function mean_x2_y2(x::AbstractArray{<:Real,1}, y::AbstractArray{<:Real,1}) = ( result = zero(eltype(x)); @inbounds for i in eachindex(x); result += ((x[i])^2)/((y[i])^2); end; return result/length(x); end; );

        @inline mean_x_y2(x::AbstractArray{<:AbstractFloat,1}, y::AbstractArray{<:AbstractFloat,1}) = ( result = zero(eltype(x)); for (i,j) in zip(x,y) result += i/(j*j); end; return result/length(x); );


        @inline mean_xy_z2(x::AbstractFloat,                    y::AbstractFloat,                    z::AbstractFloat                   ) = (x*y)/(z*z);
        @inline mean_xy_z2(x::AbstractFloat,                    y::AbstractFloat,                    z::AbstractArray{<:AbstractFloat,1}) = (x*y)*mean_1_x2(z);
        @inline mean_xy_z2(x::AbstractFloat,                    y::AbstractArray{<:AbstractFloat,1}, z::AbstractFloat                   ) = (x*mean(y))/(z*z);
        @inline mean_xy_z2(x::AbstractFloat,                    y::AbstractArray{<:AbstractFloat,1}, z::AbstractArray{<:AbstractFloat,1}) = x*mean_x_y2(y,z);
        @inline mean_xy_z2(x::AbstractArray{<:AbstractFloat,1}, y::AbstractFloat,                    z::AbstractFloat                   ) = (y*mean(x))/(z*z);
        @inline mean_xy_z2(x::AbstractArray{<:AbstractFloat,1}, y::AbstractFloat,                    z::AbstractArray{<:AbstractFloat,1}) = y*mean_x_y2(x,z);
        @inline mean_xy_z2(x::AbstractArray{<:AbstractFloat,1}, y::AbstractArray{<:AbstractFloat,1}, z::AbstractFloat                   ) = mean_xy(x,y)/(z*z);
        # @inline mean_xy_z2(x::AbstractArray{<:Real,1}, y::AbstractArray{<:Real,1}, z::AbstractArray{<:Real,1}) = mean((x.*y)./(z.*z));
        @inline mean_xy_z2(x::AbstractArray{<:AbstractFloat,1}, y::AbstractArray{<:AbstractFloat,1}, z::AbstractArray{<:AbstractFloat,1}) = ( result = zero(eltype(x)); for (i,j,k) in zip(x,y,z) result += (i*j)/(k*k); end; return result/length(x); );

        # if all0(c)
        if c==0

            # all0(a) && return NaN, NaN;
            # any0(d) && return NaN, NaN;

            # # constant = mean((b.*a)./(d.^2))/mean((a.^2)./(d.^2));
            # # mse = mean((b.^2)./(d.^2)) - mean((a.*b)./(d.^2))^2/mean((a.^2)./(d.^2))
            # mean_ab_d2 = mean_xy_z2(a,b,d)
            # mean_a2_d2 = mean_x2_y2(a,d)
            # constant = mean_ab_d2/mean_a2_d2
            # mse = mean_x2_y2(b,d) - mean_ab_d2^2/mean_a2_d2

            if isa(d, Number)
                # constant = mean(a.*b)/mean(a.^2);
                # mse = (mean(a.^2)*mean(b.^2) - mean(a.*b)^2)/(d^2*mean(a.^2))
                mean_ab = mean_xy(a,b);
                mean_a2 = mean_x2(a);
                k = mean_ab/mean_a2;
                mse = (mean_a2*mean_x2(b) - mean_ab^2)/((d^2)*mean_a2);
            else # d!=kd
                # constant = mean((b.*a)./(d.^2))/mean((a.^2)./(d.^2));
                # mse = mean((b.^2)./(d.^2)) - mean((a.*b)./(d.^2))^2/mean((a.^2)./(d.^2))
                mean_ab_d2 = mean_xy_z2(a,b,d)
                mean_a2_d2 = mean_x2_y2(a,d)
                k = mean_ab_d2/mean_a2_d2
                mse = mean_x2_y2(b,d) - mean_ab_d2^2/mean_a2_d2
            end;

            semanticNotInDomain(k, S) && return NaN, NaN;

        # elseif all0(d)
        elseif d==0

            # any0(c) && return NaN, NaN;

            # # constant = mean((b.^2)./(c.^2))/mean((a.*b)./(c.^2))
            # # mse = mean((a.^2)./(c.^2)) - (mean((a.*b)./(c.^2))^2)/mean((b.^2)./(c.^2))
            # mean_b2_c2 = mean_x2_y2(b,c);
            # mean_ab_c2 = mean_xy_z2(a,b,c);
            # constant = mean_b2_c2/mean_ab_c2
            # mse = mean_x2_y2(a,c) - mean_ab_c2/mean_b2_c2

            if isa(c, Number) # c=kc ?
                # constant = mean(b.^2)/mean(a.*b)
                # mse = (mean(a.^2)*mean(b.^2) - mean(a.*b)^2)/((c^2)*mean(b.^2))
                mean_b2 = mean_x2(b);
                mean_ab = mean_xy(a,b);
                k = mean_b2/mean_ab;
                mse = (mean_x2(a)*mean_b2 - mean_ab^2)/((c^2)*mean_b2)
            else
                # constant = mean((b.^2)./(c.^2))/mean((a.*b)./(c.^2))
                # mse = mean((a.^2)./(c.^2)) - (mean((a.*b)./(c.^2))^2)/mean((b.^2)./(c.^2))
                mean_b2_c2 = mean_x2_y2(b,c);
                mean_ab_c2 = mean_xy_z2(a,b,c);
                k = mean_b2_c2/mean_ab_c2
                mse = mean_x2_y2(a,c) - (mean_ab_c2^2)/mean_b2_c2
            end;

            semanticNotInDomain(k, S) && return NaN, NaN;

        elseif isa(c, Number) && isa(d, Number)
            # constant = sum(b.*b.*c .- a.*b.*d)/sum(a.*b.*c .- a.*a.*d);

            # mult = (b.*c .- a.*d);
            # constant = sum(mult.*b)/sum(mult.*a);

            # constant = (c*mean(b.^2) - d*mean(a.*b))/(c*mean(a.*b) - d*mean(a.^2))
            # mse = (mean(a.^2)*mean(b.^2) - mean(a.*b).^2 )/((c^2)*mean(b.^2) + (d^2)*mean(a.^2) - 2*c*d*mean(a.*b))
            mean_a2 = mean_x2(a);
            mean_b2 = mean_x2(b);
            mean_ab = mean_xy(a,b);
            k = (c*mean_b2 - d*mean_ab)/(c*mean_ab - d*mean_a2)
            mse = (mean_a2*mean_b2 - mean_ab.^2 )/((c^2)*mean_b2 + (d^2)*mean_a2 - 2*c*d*mean_ab)

            if (checkForErrors)
                mult = (b.*c .- a.*d);
                @assert(!all0(a));
                # @assert(!all0(mult));
                !isapprox0(mean(a.*mult)) && !isapprox0(mean(b.*mult)) && @assert(isapprox_DoME(k,sum(mult.*b)/sum(mult.*a)));
            end;

            semanticNotInDomain(k, S) && return NaN, NaN;

        else
            any0(c,d) && return NaN, NaN;
            return calculateConstantGeneralCase(a,b,c,d,S; checkForErrors=checkForErrors)
        end;

        return isinf(k) ? (NaN, NaN) : (k, mse);
    end;



    a=equation.a; b=equation.b; c=equation.c; d=equation.d; S=equation.S;
    newMSE = NaN;
    # constant = calculateConstantFullVectors(equation; checkForErrors=checkForErrors);
    # constant = calculateConstantLoops(equation; checkForErrors=checkForErrors);
    # constant = calculateConstantNoLoops(); if isa(constant,Tuple) constant, newMSE = constant; end;
    constant, newMSE = calculateConstantFunctions();
# @assert(eltype(a)==eltype(b)==eltype(c)==eltype(d)==eltype(constant)==eltype(newMSE))
    # if (checkForErrors)
    #     isa(constant,Number) && @assert(!isinf(constant));
    #     newConstant = calculateConstantFullVectors(equation; checkForErrors=checkForErrors);
    #     @assert(all(isnan.(constant)==isnan.(newConstant)));
        # if isa(constant,Number)
        #     !isnan(constant) && @assert((isapprox0(constant) && isapprox0(newConstant)) || isapprox_DoME(constant,newConstant));
        # else
        #     iNotNaN = .!isnan.(constant)
        #     @assert(all(isapprox_DoME.(constant[iNotNaN],newConstant[iNotNaN])))
        # end;
    # end;
    isa(constant,Number) && !isfinite(constant) && return NaN, Inf
    if isa(constant,Number)
        constant = isapprox0(constant) ? zero(constant) : constant;
    end;
    if isa(newMSE, Number) && isnan(newMSE)
        return constant, calculateMSEFromEquation(constant, equation; checkForErrors=checkForErrors);
    else
        checkForErrors && @assert(length(constant)==length(newMSE));
        checkForErrors && @assert(all([isapprox_DoME(thisNewMSE, calculateMSEFromEquation(thisConstant, equation; checkForErrors=checkForErrors)) for (thisConstant, thisNewMSE) in zip(constant, newMSE)]))
        # checkForErrors && @assert(isapprox_DoME(newMSE, calculateMSEFromEquation(constant, equation; checkForErrors=checkForErrors)));
    end;

    # @assert(isapprox(newMSE,calculateMSEFromEquation(constant, equation; checkForErrors=checkForErrors)));

    # reduction = calculateMSEReduction(constant, equation, mse; checkForErrors=checkForErrors);
    # reduction = isnan(reduction) ? -Inf : reduction;
    # reduction = isnan(newMSE) ? -Inf : (mse - newMSE);
    return (constant, newMSE);
end;

function calculateConstantMinimizeEquation(equation::NodeEquation, mse::AbstractFloat; checkForErrors=false)
    checkForErrors && @assert(isfinite(mse));
    constant, newMSE = calculateConstantMinimizeEquation(equation; checkForErrors=checkForErrors)
    reduction = isa(constant,Number) && isnan(constant) ? -Inf : (mse .- newMSE);
    return (constant, reduction);
end;
