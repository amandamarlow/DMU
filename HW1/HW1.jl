import DMUStudent.HW1

#------------- 
# Problem 4
#-------------

# Here is a functional but incorrect answer for the programming question
function f(a, bs)
    multiplied = Array{typeof(a[1,1]),2}(undef,size(a,1),length(bs))
    for i in eachindex(bs)
        # @show a*bs[i]
        multiplied[:,i] = a*bs[i]
    end
    # @show multiplied
    maxRows = Vector{typeof(a[1,1])}(undef,size(a,1))
    maxRows[1:size(a,1)] = maximum(multiplied, dims=2)
    # @show maxRows
    return maxRows
end

# You can can test it yourself with inputs like this
# a = [1.0 2.0; 3.0 4.0]
# @show a
# bs = [[1.0, 2.0], [3.0, 4.0]]
# @show bs
a = [2 0; 0 1]
@show a
bs = [[1, 2], [2, 1]]
@show bs
@show f(a, bs)

# This is how you create the json file to submit
# HW1.evaluate(f, "your.gradescope.email@colorado.edu")
HW1.evaluate(f, "amanda.marlow@colorado.edu")