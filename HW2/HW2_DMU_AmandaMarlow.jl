using DMUStudent.HW2
using POMDPs: states, actions
using POMDPTools: ordered_states
using POMDPModels
using LinearAlgebra

############
# Question 3
############

display(render(SimpleGridWorld()))
# T = transition_matrices(grid_world)
# R = reward_vectors(grid_world)

function value_iteration(m)
    
    e = 1e-6
    discount = 0.95
    n = length(states(m))

    A = collect(actions(m))
    T = transition_matrices(m)
    R = reward_vectors(m)

    V = rand(Float64, n) # this would be a good container to use for your value function
    Vnext = rand(Float64, n)
    
    while norm(V-Vnext, 2) > e
        V[:] = Vnext
        Voptions = Array{Float64, 2}(undef,length(A), n)
        q = 1
        for a in A
            Voptions[q,:] = R[a] + discount*T[a]*V
            q = q+1
        end
        Vnext[:] = maximum(Voptions, dims=1)
    end
    V = Vnext

    return V
end

V = value_iteration(grid_world)

display(render(grid_world, color=V))

#@show actions(grid_world) # prints the actions. In this case each action is a Symbol. Use ?Symbol to find out more.

# T = transition_matrices(grid_world)
# display(T) # this is a Dict that contains a transition matrix for each action

#e = 0.5;
#U = ;
#U1 = ;
