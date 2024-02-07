using DMUStudent.HW2
using POMDPs: states, actions
using POMDPTools: render,
                  Policies.policy_transition_matrix,
                  Policies.policy_reward_vector,
                  ordered_states,
                  ordered_actions,
                  VectorPolicy
using POMDPModels
using LinearAlgebra

############
# Question 3
############

display(render(SimpleGridWorld()))
# T = transition_matrices(grid_world)
# R = reward_vectors(grid_world)

function value_iteration(m, discount)
    
    e = 1e-7
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
    # pi = argmax(R[A] + discount*T[A]*V, A)

    return V
end

V3 = value_iteration(grid_world, 0.95)

display(render(grid_world, color=V3))



############
# Question 4
############

function checkDist(state)
    if (state[4] <=100) && (abs(state[1]-state[3])<=500)
        return -100;
    else
        return 0;
    end
end

function ACAS_value_iteration(m, discount)
    
    e = 1e-7
    n = length(states(m))

    # A = collect(ordered_actions(m))
    A = collect(actions(m))
    T = transition_matrices(m, sparse=true)
    R = reward_vectors(m)

    V = rand(Float64, n)
    Vnext = checkDist.(states(m))
    
    while norm(V-Vnext, 2) > e
        V[:] = Vnext
        Voptions = Array{Float64, 2}(undef,length(A), n)
        q = 1
        for a in A #ordered_actions(m)
            Voptions[q,:] = R[a] + discount*T[a]*V
            q = q+1
        end
        Vnext[:] = maximum(Voptions, dims=1)
    end
    V[:] = Vnext

    return V
end

n = 10
m = HW2.UnresponsiveACASMDP(n)
V4 = ACAS_value_iteration(m, 0.99)

@show HW2.evaluate(V4, "amanda.marlow@colorado.edu")