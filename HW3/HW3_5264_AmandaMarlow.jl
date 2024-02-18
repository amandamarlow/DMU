using DMUStudent.HW3: HW3, DenseGridWorld, visualize_tree
using POMDPs: actions, @gen, isterminal, discount, statetype, actiontype, simulate, states, initialstate
using POMDPTools: render
using D3Trees: inchrome, inbrowser
using StaticArrays: SA
using Statistics: mean, std
using BenchmarkTools: @btime


############
# Question 2
############


function rollout(mdp, policy_function, s0, max_steps=100)
    r_total = 0
    t = 0
    s = s0
    while !isterminal(mdp, s) && t < max_steps
        a = policy_function(m, s)
        s, r = @gen(:sp, :r)(mdp, s, a)
        r_total += discount(m)^t*r
        t += 1
    end
    return r_total
end

function heuristic_policy(m, s)
    # put a smarter heuristic policy here
    remainder = mod.(s, 20)
    if (remainder[1] == 0) && (remainder[2] < 10)
        a = :left
    elseif (remainder[1] == 0) && (remainder[2] > 10)
        a = :right
    elseif remainder[1] > 10
        a = :down
    elseif remainder[1] <= 10
        a = :up
    else
        throw(ArgumentError("Can't resolve action"))
    end 

    return a
end

function random_policy(m, s)
    return rand(actions(m))
end

function get_mean_SEM(data::Vector{Float64})
    n = length(data)
    # Check if there are enough observations
    if n < 2
        throw(ArgumentError("Input vector must have at least 2 elements"))
    end 
    # Calculate the mean and standard deviation
    mean_value = mean(data)
    std_dev = std(data)  
    # Calculate the standard error of the mean
    sem = std_dev / sqrt(n)

    return mean_value, sem
end


m = HW3.DenseGridWorld(seed=3)
iterations = 300

# display(render(m))

results_random = [rollout(m, random_policy, rand(initialstate(m))) for _ in 1:iterations]
mean_random, SEM_random = get_mean_SEM(results_random)
@show mean_random
@show SEM_random

# This code runs monte carlo simulations: you can calculate the mean and standard error from the results
results_heuristic = [rollout(m, heuristic_policy, rand(initialstate(m))) for _ in 1:iterations]
mean_heuristic, SEM_heuristic = get_mean_SEM(results_heuristic)
@show mean_heuristic
@show SEM_heuristic