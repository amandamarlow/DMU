using DMUStudent.HW3: HW3, DenseGridWorld, visualize_tree
using POMDPs: actions, @gen, isterminal, discount, statetype, actiontype, simulate, states, initialstate
using POMDPTools: render,
                  ordered_states
using D3Trees: inchrome, inbrowser
using StaticArrays: SA
using Statistics: mean, std
using BenchmarkTools: @btime
using LinearAlgebra


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


function heuristic_policy(mdp, s)
    
    gridSize = 100
    
    remainder = mod.(s, 20)
    # @show s
    # @show remainder
    if s[1] <= 10
        a = :right
    elseif s[1] >= gridSize-10
        a = :left
    elseif s[2] <= 10
        a = :up
    elseif s[2] >= gridSize-10
        a = :down
    elseif (remainder[1] == 0) && (remainder[2] <= 10)
        a = :down
    elseif (remainder[1] == 0) && (remainder[2] > 10)
        a = :up
    elseif remainder[1] > 10
        a = :right
    elseif remainder[1] <= 10
        a = :left
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
    # Calculate the mean and standard deviation
    mean_value = mean(data)
    std_dev = std(data)  
    # Calculate the standard error of the mean
    sem = std_dev / sqrt(n)

    return mean_value, sem
end


m = HW3.DenseGridWorld(seed=3)
iterations = 500

# results_random = [rollout(m, random_policy, rand(initialstate(m))) for _ in 1:iterations]
# mean_random, SEM_random = get_mean_SEM(results_random)
# @show mean_random
# @show SEM_random

# # This code runs monte carlo simulations: you can calculate the mean and standard error from the results
# results_heuristic = [rollout(m, heuristic_policy, rand(initialstate(m))) for _ in 1:iterations]
# mean_heuristic, SEM_heuristic = get_mean_SEM(results_heuristic)
# @show mean_heuristic
# println("Policy Improvement: ", mean_heuristic-mean_random)
# @show SEM_heuristic


############
# MCTS Code
############

function simulate!(mdp, s, d, N, Q, T=nothing)
    A = collect(actions(mdp))
    if d <= 0
        return rollout(mdp, heuristic_policy, s, d)
    elseif  !haskey(N, (s, first(A)))
        for a in A
            N[(s,a)] = 0
            Q[(s,a)] = 0.0
        end
        return rollout(mdp, heuristic_policy, s, d)
    end
    c = 200 # exploration constant
    a = explore(s, N, Q, A, c)
    s′, r = @gen(:sp, :r)(mdp, s, a)

    if T !== nothing
        T[(s, a, s′)] = get(T, (s, a, s′), 0) + 1
    end

    q = r + discount(mdp)*simulate!(mdp, s′, d-1, N, Q, T)
    N[(s,a)] += 1
    Q[(s,a)] += (q-Q[(s,a)])/N[(s,a)]
    return q
end

bonus(Nsa, Ns) = Nsa == 0 ? Inf : sqrt(log(Ns)/Nsa)
function explore(s, N, Q, A, c)
    Ns = sum(N[(s,a)] for a in A)
    return argmax(a->Q[(s,a)] + c*bonus(N[(s,a)], Ns), A)
end

############
# Question 3
############

mdp = DenseGridWorld(seed=4)
# nsims = 7
# c = 200
# s = SA[19,19]
# S = statetype(mdp)
# A = actiontype(mdp)
# N = Dict{Tuple{S, A}, Int}()
# Q = Dict{Tuple{S, A}, Float64}()
# T = Dict{Tuple{S, A, S}, Int}()
# d = 50;
# for k in 1:nsims
#     simulate!(mdp, s, d, N, Q, T)
# end

# inchrome(visualize_tree(Q, N, T, SA[19,19])) # use inbrowser(visualize_tree(q, n, t, SA[1,1]), "firefox") etc. if you want to use a different browser

############
# Question 4
############

function SelectAction(mdp, s; nsims=nothing, tlim=nothing)
    start = time_ns()

    A = collect(actions(mdp))

    N = Dict{Tuple{statetype(mdp), actiontype(mdp)}, Int}()
    Q = Dict{Tuple{statetype(mdp), actiontype(mdp)}, Float64}()
    d = 50;
    if isnothing(tlim) && !isnothing(nsims)
        for k in 1:nsims
            simulate!(mdp, s, d, N, Q)
        end
    elseif !isnothing(tlim) && isnothing(nsims) 
        while time_ns() < start + tlim
            simulate!(mdp, s, d, N, Q)
        end
    else
        throw(ArgumentError("Not enough input arguments for loop definition"))
    end
    return argmax(a->Q[(s,a)], A)
end

# @btime SelectAction(mdp, SA[35,35], nsims = 1000)


function rolloutQ4(mdp, s0, max_steps=100)
    r_total = 0
    t = 0
    s = s0
    while !isterminal(mdp, s) && t < max_steps
        a = SelectAction(mdp, s; nsims=1000)
        s, r = @gen(:sp, :r)(mdp, s, a)
        # @show r
        # @show s
        r_total += r
        t += 1
    end
    return r_total
end

iterations = 100;

# results_MCTS = [rolloutQ4(mdp, rand(initialstate(mdp))) for _ in 1:iterations]
# mean_MCTS, SEM_MCTS = get_mean_SEM(results_MCTS)
# @show mean_MCTS
# @show SEM_MCTS

############
# Question 5
############

select_action(mdp,s) = SelectAction(mdp, s, tlim=40000000)
HW3.evaluate(select_action, "amanda.marlow@colorado.edu", time=true)