using DMUStudent.HW3: HW3, DenseGridWorld, visualize_tree
using POMDPs: actions, @gen, isterminal, discount, statetype, actiontype, simulate, states, initialstate
using POMDPTools: render,
                  ordered_states
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


function heuristic_policy(m, s) # similar to random :(
    # put a smarter heuristic policy here
    remainder = mod.(s, 20)
    # @show s
    # @show remainder
    if s[1] <= 10
        a = :right
    elseif s[1] >= 50
        a = :left
    elseif s[2] <= 10
        a = :up
    elseif s[2] >= 50
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

# function heuristic_policy(m, s) # ~16 points better than random
#     # put a smarter heuristic policy here
#     A = collect(actions(m))
#     remainder = mod.(s, 20)
#     r = Vector{Float64}(undef, 4)
#     for (i, a) in enumerate(A)
#         sp = @gen(:sp)(m, s, a)
#         r[i] = @gen(:r)(m, sp, a)
#     end
#     # @show r
#     a = A[argmax(r)]

#     return a
# end

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

# p = VectorPolicy(m, heuristic_policy.(m, states))
# display(render(m))

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
# Question 3
############

S = statetype(m)
A = actiontype(m)

# These would be appropriate containers for your Q, N, and t dictionaries:
n = Dict{Tuple{S, A}, Int}()
q = Dict{Tuple{S, A}, Float64}()
t = Dict{Tuple{S, A, S}, Int}()

function SelectAction(mdp, s, nsims, c)
    S = statetype(mdp)
    A = actiontype(mdp)
    N = Dict{Tuple{S, A}, Int}()
    Q = Dict{Tuple{S, A}, Float64}()
    t = Dict{Tuple{S, A, S}, Int}()
    d = 50;
    for k in 1:nSims
        simulate!(mdp, s, d, c, N, Q)
    end
    return argmax(a->Q[(s,a)], actions(mdp))
end

function simulate!(mdp, s, d, c, N, Q, t=nothing)
    A = collect(actions(mdp))
    if d <= 0
        return rollout(mdp, heuristic_policy, s)
    elseif  !haskey(N, (s, first(A)))
        for a in A
            N[(s,a)] = 0
            Q[(s,a)] = 0.0
        end
        return rollout(mdp, heuristic_policy, s)
    end
    a = explore(s, N, Q, A, c)
    s′, r = @gen(:sp, :r)(mdp, s, a)

    if t !== nothing
        t[(s, a, s′)] = get(t, (s, a, s′), 0) + 1
    end

    q = r + discount(mdp)*simulate!(mdp, s′, d-1, c, N, Q, t)
    N[(s,a)] += 1
    Q[(s,a)] += (q-Q[(s,a)])/N[(s,a)]
    return q
end

bonus(Nsa, Ns) = Nsa == 0 ? Inf : sqrt(log(Ns)/Nsa)

function explore(s, N, Q, A, c)
    Ns = sum(N[(s,a)] for a in A)
    return argmax(a->Q[(s,a)] + c*bonus(N[(s,a)], Ns), A)
end

mdp = DenseGridWorld(seed=4)
nSims = 7
c = 200
s = SA[19,19]
@show SelectAction(mdp, s, nSims, c)

N = Dict{Tuple{S, A}, Int}()
Q = Dict{Tuple{S, A}, Float64}()
t = Dict{Tuple{S, A, S}, Int}()
d = 50;
for k in 1:nSims
    simulate!(mdp, s, d, c, N, Q, t)
end

# @show typeof(s)
# @assert s isa statetype(m)

# # here is an example of how to visualize a dummy tree (q, n, and t should actually be filled in your mcts code, but for this we fill it manually)
# q[(SA[1,1], :right)] = 0.0
# q[(SA[2,1], :right)] = 0.0
# n[(SA[1,1], :right)] = 1
# n[(SA[2,1], :right)] = 0
# t[(SA[1,1], :right, SA[2,1])] = 1

inchrome(visualize_tree(Q, N, t, SA[19,19])) # use inbrowser(visualize_tree(q, n, t, SA[1,1]), "firefox") etc. if you want to use a different browser