using POMDPs
using DMUStudent.HW6
using POMDPTools: transition_matrices, reward_vectors, SparseCat, Deterministic, RolloutSimulator, DiscreteBelief, FunctionPolicy, ordered_states, ordered_actions, DiscreteUpdater
using QuickPOMDPs: QuickPOMDP
using POMDPModels: TigerPOMDP, TIGER_LEFT, TIGER_RIGHT, TIGER_LISTEN, TIGER_OPEN_LEFT, TIGER_OPEN_RIGHT
using NativeSARSOP: SARSOPSolver
using POMDPTesting: has_consistent_distributions
using LinearAlgebra

##################
# Problem 1: Tiger
##################

#--------
# Updater
#--------

struct HW6Updater{M<:POMDP} <: Updater
    m::M
end

function POMDPs.update(up::HW6Updater, b::DiscreteBelief, a, o)
    bp_vec = zeros(length(states(up.m)))
    bp_vec[1] = 1.0

    for i in 1:length(states(up.m))
        s = states(up.m)[i]
        sp = states(up.m)[i]
        bp_vec[i] = Z(up.m, a, sp, o)*sum(T(up.m, s, a, sp))
    end
    bp_vec = bp_vec./sum(bp_vec)
    return DiscreteBelief(up.m, bp_vec)
end

# Note: you can access the transition and observation probabilities through the POMDPs.transtion and POMDPs.observation, and query individual probabilities with the pdf function. For example if you want to use more mathematical-looking functions, you could use the following:
# Z(o | a, s') can be programmed with
Z(m::POMDP, a, sp, o) = pdf(observation(m, a, sp), o)
# T(s' | s, a) can be programmed with
T(m::POMDP, s, a, sp) = pdf(transition(m, s, a), sp)
# POMDPs.transtion and POMDPs.observation return distribution objects. See the POMDPs.jl documentation for more details.

# This is needed to automatically turn any distribution into a discrete belief.
function POMDPs.initialize_belief(up::HW6Updater, distribution::Any)
    b_vec = zeros(length(states(up.m)))
    for s in states(up.m)
        b_vec[stateindex(up.m, s)] = pdf(distribution, s)
    end
    return DiscreteBelief(up.m, b_vec)
end

# Note: to check your belief updater code, you can use POMDPTools: DiscreteUpdater. It should function exactly like your updater.

#-------
# Policy
#-------

struct HW6AlphaVectorPolicy{A} <: Policy
    alphas::Vector{Vector{Float64}}
    alpha_actions::Vector{A}
end

function POMDPs.action(p::HW6AlphaVectorPolicy, b::DiscreteBelief)

    # Fill in code to choose action based on alpha vectors
    expected = Array{Float64}(undef,length(p.alpha_actions))
    for (i, a) in enumerate(p.alpha_actions)
        expected[i] = dot(p.alphas[i],beliefvec(b))
    end
    aidx = argmax(expected)
    a = p.alpha_actions[aidx]
    return a
end

beliefvec(b::DiscreteBelief) = b.b # this function may be helpful to get the belief as a vector in stateindex order

#------
# QMDP
#------

function qmdp_solve(m, discount=discount(m))

    # Fill in Value Iteration to compute the Q-values
    ################################################
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
    ##########################################

    acts = actiontype(m)[]
    alphas = Vector{Float64}[]
    # Q = deepcopy(R)
    alphas = Vector{Vector{Float64}}(undef, length(A))
    for (i, a) in enumerate(A) # key in keys(R)

        # Fill in alpha vector calculation
        # Note that the ordering of the entries in the alpha vectors must be consistent with stateindex(m, s) (states(m) does not necessarily obey this order, but ordered_states(m) does.)
        # Q[a] += discount*(T[a]*V)
        
        alphas[i] = R[a] + discount*(T[a]*V)
    end
    # @show alphas
    acts = A
    # @show acts
    # @show alphas
    return HW6AlphaVectorPolicy(alphas, acts)
end


m = TigerPOMDP()

qmdp_p = qmdp_solve(m)
# Note: you can use the QMDP.jl package to verify that your QMDP alpha vectors are correct.
sarsop_p = solve(SARSOPSolver(), m)
up = HW6Updater(m)

@show mean(simulate(RolloutSimulator(max_steps=500), m, qmdp_p, up) for _ in 1:5000)
# @show mean(simulate(RolloutSimulator(max_steps=500), m, qmdp_p, DiscreteUpdater(m)) for _ in 1:5000)
@show mean(simulate(RolloutSimulator(max_steps=500), m, sarsop_p, up) for _ in 1:5000)

# ###################
# # Problem 2: Cancer
# ###################

# cancer = QuickPOMDP(

#     # Fill in your actual code from last homework here

#     states = [:healthy, :in_situ, :invasive, :death],
#     actions = [:wait, :test, :treat],
#     observations = [true, false],
#     transition = (s, a) -> Deterministic(s),
#     observation = (a, sp) -> Deterministic(false),
#     reward = (s, a) -> 0.0,
#     discount = 0.99,
#     initialstate = Deterministic(:death),
#     isterminal = s->s==:death,
# )

# @assert has_consistent_distributions(cancer)

# qmdp_p = qmdp_solve(cancer)
# sarsop_p = solve(SARSOPSolver(), cancer)
# up = HW6Updater(cancer)

# heuristic = FunctionPolicy(function (b)

#                                # Fill in your heuristic policy here
#                                # Use pdf(b, s) to get the probability of a state

#                                return :wait
#                            end
#                           )

# @show mean(simulate(RolloutSimulator(), cancer, qmdp_p, up) for _ in 1:1000)     # Should be approximately 66
# @show mean(simulate(RolloutSimulator(), cancer, heuristic, up) for _ in 1:1000)
# @show mean(simulate(RolloutSimulator(), cancer, sarsop_p, up) for _ in 1:1000)   # Should be approximately 79

# #####################
# # Problem 3: LaserTag
# #####################

# m = LaserTagPOMDP()

# qmdp_p = qmdp_solve(m)
# up = DiscreteUpdater(m) # you may want to replace this with your updater to test it

# # Use this version with only 100 episodes to check how well you are doing quickly
# @show HW6.evaluate((qmdp_p, up), n_episodes=100)

# # A good approach to try is POMCP, implemented in the BasicPOMCP.jl package:
# using BasicPOMCP
# function pomcp_solve(m) # this function makes capturing m in the rollout policy more efficient
#     solver = POMCPSolver(tree_queries=10,
#                          c=1.0,
#                          default_action=first(actions(m)),
#                          estimate_value=FORollout(FunctionPolicy(s->rand(actions(m)))))
#     return solve(solver, m)
# end
# pomcp_p = pomcp_solve(m)

# @show HW6.evaluate((pomcp_p, up), n_episodes=100)

# # When you get ready to submit, use this version with the full 1000 episodes
# # HW6.evaluate((qmdp_p, up), "REPLACE_WITH_YOUR_EMAIL@colorado.edu")

# #----------------
# # Visualization
# # (all code below is optional)
# #----------------

# # You can make a gif showing what's going on like this:
# using POMDPGifs
# import Cairo, Fontconfig # needed to display properly

# makegif(m, qmdp_p, up, max_steps=30, filename="lasertag.gif")

# # You can render a single frame like this
# using POMDPTools: stepthrough, render
# using Compose: draw, PNG

# history = []
# for step in stepthrough(m, qmdp_p, up, max_steps=10)
#     push!(history, step)
# end
# displayable_object = render(m, last(history))
# # display(displayable_object) # <-this will work in a jupyter notebook or if you have vs code or ElectronDisplay
# draw(PNG("lasertag.png"), displayable_object)