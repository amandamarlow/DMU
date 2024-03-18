using DMUStudent.HW5: HW5, mc
using QuickPOMDPs: QuickPOMDP
using POMDPTools: Deterministic, Uniform, SparseCat, FunctionPolicy, RolloutSimulator
using Statistics: mean
import POMDPs

cancerTreatment = QuickPOMDP(
    states = [:healthy, :inSitu, :invasive, :death],
    actions = [:wait, :test, :treat],
    observations = [:positive, :negative],

    # transition should be a function that takes in s and a and returns the distribution of s'
    transition = function (s, a)
        if s == :healthy
            return SparseCat([:healthy, :inSitu], [0.98, 0.02])
        elseif s == :inSitu
            if a == :treat
                return SparseCat([:inSitu, :healthy], [0.4, 0.6])
            else
                return SparseCat([:inSitu, :invasive], [0.9, 0.1])
            end
        elseif s == :invasive
            if a == :treat
                return SparseCat([:invasive, :healthy, :death], [0.6, 0.2, 0.2])
            else
                return SparseCat([:invasive, :death], [0.4, 0.6])
            end
        else
            return Deterministic(s)
        end
    end,

    # observation should be a function that takes in s, a, and sp, and returns the distribution of o
    observation = function (s, a, sp)
        if a == :test
            if sp == :healthy
                return SparseCat([:positive, :negative], [0.05, 0.95])
            elseif sp == :inSitu
                return SparseCat([:positive, :negative], [0.8, 0.2])
            elseif sp == :invasive
                # return Deterministic(:positive)
                return Uniform([:positive])
            end
        elseif a == :treat && (sp == :inSitu || sp == :invasive)
                # return Deterministic(:positive)
                return Uniform([:positive])
        else
            return Deterministic(:negative)
            # return Uniform([:negative])
        end
    end,

    reward = function (s, a)
        if s == :death
            return 0.0
        else
            if a == :wait
                return 1.0
            elseif a == :test
                return 0.8
            elseif a == :treat
                return 0.1
            end
        end
    end,

    initialstate = Deterministic(:healthy),

    discount = 0.99
)

# evaluate with a random policy
policy = FunctionPolicy(o->POMDPs.actions(cancerTreatment)[1])
sim = RolloutSimulator(max_steps=100)
@show @time mean(POMDPs.simulate(sim, cancerTreatment, policy) for _ in 1:10_000)