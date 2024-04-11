using DMUStudent.HW5: HW5, mc, RL
using QuickPOMDPs: QuickPOMDP
using POMDPTools: Deterministic, Uniform, SparseCat, FunctionPolicy, RolloutSimulator
using Statistics: mean
import POMDPs

using Plots
using DMUStudent
using CommonRLInterface
using Flux
using CommonRLInterface.Wrappers: QuickWrapper


############
# Question 3
############

# Override to a discrete action space, and position and velocity observations rather than the matrix.
env = QuickWrapper(HW5.mc,
                   actions=[-1.0, -0.5, 0.0, 0.5, 1.0],
                   observe=mc->observe(mc)[1:2]
                  )


function epsGreedy(env, eps, Qs)
    if rand() < eps
        a_idx = rand(1:length(actions(env)))
    else
        a_idx = argmax(Qs)
    end
end                  

function dqn(env)

    epochs = 1000
    updateQ = 1000
    buffSize = 100_000
    sampleSize = 1000
    gamma = 0.99
    eps = 0.5

    # This network should work for the Q function - an input is a state; the output is a vector containing the Q-values for each action 
    Q = Chain(Dense(2, 128, relu), Dense(128, length(actions(env))))
    opt = Flux.setup(ADAM(0.005), Q)
    Qp = deepcopy(Q)
    Qbest = deepcopy(Q)
    rbest = 0.0
    buffer = []
    rtot = []
    steps = 0
    i = 0
    # for i = 1:epochs
    while (steps < 100_000) || (rtot[i-1]>45)
        i += 1
        # print("epoch = ", i, "\n")
        print(steps, " steps\n")
        RL.reset!(env)
        done = terminated(env)
        j = 0
        while !done && (gamma^j > 0.01)#j < 100_000
            j += 1
            steps += 1

            s = observe(env)
            a_ind = epsGreedy(env, eps, Q(s))
            r = act!(env, actions(env)[a_ind])
            sp = observe(env)
            done = terminated(env)

            experience_tuple = (s, a_ind, r, sp, done)

            # this container should work well for the experience buffer:
            buffer = push!(buffer, experience_tuple)
            if length(buffer)>buffSize
                popfirst!(buffer)
            end

            function loss(Q, s, a_ind, r, sp, done)
                if done
                    return (r - Q(s)[a_ind])^2
                end
                return (r + gamma*maximum(Qp(sp)) - Q(s)[a_ind])^2
            end

            if mod(steps, 100) == 0
                data = rand(buffer, sampleSize)
                Flux.Optimise.train!(loss, Q, data, opt)
            end

            if mod(steps,updateQ) == 0
                Qp = deepcopy(Q)
            end

            eps = max(0.05, eps-(1-0.05)/100_000)
        end

        push!(rtot, HW5.evaluate(s->actions(env)[argmax(Q(s[1:2]))], n_episodes=100)[1])
        print(rtot[i],'\n')
        if rtot[i] > rbest
            Qbest = deepcopy(Q)
            rbest = copy(rtot[i])
            buffer = []
        end
    end
    p = plot(1:length(rtot), rtot, label="rtot")
    display(p)
    return Qbest
end

@time begin
Q = dqn(env)
end

# @show HW5.evaluate(s->actions(env)[argmax(Q(s[1:2]))], n_episodes=100) # you will need to remove the n_episodes=100 keyword argument to create a json file; evaluate needs to run 10_000 episodes to produce a json
@show HW5.evaluate(s->actions(env)[argmax(Q(s[1:2]))], "amanda.marlow@colorado.edu") # you will need to remove the n_episodes=100 keyword argument to create a json file; evaluate needs to run 10_000 episodes to produce a json

#----------
# Rendering
#----------

# # You can show an image of the environment like this (use ElectronDisplay if running from REPL):
# display(render(env))

# The following code allows you to render the value function
using Plots
xs = -3.0f0:0.1f0:3.0f0
vs = -0.3f0:0.01f0:0.3f0
display(heatmap(xs, vs, (x, v) -> maximum(Q([x, v])), xlabel="Position (x)", ylabel="Velocity (v)", title="Max Q Value"))


# function render_value(value)
#     xs = -3.0:0.1:3.0
#     vs = -0.3:0.01:0.3
# 
#     data = DataFrame(
#                      x = vec([x for x in xs, v in vs]),
#                      v = vec([v for x in xs, v in vs]),
#                      val = vec([value([x, v]) for x in xs, v in vs])
#     )
# 
#     data |> @vlplot(:rect, "x:o", "v:o", color=:val, width="container", height="container")
# end
# 
# display(render_value(s->maximum(Q(s))))

# str = "All done!"
# run(`powershell -Command "\$sp = New-Object -ComObject SAPI.SpVoice; \$sp.Speak(\"$str\")"`)
