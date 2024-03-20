using DMUStudent.HW5: HW5, mc
using QuickPOMDPs: QuickPOMDP
using POMDPTools: Deterministic, Uniform, SparseCat, FunctionPolicy, RolloutSimulator
using Statistics: mean
import POMDPs

using Plots: scatter, scatter!, plot, plot!
using Flux
using StaticArrays
using Random: randperm
# using IJulia

############
# Question 1
############

# cancerTreatment = QuickPOMDP(
#     states = [:healthy, :inSitu, :invasive, :death],
#     actions = [:wait, :test, :treat],
#     observations = [:positive, :negative],

#     # transition should be a function that takes in s and a and returns the distribution of s'
#     transition = function (s, a)
#         if s == :healthy
#             return SparseCat([:healthy, :inSitu], [0.98, 0.02])
#         elseif s == :inSitu
#             if a == :treat
#                 return SparseCat([:inSitu, :healthy], [0.4, 0.6])
#             else
#                 return SparseCat([:inSitu, :invasive], [0.9, 0.1])
#             end
#         elseif s == :invasive
#             if a == :treat
#                 return SparseCat([:invasive, :healthy, :death], [0.6, 0.2, 0.2])
#             else
#                 return SparseCat([:invasive, :death], [0.4, 0.6])
#             end
#         else
#             return Deterministic(s)
#         end
#     end,

#     # observation should be a function that takes in s, a, and sp, and returns the distribution of o
#     observation = function (s, a, sp)
#         if a == :test
#             if sp == :healthy
#                 return SparseCat([:positive, :negative], [0.05, 0.95])
#             elseif sp == :inSitu
#                 return SparseCat([:positive, :negative], [0.8, 0.2])
#             elseif sp == :invasive
#                 # return Deterministic(:positive)
#                 return Uniform([:positive])
#             end
#         elseif a == :treat && (sp == :inSitu || sp == :invasive)
#                 # return Deterministic(:positive)
#                 return Uniform([:positive])
#         else
#             return Deterministic(:negative)
#             # return Uniform([:negative])
#         end
#     end,

#     reward = function (s, a)
#         if s == :death
#             return 0.0
#         else
#             if a == :wait
#                 return 1.0
#             elseif a == :test
#                 return 0.8
#             elseif a == :treat
#                 return 0.1
#             end
#         end
#     end,

#     initialstate = Deterministic(:healthy),

#     discount = 0.99
# )

# # evaluate with a random policy
# policy = FunctionPolicy(o->POMDPs.actions(cancerTreatment)[1])
# sim = RolloutSimulator(max_steps=100)
# @show @time mean(POMDPs.simulate(sim, cancerTreatment, policy) for _ in 1:10_000)

############
# Question 2
############

n = 100
dx = rand(Float32,n)
# dy = sin.(4*pi*dx) + 0.1*randn(n);
# scatter(dx, dy)
# dx = LinRange(0.0,1.0,n)
# dy = (1.0 .− dx).*sin.(20.0*log.(dx .+ 0.2))  + 0.1*randn(Float32, n)
# f(x) = (1 − x).*sin.(20*log.(x .+ 0.2)) + 0.1*randn(Float32)
f(x) = (1 − x).*sin.(20*log.(x .+ 0.2))
dy = f.(dx)
# dy = sin.(4*pi*dx)

# scatter(dx, dy)

# layerSize = 128
# layerSize = 64
layerSize = 50
# m = Chain(Dense(1=>50,σ), Dense(50=>50,σ), Dense(50=>1))
# m = Chain(Dense(1=>layerSize,σ), Dense(layerSize=>layerSize,σ), Dense(layerSize=>1))
m = Chain(Dense(1=>layerSize,relu), Dense(layerSize=>layerSize,relu), Dense(layerSize=>1))
# m = Chain(Dense(1=>25,relu), Dense(25=>50,relu), Dense(50=>1))

# loss(x, y) = Flux.mse(m(x), y)
loss(x, y) = sum((m(x) .-y).^2)

data = [(SVector(dx[i]), SVector(dy[i])) for i in 1:length(dx)]
# data = [(SVector(x[i]), SVector(y[i])) for i in 1:length(x)]

# for i in 1:2000
#     Flux.train!(loss, Flux.params(m), data, Adam())
#     if i%50 == 0
#         p = plot(sort(dx), x->(sin(4*pi*x)), label="sin(4π x)")
#         plot!(p, sort(dx), first.(m.(SVector.(sort(dx)))), label="NN approximation")
#         scatter!(p, dx, dy, label="data")
#         display(i)
#         display(p)
#     end
# end

epochs = 10000
losses = Array{Float32}(undef,epochs)
for i in 1:epochs
    # Flux.train!(loss, Flux.params(m), data, Descent(0.1))
    Flux.train!(loss, Flux.params(m), data, Adam())
    losses[i] = mean(loss.(SVector.(dx), dy))
    if i%50 == 0
        p = plot(sort(dx), x->((1 .−x).*sin.(20*log.(x .+0.2))), label=" (1−x)sin(20log(x+0.2))")
        plot!(p, sort(dx), first.(m.(SVector.(sort(dx)))), label="NN approximation")
        scatter!(p, dx, dy, label="data")
        display(i)
        display(p)
    end
end

plot(losses)

xplot = collect(LinRange(0.0,1.0,n))
yplot = first.(m.(SVector.(sort(xplot))))
trainedPlot = plot(xplot, x->((1 .−x).*sin.(20*log.(x .+0.2))), label=" (1−x)sin(20log(x+0.2))")
plot!(trainedPlot, xplot, yplot, label="Trained NN approximation")

