using Distributions

############
# Question 1
############

ep_rand = 1/3*0.2+1/3*0.3+1/3*0.7
@show ep_epsGreedy = 0.15*ep_rand+0.85*0.7

# soft max
theta = [0.2, 0.3, 0.7]
lambda = 5
@show P = exp.(lambda*theta)
@show P = P/sum(exp.(lambda*theta))

# Bayesian estimation

Beta0 = [1,1]
W = [0,1,3]
L = [1,0,2]
dists = Matrix{Float64, 1}(undef,3)
for i in 1:3
    dists[i] = Beta.(Beta0 + [W[i], L[i]])
end

############
# Question 2
############

thetaL = 0.5;
thetaR = 0.5;
@show gradTau1 = [1-(exp(thetaL)/(exp(thetaL)+exp(thetaR))), -(exp(thetaR)/(exp(thetaL)+exp(thetaR)))]*10
@show gradTau1 = [-(exp(thetaR)/(exp(thetaL)+exp(thetaR))), 1-(exp(thetaL)/(exp(thetaL)+exp(thetaR)))]*20