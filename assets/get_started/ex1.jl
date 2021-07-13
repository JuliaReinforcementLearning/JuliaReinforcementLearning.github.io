# This file was generated, do not modify it. # hide
using ReinforcementLearning # hide
using Plots
pyplot() #hide # hide

experiment = E`JuliaRL_BasicDQN_CartPole`
hook = TotalRewardPerEpisode()
run(experiment.policy, experiment.env, experiment.stop_condition, hook)
plot(hook.rewards)
savefig(joinpath(@OUTPUT, "episode.svg")) # hide