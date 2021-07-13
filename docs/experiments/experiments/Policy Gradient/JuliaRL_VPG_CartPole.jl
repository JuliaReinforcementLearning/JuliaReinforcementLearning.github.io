using ReinforcementLearning
using StableRNGs
using Flux
using Flux.Losses
using Distributions

function RL.Experiment(
    ::Val{:JuliaRL},
    ::Val{:VPG},
    ::Val{:CartPole},
    ::Nothing;
    seed = 123,
)
    rng = StableRNG(seed)
    env = CartPoleEnv(; T = Float32, rng = rng)
    ns, na = length(state(env)), length(action_space(env))

    agent = Agent(
        policy = VPGPolicy(
            approximator = NeuralNetworkApproximator(
                model = Chain(
                    Dense(ns, 128, relu; init = glorot_uniform(rng)),
                    Dense(128, 128, relu; init = glorot_uniform(rng)),
                    Dense(128, na; init = glorot_uniform(rng)),
                ),
                optimizer = ADAM(),
            ) |> cpu,
            baseline = NeuralNetworkApproximator(
                model = Chain(
                    Dense(ns, 128, relu; init = glorot_uniform(rng)),
                    Dense(128, 128, relu; init = glorot_uniform(rng)),
                    Dense(128, 1; init = glorot_uniform(rng)),
                ),
                optimizer = ADAM(),
            ) |> cpu,
            action_space = action_space(env),
            dist = Categorical,
            γ = 0.99f0,
            rng = rng,
        ),
        trajectory = ElasticSARTTrajectory(state = Vector{Float32} => (ns,)),
    )
    stop_condition = StopAfterEpisode(500, is_show_progress=!haskey(ENV, "CI"))

    hook = TotalRewardPerEpisode()
    description = "# Play CartPole with VPG"
    Experiment(agent, env, stop_condition, hook, description)
end

using Plots
pyplot() #hide
ex = E`JuliaRL_VPG_CartPole`
run(ex)
plot(ex.hook.rewards)
savefig("assets/JuliaRL_VPG_CartPole.png") #hide

# This file was generated using Literate.jl, https://github.com/fredrikekre/Literate.jl

