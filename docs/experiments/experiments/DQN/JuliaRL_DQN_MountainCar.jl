using ReinforcementLearning
using GridWorlds
using StableRNGs
using Flux
using Flux.Losses

function RL.Experiment(
    ::Val{:JuliaRL},
    ::Val{:DQN},
    ::Val{:MountainCar},
    ::Nothing;
    seed = 123,
)
    rng = StableRNG(seed)
    env = MountainCarEnv(; T = Float32, max_steps = 5000, rng = rng)
    ns, na = length(state(env)), length(action_space(env))

    agent = Agent(
        policy = QBasedPolicy(
            learner = DQNLearner(
                approximator = NeuralNetworkApproximator(
                    model = Chain(
                        Dense(ns, 64, relu; init = glorot_uniform(rng)),
                        Dense(64, 64, relu; init = glorot_uniform(rng)),
                        Dense(64, na; init = glorot_uniform(rng)),
                    ) |> cpu,
                    optimizer = ADAM(),
                ),
                target_approximator = NeuralNetworkApproximator(
                    model = Chain(
                        Dense(ns, 64, relu; init = glorot_uniform(rng)),
                        Dense(64, 64, relu; init = glorot_uniform(rng)),
                        Dense(64, na; init = glorot_uniform(rng)),
                    ) |> cpu,
                    optimizer = ADAM(),
                ),
                loss_func = huber_loss,
                stack_size = nothing,
                batch_size = 32,
                update_horizon = 1,
                min_replay_history = 100,
                update_freq = 1,
                target_update_freq = 100,
                rng = rng,
            ),
            explorer = EpsilonGreedyExplorer(
                kind = :exp,
                ϵ_stable = 0.01,
                decay_steps = 500,
                rng = rng,
            ),
        ),
        trajectory = CircularArraySARTTrajectory(
            capacity = 50000,
            state = Vector{Float32} => (ns,),
        ),
    )

    stop_condition = StopAfterStep(40_000, is_show_progress=!haskey(ENV, "CI"))
    hook = TotalRewardPerEpisode()
    Experiment(agent, env, stop_condition, hook, "")
end

using Plots
pyplot() #hide
ex = E`JuliaRL_DQN_MountainCar`
run(ex)
plot(ex.hook.rewards)
savefig("assets/JuliaRL_DQN_MountainCar.png") #hide

# This file was generated using Literate.jl, https://github.com/fredrikekre/Literate.jl

