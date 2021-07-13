using ReinforcementLearning
using OpenSpiel

function RL.Experiment(
    ::Val{:JuliaRL},
    ::Val{:TabularCFR},
    ::Val{:OpenSpiel},
    game;
    n_iter = 300,
    seed = 123,
)
    env = OpenSpielEnv(game)
    rng = StableRNG(seed)
    π = TabularCFRPolicy(; rng = rng)

    description = "# Play `$game` in OpenSpiel with TabularCFRPolicy"
    Experiment(π, env, StopAfterStep(300, is_show_progress=!haskey(ENV, "CI")), EmptyHook(), description)
end

ex = E`JuliaRL_TabularCFR_OpenSpiel(kuhn_poker)`
run(ex)

# This file was generated using Literate.jl, https://github.com/fredrikekre/Literate.jl

