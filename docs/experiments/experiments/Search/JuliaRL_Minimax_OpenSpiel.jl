using ReinforcementLearning
using OpenSpiel

function RL.Experiment(::Val{:JuliaRL}, ::Val{:Minimax}, ::Val{:OpenSpiel}, game;)
    env = OpenSpielEnv(string(game))
    agents = MultiAgentManager(
        NamedPolicy(0 => MinimaxPolicy()),
        NamedPolicy(1 => MinimaxPolicy()),
    )
    hooks = MultiAgentHook(0 => TotalRewardPerEpisode(), 1 => TotalRewardPerEpisode())
    description = "# Play `$game` in OpenSpiel with Minimax"
    Experiment(agents, env, StopAfterEpisode(1, is_show_progress=!haskey(ENV, "CI")), hooks, description)
end

using Plots
ex = E`JuliaRL_Minimax_OpenSpiel(tic_tac_toe)`
run(ex)

ex.hook

# This file was generated using Literate.jl, https://github.com/fredrikekre/Literate.jl

