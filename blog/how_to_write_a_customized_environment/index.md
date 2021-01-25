@def title = "How to write a customized environment?"
@def description = "The first step to apply algorithms in ReinforcementLearning.jl is to define the problem you want to solve in a recognizable way. Here we'll demonstrate how to write many different kinds of environments based on interfaces defined in [ReinforcementLearningBase.jl][]."
@def is_enable_toc = false
@def bibliography = "bibliography.bib"

@def front_matter = """
    {
        "authors": [
            {
                "author":"Jun Tian",
                "authorURL":"https://github.com/findmyway",
                "affiliation":"",
                "affiliationURL":""
            }
        ],
        "publishedDate":"$(now())"
    }"""

The most commonly used interfaces to describe reinforcement learning tasks is [OpenAI/Gym](https://gym.openai.com/). Inspired by it, we expand those interfaces a little to utilize the multiple-dispatch in Julia and to cover multi-agent environments.

## The minimal interfaces to implement

Many interfaces in [ReinforcementLearningBase.jl][] have a default implementation. So in most cases, you only need to implement the following functions to define a customized environment:

```julia
action_space(env::YourEnv)
state(env::YourEnv)
reward(env::YourEnv)
is_terminated(env::YourEnv)
reset!(env::YourEnv)
(env::YourEnv)(action)
```

Here we use an example introduced in [Monte Carlo Tree Search: A Tutorial](https://www.informs-sim.org/wsc18papers/includes/files/021.pdf) to demonstrate how to write a simple environment.

The game is defined like this: assume you have \$10 in your pocket, and you are faced with the following three choices:

1. Buy a PowerRich lottery ticket (win \$100M w.p. 0.01; nothing otherwise);
1. Buy a MegaHaul lottery ticket (win \$1M w.p. 0.05; nothing otherwise);
1. Do not buy a lottery ticket.

First we define a concrete subtype of `AbstractEnv` named `LotteryEnv`:

```julia:./lottery_env
using ReinforcementLearningBase

mutable struct LotteryEnv <: AbstractEnv
    reward::Union{Nothing, Int}
end

LotteryEnv() = LotteryEnv(nothing)
```

`LotteryEnv` has only one field named `reward`, by default it is initialized with `nothing`. Now let's implement the necessary interfaces:

```julia:./lottery_env
RLBase.action_space(env::LotteryEnv) = (:PowerRich, :MegaHaul, nothing)
```

Here `RLBase` is just an alias for `ReinforcementLearningBase`.

```julia:./lottery_env
RLBase.reward(env::LotteryEnv) = env.reward
RLBase.state(env::LotteryEnv) = !isnothing(env.reward)
RLBase.is_terminated(env::LotteryEnv) = !isnothing(env.reward)
RLBase.reset!(env::LotteryEnv) = env.reward = nothing
```

Because the lottery game is just a simple one-shot game. If the `reward` is `nothing` then the game is not terminated yet and we say the game is in state `false`, otherwise the game is terminated and the state is `true`. By `reset!` the game, we simply assign the reward with `nothing`, meaning that it's in the initial state.

The only left one is to implement the game logic:

```julia:./lottery_env
function (env::LotteryEnv)(action)
    if action == :PowerRich
        env.reward = rand() < 0.01 ? 100_000_000 : -10
    elseif action == :MegaHaul
        env.reward = rand() < 0.05 ? 1_000_000 : -10
    else
        env.reward = 0
    end
end
```

A simple way to check that your environment works is to apply the [`RandomPolicy`](https://juliareinforcementlearning.org/ReinforcementLearning.jl/latest/rl_base/#ReinforcementLearningBase.RandomPolicy) to the environment.

```julia:./lottery_env
env = LotteryEnv()
run(RandomPolicy(env), env)
```

One step further is to test that other components in ReinforcementLearning.jl also work:

```julia:./lottery_env
using ReinforcementLearning
using Random # hide
Random.seed!(123) # hide
hook = TotalRewardPerEpisode()
run(
    Agent(
        ;policy = RandomPolicy(env),
        trajectory = VectCompactSARTSATrajectory(
            state_type=Bool,
            action_type=Any,
            reward_type=Int,
            terminal_type=Bool,
        ),
    ),
    LotteryEnv(),
    StopAfterEpisode(1_000),
    hook
)

println(sum(hook.rewards) / 1_000)
```

\output{./lottery_env}

## Traits of environments

If you run `LotteryEnv()` in the REPL, you'll get the following summary of the environment:

```julia:./show_lottery_env
# hideall
show(stdout, MIME"text/plain"(), LotteryEnv())  # hide
```

\output{./show_lottery_env}

The **Traits** section describes which category the environment belongs to. As you can see, by default an environment is assumed to be of:

- `SingleAgent`
- `Sequential`
- `PerfectInformation`
- `Deterministic`
- `StepReward`
- `GeneralSum`
- `MinimalActionSet`

### ActionStyle

```julia:./doc_of_ActionStyle
# hideall
print(@doc ActionStyle)
```

\textoutput{./doc_of_ActionStyle}

For environments of `FULL_ACTION_SET`, the following methods must be implemented:

- `legal_action_space(env)`
- `legal_action_space_mask(env)`

### DynamicStyle


```julia:./doc_of_DynamicStyle
# hideall
print(@doc DynamicStyle)
```

\textoutput{./doc_of_DynamicStyle}

For environment of `SIMULTANEOUS`, the actions in each step must be a collection, representing the joint actions from all players.

### UtilityStyle

```julia:./doc_of_UtilityStyle
# hideall
print(@doc UtilityStyle)
```

\textoutput{./doc_of_UtilityStyle}

### RewardStyle

```julia:./doc_of_RewardStyle
# hideall
print(@doc RewardStyle)
```

\textoutput{./doc_of_RewardStyle}

Some algorithms may use this trait for acceleration.

### ChanceStyle

```julia:./doc_of_ChanceStyle
# hideall
print(@doc ChanceStyle)
```

\textoutput{./doc_of_ChanceStyle}

Possible values are:

- `Deterministic`
- `Stochastic`
- `ExplicitStochastic`
- `SampledStochastic`

Some algorithms may only work on environments of `Deterministic` or `ExplicitStochastic`.

### InformationStyle

```julia:./doc_of_InformationStyle
# hideall
print(@doc InformationStyle)
```

\textoutput{./doc_of_InformationStyle}

### NumAgentStyle

```julia:./doc_of_NumAgentStyle
# hideall
print(@doc NumAgentStyle)
```

\textoutput{./doc_of_NumAgentStyle}

The `NumAgentStyle` trait is used to define the number of agents in an environment. Possible values are `SINGLE_AGENT` or `MultiAgent{N}()`. In multi-agent environments, a special case is `Two_Agent`, which is an alias of `MultiAgent{2}()`. For multi-agent environments, many functions need to accept another argument named `player` (for example `reward(env,player)`) to support getting information from the perspective of a specific player. Here's the list of these functions:

```julia:./list_of_multi_agent_methods
# hideall
for x in ReinforcementLearningBase.MULTI_AGENT_ENV_API
    println(x)
end
```

\output{./list_of_multi_agent_methods}

## Environment wrappers

Some useful environment wrappers are also provided in [ReinforcementLearningBase.jl][] to mimic OOP. For example, in the above `LotteryEnv`, actions are of type `Union{Symbol, Nothing}`. Some algorithms may require that the actions must be discrete integers. Then we can create a wrapped environment:

```julia
inner_env = LotteryEnv()
env = inner_env |> ActionTransformedEnv(a -> action_space(inner_env)[a])
RLBase.action_space(env::ActionTransformedEnv{<:LotteryEnv}) = 1:3
```

In some other cases, we may want to transform the state into integers. Similarly we can achieve this goal with the following code:

```julia
env = LotteryEnv() |> StateOverriddenEnv(s -> Int(s))
```

See the full list of other environment wrappers [here](https://github.com/JuliaReinforcementLearning/ReinforcementLearningBase.jl/blob/master/src/implementations/environments.jl).

[ReinforcementLearningBase.jl]: https://github.com/JuliaReinforcementLearning/ReinforcementLearningBase.jl/blob/master/src/interface.jl