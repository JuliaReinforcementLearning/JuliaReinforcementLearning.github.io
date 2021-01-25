@def title = "A Whirlwind Tour of ReinforcementLearning.jl"
@def description = "Welcome to the world of reinforcement learning in Julia. Now let's get started in 3 lines!"
@def is_enable_toc = false
@def has_code = true
@def has_math = true

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
        "publishedDate":"$(now())",
        "citationText":"Jun Tian, $(Dates.format(now(), "Y"))"
    }"""

@def appendix = """
    ### Corrections
    If you see mistakes or want to suggest changes, please [create an issue](https://github.com/JuliaReinforcementLearning/JuliaReinforcementLearning.github.io/issues) on the source repository.
    """

## Prepare

First things first, [download](https://julialang.org/downloads/) and install Julia of the latest stable version. ReinforcementLearning.jl is tested on all platforms, so just choose the one you are familiar with. If you already have Julia installed, please make sure that it is `v1.3` or above.

\aside{ReinforcementLearning.jl relies on some features introduced since `v1.3`, like [MultiThreading](https://docs.julialang.org/en/v1/base/multi-threading/index.html), and [Artifacts](https://julialang.github.io/Pkg.jl/dev/artifacts/)}

Another useful tool is [tensorboard](https://github.com/tensorflow/tensorboard) \footnote{You don't need to install the whole TensorFlow to use the TensorBoard. Behind the scene, ReinforcementLearning.jl uses [TensorBoardLogger.jl](https://github.com/PhilipVinc/TensorBoardLogger.jl) to write data into the format that TensorBoard recognizes.}. You can install it via `pip install tensorboard` with the python package installer [`pip`](https://pip.pypa.io/en/stable/installing/).

## Get Started

Run `julia` in the command line (or double-click the Julia executable) and now you are in an interactive session (also known as a read-eval-print loop or "REPL"). Then execute the following code:

```julia
] add ReinforcementLearning

using ReinforcementLearning

run(E`JuliaRL_BasicDQN_CartPole`)
```

So what's happening here?

1. In the first line, typing `]` will bring you to the *Pkg* mode. `add ReinforcementLearning` will install the latest version of `ReinforcementLearning.jl` for you. And then remember to press backspace or ^C to get back to the normal mode.
1. `using ReinforcementLearning` will bring the names exported in `ReinforcementLearning` into global scope. If this is your first time to run, you'll see *precompiling ReinforcementLearning*. And it may take a while.
1. The third line means, `run` an **E**xperiment named `JuliaRL_BasicDQN_CartPole` \footnote{The ``E`JuliaRL_BasicDQN_CartPole` `` is a handy [command literal](https://docs.julialang.org/en/v1/manual/metaprogramming/index.html#Non-Standard-String-Literals-1) to instantiate a prebuilt experiment.}.

CartPole is considered to be one of the simplest environments for DRL (Deep Reinforcement Learning) algorithms testing. The state of the CartPole environment can be described with 4 numbers and the actions are two integers(`1` and `2`). Before game terminates, agent can gain a reward of `+1` for each step. And the game will be forced to end after 200 steps, thus the maximum reward of an episode is `200`. 

While the experiment is running, you'll see the following information and a progress bar. The information may be slightly different based on your platform and your current working directory. Note that the first run would be slow. On a modern computer, the experiment should be finished in a minute.

```julia:./display_JuliaRL_BasicDQN_CartPole_1
#hideall
using ReinforcementLearning
e = E`JuliaRL_BasicDQN_CartPole`
print(e.description)
```

\output{./display_JuliaRL_BasicDQN_CartPole_1}

```julia:./display_JuliaRL_BasicDQN_CartPole_2
#hideall
println(e)
```

Follow the instruction above and run `tensorboard --logdir /the/path/shown/above`, then a link will be prompted (typically it's `http://YourHost:6006/`). Now open it in your browser, you'll see a webpage similar to the following one:

\dfig{page;tensorboard_demo.png;Here two important variables are logged: training **loss** per update and total **reward** of each episode during training. As you can see, our agent can reach the maximum reward after training for about 4k steps.}

## Exercise

Now that you already know how to run the experiment of [BasicDQN](https://juliareinforcementlearning.org/ReinforcementLearning.jl/latest/rl_zoo/#ReinforcementLearningZoo.BasicDQNLearner) algorithm with the CartPole environment. You are suggested to try some other experiments below to compare the performance of different algorithms \footnote{For the full list of supported algorithms, please visit [ReinforcementLearningZoo.jl](https://github.com/JuliaReinforcementLearning/ReinforcementLearningZoo.jl#list-of-built-in-experiments)}:

\aside{Note that the parameters in the experiments listed here are tuned.}

- ``E`JuliaRL_BasicDQN_CartPole` ``
- ``E`JuliaRL_DQN_CartPole` ``
- ``E`JuliaRL_PrioritizedDQN_CartPole` ``
- ``E`JuliaRL_Rainbow_CartPole` ``
- ``E`JuliaRL_IQN_CartPole` ``
- ``E`JuliaRL_A2C_CartPole` ``
- ``E`JuliaRL_A2CGAE_CartPole` ``
- ``E`JuliaRL_PPO_CartPole` ``

## Basic Components

Now let's take a closer look at what's in an experiment.

\output{./display_JuliaRL_BasicDQN_CartPole_2}

In the highest level, each experiment contains the following four parts:

- [Agent](#agent)
- [Environment](#environment)
- [Hook](#hook)
- [Stop Condition](#stop_condition)

\dfig{body;agent_env.png;The relation between **agent** and **env**. The agent takes in an environment and feed an action back. This process repeats until a stop condition meets. In each step, the agent needs to improve its policy in order to maximize the expected total reward.}

When executing ``run(E`JuliaRL_BasicDQN_CartPole`)``, it will be dispatched to `run(agent, env, stop_condition, hook)`. So it's just the same as running the following lines:

\aside{[Multiple Dispatch](https://docs.julialang.org/en/v1/manual/methods/) is fully utilized in this package. And it's the secret of high extensibility.}

```julia
experiment     = E`JuliaRL_BasicDQN_CartPole`
agent          = experiment.agent
env            = experiment.env
stop_condition = experiment.stop_condition
hook           = experiment.hook

run(agent, env, stop_condition, hook)
```

Now let's explain these components one by one.

### Agent

In a nutshell, agent is a functional object which takes in an environment and returns an action. That's all.

```julia
agent = experiment.agent
env = experiment.environment
action = agent(env)
```

In the above experiment, we created an agent of type [`Agent`](https://juliareinforcementlearning.org/ReinforcementLearning.jl/latest/rl_core/#ReinforcementLearningCore.Agent), which is the most common and the default agent in this package. Inside of the agent, a [`BasicDQNLearner`](https://juliareinforcementlearning.org/ReinforcementLearning.jl/latest/rl_zoo/#ReinforcementLearningZoo.BasicDQNLearner) is used to estimate the state-action value. Here we can modify it in-place to change some parameters.

\aside{Remember to install Plots by `] add Plots` first.}

```julia:./ex2
using ReinforcementLearning # hide
using Plots

experiment = E`JuliaRL_BasicDQN_CartPole`
experiment.agent.policy.learner.γ = 0.98
hook = TotalRewardPerEpisode()

run(experiment.agent, experiment.env, experiment.stop_condition, hook)
plot(hook.rewards)
savefig(joinpath(@OUTPUT, "reward_gamma.svg"))  # hide
```

\dfig{body;reward_gamma.svg;Total reward of each episode during training with $\gamma = 0.98$.}

Try to change some other parameters in `agent.policy.learner` and see how the rewards are affected.

### Environment

In this package, many different kinds of environments are supported. Here we use the [CartPoleEnv](https://juliareinforcementlearning.org/ReinforcementLearning.jl/latest/rl_envs/#ReinforcementLearningEnvironments.CartPoleEnv-Tuple{}) to demonstrate some common methods that an environment should implement:

```julia:./env_cart_pole
using ReinforcementLearning # hide
env = CartPoleEnv()
show(stdout, MIME"text/plain"(), env)  # hide
```

You can see the summary of the `CartPoleEnv` as below:

\output{./env_cart_pole}

Some commonly used methods are:

```julia
reset!(env)                  # reset env to the initial state
get_state(env)               # get the state from environment, usually it's a tensor
get_reward(env)              # get the reward since last interaction with environment
get_terminal(env)            # check if the game is terminated or not
env(rand(get_actions(env)))  # feed a random action to the environment
```

You may also read the detailed description for [how to write a customized environment](http://juliareinforcementlearning.org/guide/#how_to_write_a_customized_environment).

### Hook

A hook is usually used to collect experiment data or modify agent/env while running. You can check the list of provided hooks [here](https://juliareinforcementlearning.org/ReinforcementLearning.jl/latest/rl_core/#Hooks-1). Two common hooks are [`TotalRewardPerEpisode`](https://juliareinforcementlearning.org/ReinforcementLearning.jl/latest/rl_core/#ReinforcementLearningCore.TotalRewardPerEpisode) and [`StepsPerEpisode`](https://juliareinforcementlearning.org/ReinforcementLearning.jl/latest/rl_core/#ReinforcementLearningCore.StepsPerEpisode).

```julia:./ex1
using ReinforcementLearning # hide
using Plots # hide

experiment = E`JuliaRL_BasicDQN_CartPole`
hook = TotalRewardPerEpisode()
run(experiment.agent, experiment.env, experiment.stop_condition, hook)
plot(hook.rewards)
savefig(joinpath(@OUTPUT, "episode.svg")) # hide
```

\dfig{body;episode.svg;Total reward of each episode during training.}

Still wondering how is the tensorboard logging generated? Learn [how to use tensorboard](https://juliareinforcementlearning.org/guide/#how_to_use_tensorboard) and [how to write a customized hook](https://juliareinforcementlearning.org/guide/#how_to_write_a_customized_hook).


### Stop Condition

A stop condition is used to determine when to stop an experiment. Two typical ones are [`StopAfterStep`](https://juliareinforcementlearning.org/ReinforcementLearning.jl/latest/rl_core/#ReinforcementLearningCore.StopAfterStep) and [`StopAfterEpisode`](https://juliareinforcementlearning.org/ReinforcementLearning.jl/latest/rl_core/#ReinforcementLearningCore.StopAfterEpisode). As you may have seen, the above experiment uses `StopAfterStep(10_000)` as the stop condition. Try to change the stop condition and see if it works as expected.

```julia
experiment = E`JuliaRL_BasicDQN_CartPole`
run(experiment.agent, experiment.env, StopAfterEpisode(100), experiment.hook)
```

Check out the full list of available stop conditions [here](https://juliareinforcementlearning.org/ReinforcementLearning.jl/latest/rl_core/#Stop-Conditions-1). You can also learn [how to write a customized stop condition](https://juliareinforcementlearning.org/guide/#how_to_write_a_customized_stop_condition).

## What's Next?

Now you are familiar with some basic concepts in ReinforcementLearning.jl, you are encouraged to read the [guide](/guide) section to have a better understanding of how each component is implemented and composed. In the [blog](/blog) section, we'll share some details of how algorithms in this package are implemented.
