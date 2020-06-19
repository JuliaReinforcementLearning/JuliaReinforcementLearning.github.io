@def title = "A Whirlwind Tour of ReinforcementLearning.jl"
@def description = "Welcome to the world of reinforcement learning in Julia. Now let's get started in 3 lines!"
@def is_enable_toc = false
@def has_code = true

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
        "publishedDate":"2020-06-18T12:36:15.000+08:00",
        "citationText":"Jun Tian, 2020"
    }"""

@def appendix = """
    ### Corrections
    If you see mistakes or want to suggest changes, please [create an issue](https://github.com/JuliaReinforcementLearning/JuliaReinforcementLearning.github.io/issues) on the source repository.
    """

## Prepare

First things first, [download](https://julialang.org/downloads/) and install the latest stable Julia version. ReinforcementLearning.jl is tested on all platforms, so just choose the one you are familiar with. If you already have Julia installed, please make sure that it is `v1.3` or above.

\aside{ReinforcementLearning.jl relies on some features introduced since `v1.3`, like [MultiThreading](https://docs.julialang.org/en/v1/base/multi-threading/index.html), and [Artifacts](https://julialang.github.io/Pkg.jl/dev/artifacts/)}

Another useful tool is [tensorboard](https://github.com/tensorflow/tensorboard) \footnote{You don't need to install the whole TensorFlow to use the TensorBoard. Behind the scene, ReinforcementLearning.jl uses [TensorBoardLogger.jl](https://github.com/PhilipVinc/TensorBoardLogger.jl) to write data into the format that TensorBoard recoganizes.}. You can install it via `pip install tensorboard` with the python package installer [`pip`](https://pip.pypa.io/en/stable/installing/).

## Get Started

Run `julia` in the command line (or double-click the Julia executable) and now you are in an interactive session (also known as a read-eval-print loop or "REPL"). Then execute the following code:

```julia
julia> ] add ReinforcementLearning

julia> using ReinforcementLearning

julia> run(E`JuliaRL_BasicDQN_CartPole`);
```

So what's happening here?

1. In the first line, typing `]` will bring you to the *Pkg* mode. `add ReinforcementLearning` will install the latest version of `ReinforcementLearning.jl` for you. And then remember to press backspace or ^C to get back to the normal mode.
1. `using ReinforcementLearning` will bring the names exported in `ReinforcementLearning` into global scope. If this is your first time to run, you'll see *precompiling ReinforcementLearning*. And it may take a while.
1. The third line means, `run` an **E**xperiment named `JuliaRL_BasicDQN_CartPole` \footnote{The ``E`JuliaRL_BasicDQN_CartPole` `` is a handy [command literal](https://docs.julialang.org/en/v1/manual/metaprogramming/index.html#Non-Standard-String-Literals-1) to instantiate a prebuilt experiment.}.

CartPole is considered to be one of the simplest environments for DRL (Deep Reinforcement Learning) algorithms testing. The state of the CartPole environment can be described with 4 numbers and the actions are two integers(`1` and `2`). Before game terminates, agent can gain a reward of `+1` for each step. And the game will be forced to end after 200 steps, thus the maximum reward of an episode is `200`. 

While the experiment is running, you'll see the following information and a progress bar. The information may be slightly different based on your platform and your current working directory. Note that the first run would be slow. On a mordern computer, the experiment should be finished in a minute.

```julia:./run_JuliaRL_BasicDQN_CartPole
#hideall
using ReinforcementLearning
e = run(E`JuliaRL_BasicDQN_CartPole`)
println(e.description)
```

\output{./run_JuliaRL_BasicDQN_CartPole}

Follow the instruction above and run `tensorboard --logdir /the/path/shown/above` and a link will be prompted (typically it's `http://YourHost:6006/`). Now open it in your browser, you'll see a webpage like the following one:

\dfig{page;./tensorboard_demo.png;Here two important varialbes are logged: training **loss** per update and total **reward** of each episode during training. As you can see, our agent can reach the maximum reward after training for about 4k steps.}

## Exercise

Now that you already know how to run the experiment of [BasicDQN](https://juliareinforcementlearning.org/ReinforcementLearning.jl/latest/rl_zoo/#ReinforcementLearningZoo.BasicDQNLearner) algorithm with the CartPole environment. You are suggested to try some other algorithms \footnote{For the full list of supported algorithms, please visit [ReinforcementLearningZoo.jl](https://github.com/JuliaReinforcementLearning/ReinforcementLearningZoo.jl)}:

- `JuliaRL_DQN_CartPole`
- `JuliaRL_PrioritizedDQN_CartPole`
- `JuliaRL_Rainbow_CartPole`
- `JuliaRL_IQN_CartPole`
- `JuliaRL_A2C_CartPole`
- `JuliaRL_A2CGAE_CartPole`
- `JuliaRL_PPO_CartPole`
- `JuliaRL_DDPG_Pendulum`

## Basic Components

Every experiment at least contains the following two components: **agent** and **env**.

\dfig{body;./agent_env_relation.png;The relation between **agent** and **env**. The agent takes in an observation from environment and feed an action back to the environment. This process repeats until a stop condition meets. During this time, the agent needs to improve its policy in order to maximise the expected total reward.}

First

```julia:./test1
a = 1
@show a
```

\output{./test1}

```julia:./test2
b = a + 1
@show b
```

\output{./test2}