@def title = "A Beginner's Guide to ReinforcementLearning.jl"
@def description = "From Novice to Professional"
@def is_enable_toc = true
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
        "publishedDate":"$(now())",
        "citationText":"Jun Tian, $(Dates.format(now(), "Y"))"
    }"""

Here we collect some common questions and answers to help you gain a better understanding of ReinforcementLearning.jl. After trying some prebuilt examples, people are usually interested the following questions:

- How to understand the XXX algorithm implemented in this package?
- How to apply the XXX algorithm to my problem?
- How to write a new algorithm by reusing components in this package as much as possible?

Now let's discuss them one by one. First, we'll focus on the simplest one: DQN\dcite{mnih2013playing}. Before looking into details, let's answer some general questions first.

## What is an agent?

*"So what is the definition of agent in this package?"*

As we have said in the [Get Started](/get_started#agent) section:

> agent is a functional object which takes in an observation and returns an action

*"Nonono, I mean how to understand the Agent data structure used in this package."*

Well, though it's not easy to fully understand all the functionalities for the first time, an agent simply contains two parts:

- **Policy**
- **Trajectory**

### Policy

Similar to agent, a policy is also *a functional object which takes in an observation and returns an action*.

\aside{In Reinforcement Learning, people usually like to use the character $\pi$ to represent policy.}

*"What? So why not just call it agent?"*

Because in our design, policy is a more low-level concept compared to agent. You can think of agent as an orchestrator. It passes observations and actions between policy and environment. In the meanwhile, an agent will record some useful data into **trajectory** \footnote{People usually call it experience replay buffer. However, we choose the name of *trajectory* here because it is more general. It can be used to store not only experiences but also some intermediate data.} and generate training data to update policy at appropriate time.

The simplest policy is [`RandomPolicy`](https://juliareinforcementlearning.org/ReinforcementLearning.jl/latest/rl_base/#ReinforcementLearningBase.RandomPolicy). It doesn't need to be updated at all.

```julia
using ReinforcementLearning
env = CartPoleEnv()
p = RandomPolicy(env)
a = p(observe(env))
```

Another more common policy is [`QBasedPolicy`](https://juliareinforcementlearning.org/ReinforcementLearning.jl/latest/rl_core/#ReinforcementLearningCore.QBasedPolicy). It first maps an observation into action values via a Q `learner`, then an `explorer` is applied to get the final action. One of the most common explorer is [`EpsilonGreedyExplorer`](https://juliareinforcementlearning.org/ReinforcementLearning.jl/latest/rl_core/#Explorers-1). For a full list of available explorers, please visit the [doc](https://juliareinforcementlearning.org/ReinforcementLearning.jl/latest/rl_core/#Explorers-1).

\dfig{body;q_based_policy.png;A visual explanation of [`QBasedPolicy`](https://juliareinforcementlearning.org/ReinforcementLearning.jl/latest/rl_core/#ReinforcementLearningCore.QBasedPolicy) in a maze environment.}

### Trajectory

As we mentioned above, a trajectory is used to store some useful data during interactions between policy and environment. Different trajectories have different implementations internally for efficiency or some specific scenario. But they all have the following methods implemented:

```julia
t = CircularCompactSARTSATrajectory(;capacity=3)
push!(t; state=1, action=1, reward=0., terminal=false, next_state=2, next_action=2)
get_trace(t, :state)
empty!(t)
```

\dfig{page;trajectory.png;A visual explanation of [`CircularCompactSARTSATrajectory`](https://juliareinforcementlearning.org/ReinforcementLearning.jl/latest/rl_core/#ReinforcementLearningCore.CircularCompactSARTSATrajectory)}

### A concrete example

The following code constructs an agent to use the [`BasicDQNLearner`](https://juliareinforcementlearning.org/ReinforcementLearning.jl/latest/rl_zoo/#ReinforcementLearningZoo.BasicDQNLearner). It is the same with the one used in the `JuliaRL_BasicDQN_CartPole` experiment.

```julia
env = CartPoleEnv(; T = Float32, seed = 11)
ns, na = length(rand(get_observation_space(env))), length(get_action_space(env))
agent = Agent(
    policy = QBasedPolicy(
        learner = BasicDQNLearner(
            approximator = NeuralNetworkApproximator(
                model = Chain(
                    Dense(ns, 128, relu; initW = seed_glorot_uniform(seed = 17)),
                    Dense(128, 128, relu; initW = seed_glorot_uniform(seed = 23)),
                    Dense(128, na; initW = seed_glorot_uniform(seed = 39)),
                ) |> cpu,
                optimizer = ADAM(),
            ),
            batch_size = 32,
            min_replay_history = 100,
            loss_func = huber_loss,
            seed = 22,
        ),
        explorer = EpsilonGreedyExplorer(
            kind = :exp,
            ϵ_stable = 0.01,
            decay_steps = 500,
            seed = 33,
        ),
    ),
    trajectory = CircularCompactSARTSATrajectory(
        capacity = 1000,
        state_type = Float32,
        state_size = (ns,),
    ),
)
```

\aside{You might have noticed that [Flux.jl](https://github.com/FluxML/Flux.jl) is used here to build the deep learning model. With the abstraction layer of `Approximator`, we can replace Flux.jl with Knet.jl or even PyTorch or TensorFlow.}

In the construction part of `BasicDQNLearner`, a [`NeuralNetworkApproximator`](https://juliareinforcementlearning.org/ReinforcementLearning.jl/latest/rl_core/#ReinforcementLearningCore.NeuralNetworkApproximator) is used to estimate the Q value. The core algorithm part is implemented in the learner. The `BasicDQNLearner` accepts an observation from environment and returns state-action values.

## How to write a customized environment?

In short, implement the following methods for single-agent synchronous environments:

```julia
get_action_space(env::YourEnv)  # -> AbstractSpace
reset!(env::YourEnv)            # -> nothing
(env::YourEnv)(action)          # -> nothing
observe(env::YourEnv)           # -> NamedTuple{(:state, :reward, :terminal)}
```

For more complicated environments, you may refer those implemented in [ReinforcementLearningEnvironments.jl](https://github.com/JuliaReinforcementLearning/ReinforcementLearningEnvironments.jl).

## How to write a customized stop condition?

Stop condition is just a function which is executed after interacting environment and returns a bool value indicating whether to stop an experiment or not.

```julia
function hook(agent, env, obs)::Bool
    # ...
end
```

Usually a closure or a functional object will be used to store some intermediate data.

## How to write a customized hook?

In most cases, you don't need to write a customized hook. Some ver general hooks are provided so that you can inject any runtime logic at appropriate time:

- [`DoEveryNStep`](https://juliareinforcementlearning.org/ReinforcementLearning.jl/latest/rl_core/#ReinforcementLearningCore.DoEveryNStep)
- [`DoEveryNEpisode`](https://juliareinforcementlearning.org/ReinforcementLearning.jl/latest/rl_core/#ReinforcementLearningCore.DoEveryNEpisode)

However, if you do need to write a customized hook, the following methods must be provided:

- `(hook::YourHook)(::PreActStage, agent, env, obs, action)`, note that there's an extra argument of `action`.
- `(hook::YourHook)(::PostActStage, agent, env, obs)`
- `(hook::YourHook)(::PreEpisodeStage, agent, env, obs)`
- `(hook::YourHook)(::PostEpisodeStage, agent, env, obs)`

If your hook is a subtype of `AbstractHook`, then all the above methods will have a default implementation which just returns `nothing`. So that you only need to extend the necessary method you want.

## How to use TensorBoard?

This package adopts a non-invasive way for logging. So you can log everything you like with a hook. For example, to log the loss of each step. You can use the [`DoEveryNStep`](https://juliareinforcementlearning.org/ReinforcementLearning.jl/latest/rl_core/#ReinforcementLearningCore.DoEveryNStep).

```julia
DoEveryNStep() do t, agent, env, obs
    with_logger(lg) do
        @info "training" loss = agent.policy.learner.loss
    end
end,
```
