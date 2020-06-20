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

Well, though it's not easy to fully understand all the functionalities, an agent simply contains two parts:

- **Policy**
- **Trajectory**

### Policy

Similar to agent, a policy is also *a functional object which takes in an observation and returns an action*.

\aside{In Reinforcement Learning, people usually like to use the character $\pi$ to represent policy.}

*"What? So why not just call it agent?"*

Because in our design, policy is a more low-level concept compared to agent. You can think of agent as a midware. It passes observations and actions between policy and environment. In the meanwhile, an agent will record some useful data into **trajectory** \footnote{People usually call it experience replay buffer. However, we choose the name of *trajectory* here because it is more general. It can be used to store not only experiences but also some intermediate data.} and generate training data to update policy at appropriate time.

The simplest policy is [`RandomPolicy`](https://juliareinforcementlearning.org/ReinforcementLearning.jl/latest/rl_base/#ReinforcementLearningBase.RandomPolicy). It doesn't need to be updated at all.

```julia
using ReinforcementLearning
env = CartPoleEnv()
p = RandomPolicy(env)
a = p(observe(env))
```

Another more common policy is [`QBasedPolicy`](https://juliareinforcementlearning.org/ReinforcementLearning.jl/latest/rl_core/#ReinforcementLearningCore.QBasedPolicy). It first maps an observation into action values via a Q `learner`, then an `explorer` is applied to get the final action. One of the most common explorer is [`EpsilonGreedyExplorer`](https://juliareinforcementlearning.org/ReinforcementLearning.jl/latest/rl_core/#Explorers-1). For a full list of available explorers, please visit the [doc](https://juliareinforcementlearning.org/ReinforcementLearning.jl/latest/rl_core/#Explorers-1).

\dfig{body;q_based_policy.png;Maze}

### Trajectory

As we mentioned above, a trajectory is used to store some useful data during interactions between policy and environment. Different trajectories have different implementations internally for efficiency or some specific scenario. But they all have the following methods implemented:

```julia
t = CircularCompactSARTSATrajectory(;capacity=3)
push!(t; state=1, action=1, reward=0., terminal=false, next_state=2, next_action=2)
get_trace(t, :state)
empty!(t)
```

### A concrete example

The following code constructs an agent to use the DQN algorithm. It is the same with the one used in the `JuliaRL_BasicDQN_CartPole` experiment.

## How to write a customized stop condition?

## How to use TensorBoard?

## How to write a customized hook?

## How to write a customized environment?