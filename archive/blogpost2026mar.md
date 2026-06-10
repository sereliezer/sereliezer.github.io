---
title: "Divergence Training for Generative Models"
date: 2026-03-13
author: "Eliezer de Souza da Silva"
tags: ["divergence measures", "gflownet", "ML theory"]
categories: ["blog"]
math: true
---

### Divergence Training for Generative Models  
#### 0. Loss Design, Policy Gradients, and On/Off-Policy Optimization

Over the last few years, several machine learning communities have converged on a common mathematical structure underlying many training algorithms:

- **Generative Flow Networks (GFlowNets)**
- **Reinforcement Learning**
- **Diffusion models**
- **Language model alignment**
- **General generative model training**

Despite appearing very different on the surface, these methods often optimize objectives that can be interpreted as **minimizing divergences between distributions**.

In this post, we explore how these connections arise and why they matter. In particular, we discuss:

- the role of **$f$-divergences**
- how **loss functions correspond to divergences**
- the relationship between **policy gradients and divergence minimization**
- and how **on-policy vs off-policy optimization** shapes these objectives.

---

#### 1. The score-function identity

A central mathematical tool in probabilistic learning is the **score-function identity**.

For a parameterized distribution $p_\theta(x)$ and a function $f(x)$,

$$
\nabla_\theta {E}_{x\sim p_\theta}[f(x)] = E_{x\sim p_\theta} \left[ f(x)\nabla_\theta \log p_\theta(x) \right].
$$

This identity appears in:

- REINFORCE
- policy gradient algorithms
- generative model training
- variational inference.

It shows that gradients of expectations can be estimated using **weighted log-likelihood gradients**.


#### 2. Divergences between distributions

Suppose we want to match a model distribution $p_\theta(x)$ to a target distribution $q(x)$.

A common way to measure the mismatch is through an **$f$-divergence**:

$$
D_f(p \Vert q) = E_{x\sim q} \left[ f\left(\frac{p(x)}{q(x)}\right) \right].
$$

Different choices of $f$ produce familiar divergences:

| divergence | $f(u)$ |
|---|---|
KL | $u\log u$ |
reverse KL | $-\log u$ |
$\chi^2$ | $(u-1)^2$ |
Jensen–Shannon | mixture of KLs |

Each divergence induces different training dynamics.



#### 3. Gradient of an $f$-divergence

Let

$$
u(x)=\frac{p_\theta(x)}{q(x)}.
$$

The gradient of the divergence is

$$
\nabla_\theta D_f(p_\theta \Vert q) = E_{x\sim p_\theta} \left[ f'(u(x)) \nabla_\theta \log p_\theta(x) \right].
$$

This equation reveals a key insight:

> **Minimizing divergences produces policy-gradient-style updates.**

The update resembles reinforcement learning with reward

$$
R(x)=f'(u(x)).
$$



#### 4. Three perspectives that rediscover the same structure

Researchers have arrived at this structure from several directions.

##### 4.1. Divergence-first

Start with

$$
\min_\theta D_f(p_\theta \Vert q)
$$

This viewpoint appears in work on **divergence training of GFlowNets**.


##### 4.2. Loss-first

Start with a surrogate loss on log probability differences:

$$
g(\log p_\theta(x)-\log q(x)).
$$

Then derive which divergence this loss implicitly minimizes.

For example:

$$
f(t) = t\int_1^t \frac{g'(\log s)}{s^2}ds
$$

and

$$
g(t) = f(e^t) - \int_1^{e^t}\frac{f(s)}{s}ds.
$$

This establishes a **loss–divergence correspondence**.

##### 4.3. Policy-gradient viewpoint

In reinforcement learning,

$$
\nabla_\theta J(\theta) = E_{x\sim p_\theta} \left[ R(x)\nabla_\theta \log p_\theta(x) \right].
$$

Choosing

$$
R(x)=f'(u(x))
$$

recovers the divergence gradient.

This insight leads to **$f$-policy gradients**.


#### 5. On-policy vs off-policy optimization

The distinction between **on-policy** and **off-policy** optimization plays a crucial role in divergence-based training.

##### 5.1 On-policy gradients

In on-policy optimization, samples are drawn from the current model distribution:

$$
x\sim p_\theta(x).
$$

The gradient estimator becomes

$$
E_{x\sim p_\theta} \left[ w(x)\nabla_\theta \log p_\theta(x) \right].
$$

This estimator is **unbiased** but can have high variance.

Many divergence-based methods rely on this form.


##### 5.2 Off-policy gradients

In off-policy optimization, samples come from another distribution $q(x)$.

To correct for the mismatch we use **importance weights**:

$$
E_{x\sim q} \left[ \frac{p_\theta(x)}{q(x)}w(x)\nabla_\theta \log p_\theta(x) \right].
$$

This introduces two challenges:

1. **variance explosion** due to importance weights  
2. **support mismatch** between distributions.

These issues are central in both reinforcement learning and GFlowNets.


#### 6. Trajectory Balance as divergence training

In GFlowNets, the goal is to learn

$$
p_\theta(x)\propto R(x).
$$

Trajectory Balance introduces the loss

$$
(\log Z + \log P_F - \log P_B - \log R)^2.
$$

This loss operates on **log probability ratios**, which makes it compatible with divergence interpretations.

Recent work shows that many such losses correspond to minimizing divergences between forward and backward trajectory distributions.


#### 7. Why off-policy training is tricky

Off-policy training introduces subtle issues.

If samples are drawn from a replay buffer or proposal distribution $q(x)$, then the gradient becomes

$$
\mathbb{E}_{x\sim q}
\left[
\frac{p_\theta(x)}{q(x)}f'(u(x))\nabla_\theta \log p_\theta(x)
\right].
$$

This creates two practical problems:

##### A) importance weight variance

The ratio

$$
\frac{p_\theta(x)}{q(x)}
$$

can become extremely large.


##### B) support mismatch

If

$$
q(x)=0
$$

for regions where

$$
p_\theta(x)>0
$$

then the divergence gradient becomes undefined.

This issue is often overlooked in theoretical analyses but becomes critical in practice.

#### 8. The equivalence class of losses

Another subtle observation is that **many losses induce the same gradient**.

Because

$$
\mathbb{E}_{p_\theta}[\nabla_\theta \log p_\theta(x)] = 0,
$$

adding certain terms to a loss does not change its expected gradient.

For example

$$
L(\Delta)
$$

and

$$
L(\Delta)+c\Delta
$$

can produce identical updates.

Thus there exists a **family of equivalent losses** corresponding to the same divergence.

---

#### 9. A unifying perspective

We can now summarize the relationships.

| starting point | resulting update |
|---|---|
divergence minimization | policy-gradient update |
policy gradient | divergence gradient |
loss design | divergence minimization |

At their core, these methods optimize gradients of the form

$$
\mathbb{E}_{x\sim p_\theta}
\left[
w(x)\nabla_\theta \log p_\theta(x)
\right].
$$

---

#### 10. Open questions

Several theoretical questions remain.

- Characterizing loss equivalence: which losses produce identical divergence gradients?
- Variance and optimization: different losses may produce the same expected gradient but very different variance properties.
- Off-policy stability: how can we design divergence objectives that remain stable under off-policy sampling?

---

#### Final thoughts

Divergence training provides a powerful unifying perspective on modern generative model training.

Many algorithms that appear unrelated are actually optimizing the same mathematical objects from different angles.

Understanding this structure can help us:

- design better objectives
- stabilize training
- and unify ideas across machine learning fields.


#### References

- **Agarwal et al. (2023)** — *f-Policy Gradients: A General Framework for Goal-Conditioned RL using f-Divergences*. NeurIPS 2023.  
  [OpenReview](https://openreview.net/forum?id=EhhPtGsVAv&noteId=Yi1UezNKJP)

- **Hu et al. (2025)** — *Beyond Squared Error: Exploring Loss Design for Enhanced Training of Generative Flow Networks*. ICLR 2025.  
  [OpenReview](https://openreview.net/forum?id=4NTrco82W0)

- **Silva, Silva, Mesquita (2024)** — *On Divergence Measures for Training GFlowNets*. NeurIPS 2024.   
 [OpenReview](https://openreview.net/forum?id=N5H4z0Pzvn)

- **Malkin et al. (2022)** — *Trajectory Balance: Improved Credit Assignment in GFlowNets*. NeurIPS 2022.  
  [Arxiv](https://arxiv.org/abs/2201.13259)

- **Nowozin et al. (2016)** — *f-GAN: Training Generative Neural Samplers using Variational Divergence Minimization*. NeurIPS 2016.  
[Arxiv](https://arxiv.org/abs/1606.00709)
