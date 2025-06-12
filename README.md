

# Effective Utility-Driven Spatial Segregation and Its Impact on Cooperation Evolution: A Cultural Weight-Dependent Perspective

This repository contains the source code for the agent-based model presented in the paper "Effective Utility-Driven Spatial Segregation and Its Impact on Cooperation Evolution: A Cultural Weight-Dependent Perspective" 

[](https://opensource.org/licenses/MIT)

-----

## English Introduction

### Overview

This project investigates the evolution of cooperation in a spatially structured population where individuals exhibit cultural heterogeneity. We propose a spatial evolutionary game model where each agent possesses an evolvable **"cultural weight" ($C\_k$)**. This weight dictates how an individual values their own payoff versus their neighbor's payoff when evaluating interaction outcomes.

The core mechanism is an **effective utility function** that is dependent on this cultural weight. Agents update their strategies (Cooperate or Defect) and their cultural weights by comparing their accumulated effective utility with that of their neighbors.

Our simulations demonstrate that this utility-evaluation mechanism spontaneously drives significant **spatial segregation**, where like-minded individuals cluster together. This self-organized structure, in turn, powerfully influences cooperation through a **"boundary effect"**: individuals at the interfaces of different cultural clusters exhibit significantly lower cooperation rates than those in homogeneous group interiors.

### Model Core Mechanism

The model is an agent-based simulation set on an $L \\times L$ two-dimensional square lattice with periodic boundary conditions.

1.  **Agent Attributes**: Each agent `k` has two core attributes:

      * **Strategy ($S\_k$)**: Can be either Cooperate (C) or Defect (D).
      * **Cultural Weight ($C\_k$)**: A continuous variable in $[0, 1]$ that quantifies the agent's concern for others' payoffs.
          * $C\_k=0$ represents pure self-interest (individualistic).
          * $C\_k=1$ represents pure other-regard (altruistic/collectivistic).

2.  **Interaction Game**: Agents play a pairwise Prisoner's Dilemma game with their 8 immediate neighbors (Moore neighborhood). The payoff matrix is defined as:

      * (C, C): Both get $R=1$.
      * (D, C): Defector gets $T=b$, Cooperator gets $S=0$.
      * (D, D): Both get $P=0$.
      * where $b \> 1$ is the temptation to defect.

3.  **Effective Utility Calculation**: The key innovation is that decisions are based on a culturally-weighted effective utility, not raw payoffs. For an interaction between agent `k` and neighbor `l`, the utility for agent `k` is:

    ```
    U_k = (1 - C_k) * π(S_k, S_l) + C_k * π(S_l, S_k)
    ```

    where $\\pi(S\_k, S\_l)$ is the payoff to agent `k`. An agent's total fitness in a round is the sum of these utilities from all neighbor interactions.

4.  **Evolutionary Dynamics**: Each time step consists of four phases:

    1.  **Calculate Utility**: Every agent calculates its accumulated effective utility.
    2.  **Strategy Update**: An agent `k` randomly selects a neighbor `l` and adopts their strategy with a probability given by the Fermi rule, based on the difference in their accumulated utilities.
    3.  **Culture Update**: With a probability $p\_C$, agent `k` attempts a culture update. It again selects a random neighbor `m` and adopts their cultural weight $C\_m$ based on the Fermi rule.
    4.  **Mutation**: Each agent's strategy and cultural weight can mutate with small probabilities ($p\_{s,m}$ and $p\_{c,m}$ respectively).

### How to Cite

If you use this code or model in your research, please cite our paper:

```bibtex
@article{WangChen2025,
  title   = {Effective Utility-Driven Spatial Segregation and Its Impact on Cooperation Evolution: A Cultural Weight-Dependent Perspective},
  author  = {Jinjin Wang and Yuyou Chen},

  year    = {2025},
  note    = {Preprint submitted on June 12, 2025}
}
```

### Contact

For any questions, please contact the corresponding author:
Yuyou Chen (chenyuyou@126.com)

-----

## 中文介绍

### 项目总览

本项目旨在研究具有文化异质性的空间结构群体中的合作演化问题。我们提出了一个空间演化博弈模型，其中每个主体都拥有一个可演化的\*\*“文化权重” ($C\_k$)\*\*。该权重决定了个体在评估互动结果时，对其自身收益和邻居收益的重视程度。

模型的核心机制是一个依赖于该文化权重的\*\*“效益效用函数”\*\*。主体通过比较自身与邻居的累积效益效用，来更新自己的策略（合作或背叛）和文化权重。

我们的模拟结果表明，这种效用评估机制能够自发地驱动显著的**空间隔离**，即拥有相似文化观念的个体会聚集在一起。这种自组织结构反过来又通过\*\*“边界效应”\*\*深刻地影响合作的演化：位于不同文化簇交界处的个体的合作率，显著低于同质文化簇内部的个体。

### 模型核心机制

本模型是一个基于主体的模拟，搭建在具有周期性边界条件的 $L \\times L$ 二维方形晶格上。

1.  **主体属性**: 每个主体 `k` 拥有两个核心属性：

      * **策略 ($S\_k$)**: 可选“合作 (C)”或“背叛 (D)” 。
      * **文化权重 ($C\_k$)**: 一个在 $[0, 1]$ 区间内的连续变量，用于量化个体对他人收益的关注度 。
          * $C\_k=0$ 代表纯粹的利己主义。
          * $C\_k=1$ 代表纯粹的利他主义或集体主义。

2.  **互动博弈**: 主体与其8个直接邻居（摩尔邻域）进行成对的囚徒困境博弈 。收益矩阵定义如下：

      * (C, C): 双方均获得 $R=1$。
      * (D, C): 背叛者获得 $T=b$，合作者获得 $S=0$。
      * (D, D): 双方均获得 $P=0$。
      * 其中 $b \> 1$ 是背叛的诱惑值 。

3.  **效益效用计算**: 模型的关键创新在于，决策是基于文化加权的效益效用，而非原始收益。对于主体 `k` 和邻居 `l` 之间的一次互动，主体 `k` 获得的效用为：

    ```
    U_k = (1 - C_k) * π(S_k, S_l) + C_k * π(S_l, S_k)
    ```

    其中 $\\pi(S\_k, S\_l)$ 是主体 `k` 的原始收益。一个主体在一轮中的总适应度是其与所有邻居互动所得效用的总和。

4.  **演化动力学**: 每个时间步包含四个阶段：

    1.  **计算效用**: 每个主体计算其累积的效益效用 。
    2.  **策略更新**: 主体 `k` 随机选择一个邻居 `l`，并根据两者累积效用的差异，依据费米规则（Fermi rule）概率性地学习对方的策略。
    3.  **文化更新**: 主体 `k` 以概率 $p\_C$ 尝试进行文化更新。它同样随机选择一个邻居 `m`，并依据费米规则学习对方的文化权重 $C\_m$。
    4.  **突变**: 每个主体的策略和文化权重都可能以一个微小的概率（分别为 $p\_{s,m}$ 和 $p\_{c,m}$）发生随机突变。

### 如何引用

如果您在研究中使用了本代码或模型，请引用我们的论文：

```bibtex
@article{WangChen2025,
  title   = {Effective Utility-Driven Spatial Segregation and Its Impact on Cooperation Evolution: A Cultural Weight-Dependent Perspective},
  author  = {Jinjin Wang and Yuyou Chen},

  year    = {2025},
  note    = {Preprint submitted on June 12, 2025}
}
```

### 联系方式

如有任何问题，请联系本文的通讯作者：
 (chenyuyou@126.com) 
