

# MM-HELIX: Boosting Multimodal Long-Chain Reflective Reasoning with Holistic Platform and Adaptive Hybrid Policy Optimization


<!-- [![Paper](https://img.shields.io/badge/Paper-Arixv-orange)](#)
[![Tasks](https://img.shields.io/badge/Project--blue)](#)
[![Instances](https://img.shields.io/badge/Instances-1%2C260-blue)](#)
[![Dataset](https://img.shields.io/badge/CoT-100k-green)](#) -->


<p align="center">
        🌐 <a href="">HomePage</a>&nbsp&nbsp | &nbsp&nbsp📖 <a href="">Paper</a>&nbsp&nbsp | &nbsp&nbsp🏆 <a href="">Leaderboard</a>
<br>
🤗 <a href="https://huggingface.co/PhoenixZ/MM-HELIX-7B-Thinking">MM-HELIX-7B-Thinking</a>&nbsp&nbsp | &nbsp&nbsp🤗 <a href="">MM-HELIX Benchmark</a>&nbsp&nbsp | &nbsp&nbsp🤗 <a href="">MM-HELIX-100K</a>
</p>


<p align="center">
  <img width="100%" src="images/Teaser09241052.png">
</p>

## Introduction
**MM-HELIX** is a holistic platform to **evaluate** and **improve** the *long-chain reflective reasoning* capability of Multimodal LLMs (MLLMs). It includes an RL environment integrating **generator**, **verifier**, and **SERG pipeline**, built upon **42 game / algorithm / puzzle / graph tasks**. It supports **task generation, automatic verification, and reflective CoT synthesis**, enabling scalable training and evaluation of long-chain reflective reasoning.

Based on things above, our work contains:

* **MM-HELIX (Benchmark)** : 42 diverse tasks × 5 difficulty levels = **1,260** multimodal instances; auto **Verifier** ensures objective scoring and reward.
* **MM-HELIX-100K** : a **100k** sample dataset of high-quality, reflective **Chain-of-Thought (CoT)** traces produced via **SERG** (Step-Elicited Response Generation).
* **AHPO** : **Adaptive Hybrid Policy Optimization** that unifies off-policy supervision and on-policy exploration in one stage to overcome sparse rewards & forgetting.

> Using **Qwen2.5-VL-7B** as the base, **AHPO** yields **+18.6%** absolute accuracy on MM-HELIX and **+5.7%** average gain across general math/logic benchmarks!




## To-Do List
<!-- We will **progressively release** these key resources to the community: -->

-  **Arxiv Paper** [✅]
- **MM-HELIX-100K** [✅]
- **MM-HELIX-7B-Thinking Checkpoint** [✅]
- **MM-HELIX Benchmark** [✅]
- **Training Code** [*Coming Soon*]
- **Evaluation Code** [*Coming Soon*]
- **SandBox Code** [*Coming Soon*]

<!-- - **SERG Pipline Code** [*Coming Soon*] -->


<!-- ## Table of Contents

* [1. Motivation & Summary](#1-motivation--summary)
* [2. Quick Start](#2-Quick-Start)
* [3. Adaptive Hybrid Policy Optimization (AHPO)](#2-whats-inside)
* [4. MM-HELIX-Benchmark & Leaderboard](#2-whats-inside)
* [5. MM-HELIX-100K](#2-whats-inside)
* [6. MM-HELIX-SandBox](#2-whats-inside)
* [7. Citation](#2-whats-inside) -->


<!-- ## 1. Motivation & Summary

* Current MLLMs are good at *one-shot* math/logic but struggle with **iterative reflection** (plan → act → check → backtrack) in rich visual contexts.
* **MM-HELIX** addresses the gap with:

  1. A **comprehensive benchmark** stressing *long-chain reflective reasoning*;
  2. A **scalable data engine(SERG) and RL environment(MM-HELIX-SandBox)to build *reflective CoTs*;
  3. **AHPO** to *learn* and *generalize* reflective skills efficiently. -->

<!-- ## 2. Quick Start

## 3. Adaptive Hybrid Policy Optimization (AHPO)

## 4. MM-HELIX-Benchmark & Leaderboard

## 5. MM-HELIX-100K

## 6. MM-HELIX-SandBox -->

## Citation
