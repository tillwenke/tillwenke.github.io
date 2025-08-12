---
layout: post
title:  "Feature Representations for Semantic Code Retrieval (Bachelor's Thesis)"
date:   2024-09-30
author: Till Wenke
section: blog
tags:
  - ML
---
My Bachelor's thesis was part of a year-long project around LLM-driven code generation - imagine tackling [SWE-bench](https://arxiv.org/abs/2310.06770) but on our own dataset of public Java projects. In my thesis I covered the aspect of how to represent code to make it searchable.

We presented our work in an easily digestable format at our [Bachelor's podium](https://www.tele-task.de/lecture/video/10764/).

Our work was featured and we attended SAP Sapphire Barcelona 2024.


# Abstract

Semantic code retrieval describes the task of finding code based on the content of a
natural language query. More specifically, this work is concerned with repository-
level code file retrieval from natural language issue descriptions which can be part
of broader repository-level code generation pipelines. For this purpose, I apply a
Bayesian linear probit regression model to issue-code instances that are represented
by interaction features between their two components. The interaction features that
I investigate stem from textual representations of the components, the structure of
their repository and the past activity in the repository with respect to time and the
developers involved. Each of the features was chosen for alone being indicative
of a file change. I could show that present text embedding models yield a good
semantic representation of natural language as well as code and that a simple linear
model can fine-tune a text similarity metric on those embeddings to the specific
task of code retrieval. Making use of the interaction features beyond the textual
representations did not notably increase or even decreased the model’s performance
on two single-repository datasets as well as a multi-repository dataset that were
specifically compiled for this work.
