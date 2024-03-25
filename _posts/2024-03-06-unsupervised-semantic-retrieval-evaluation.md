---
layout: post
title:  "Unsupervised Evaluation of Semantic Retrieval by Generating Relevance Judgments with an LLM Judge"
date:   2024-03-21
author: Till Wenke, Fabian Bergmann
section: blog
---
I summarized the work on semantic retrieval that I did during my internship at [ML6](https://www.ml6.eu/) in a blog post.
Download the full article [here](https://raw.githubusercontent.com/tillwenke/tillwenke.github.io/main/_posts/assets/ml6_blog_post.pdf).
See [here](https://blog.ml6.eu/unsupervised-evaluation-of-semantic-retrieval-by-generating-relevance-judgments-with-an-llm-judge-ea244cc80908) for the original post.

# Abstract

Our results reveal that LLMs are capable of detecting general performance trends. The information retrieval metrics that were computed from the LLM judgments strongly correlate with those that were computed from human relevance labels - we can report a 0.91 Pearson correlation coefficient. In general, the ranking of embedding models by retrieval performance using the classic supervised evaluation approach is mostly reproduced by our proposed unsupervised approach. Thereby, the LLM's judgments are more aligned with the ground truth human labels when making use of  Q/A pairs. We believe the information of the desired answer adds valuable steering information for the LLM judge. Conducting retriever model selection with an LLM judge is therefore a viable option in an un-/semi-supervised setting.
