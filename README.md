# InfiniGPT - Infinite Context Transformers with Infini-attention
### The implementation is inspired from the paper [Infini Transformer](https://arxiv.org/abs/2404.07143) & [dingo-actual](https://github.com/dingo-actual/infini-transformer/blob/main/readme.md)

## Abstract
The purpose of the **Infini Tranformer** is to scale the context length to infinitely long inputs but with bounded memory & computation. A concept of **Infini attention** is used along with vanilla attention mechanism. Both **masked local attention** and **long-term linear attention** is used so as to not lose context in longer inputs. A limitation of bounded memory parameters enables fast inference in LLMs.
<p align="center">
  <img src="https://github.com/ArionDas/InfiniGPT/assets/117722561/ad0fc024-2c32-4c10-9374-334c72382196">
</p>

## Motivation
Transformer-based LLMs have a constrained context-dependent memory, due to the nature of the attention mechanism :
1) Constrained, context dependent memory
2) Quadratic complexity in both memory footprint & computation time
3) Serving models with longer contexts is costly

Advantages of using **Compressive Memory** : 
1) Maintains fixed no. of parameters with limited costs
2) New info is added by changing these parameters. Note that this is done with an objective that this information can be retrived back

Our main aim is to store bindings of key & value states in the compressive memory & retrieve by using the query vectors, similar to the concept [Meta-learned Neural Memory](https://arxiv.org/abs/1907.09720)

## Methodology
This is the working of [Tranformer-XL](https://arxiv.org/abs/1901.02860)
<p align="center">
  <img src="https://github.com/ArionDas/InfiniGPT/assets/117722561/7e58f6ee-a885-4146-8443-c6b41088dc12">
</p>
The input sequence is broken down into segments to effectively tend to intricate details (refer to the paper for more detials). Notice how it discards old contexts since it caches the KV states for the last segment only.
However, using the Infini Transformer, one can carry forward the entire context history.
<br>
<p align="center">
  <img src="https://github.com/ArionDas/InfiniGPT/assets/117722561/a0ae4783-0575-4bff-8d6d-679410ec2886">
</p>

<br>
Infini attention computes both local and global context states and combines them for its output effectively reusing old KV attention states. It maintains <strong>H</strong> number of parallel compressive memory per attention layer (H is the number of attention heads)

### Compressive Memory
This is the essence of the paper. These are formulae from the paper using which a certain bounded parameters are used to compress the long context for inference.
Instead of computing new memory entries for compressive memory, the query, key and value states from the dot-product attention computation are reused. This helps in **long context adaption** & **speeds up training & inference**. A **linear attention mechanism** is used to cast the memory update and retrieveal process.
<br>
Here, 's' is the segment number. For an example, when we are at segment (let's say) 2, s-1 would mean the previous / segment 1 of tokens. The paper proposes each segment length to be 2048 tokens. You can modify it according to your input sequence length.
#### Memory Retrieval
New content (A) is retrieved from the memory (M) by using the query (Q) as : 
<br>
<p align="center">
  <img src="https://github.com/ArionDas/InfiniGPT/assets/117722561/33473aba-2aa4-4815-b0cb-ede6a275ccdb">
</p>
Here, <strong>σ</strong> and <strong>z</strong> are a nonlinear activation function and a normalization term respectively (refer to the paper to understand how the activation function is used).

#### Memory Update
The memory and the normalization terms are updated with the new KV entries and the next states are obtained as : 
<br>
<p align="center">
  <img src="https://github.com/ArionDas/InfiniGPT/assets/117722561/e9d957ee-540a-4d1c-ac76-b234ad8bc7b7">
</p>
These new memory states are then recursively passed to the next segment, s+1 (in each attention layer). A slight optimization can be performed using the <strong>Delta rule</strong> (refer to the Neural Memory paper). It attempts a slightly improved memory update by first retrieving existing value entries and subtracting them from the new values before applying the new update : 
<br>
<p align="center">
  <img src="https://github.com/ArionDas/InfiniGPT/assets/117722561/e200106c-2d79-44e5-bdf6-4d9a8cf20036">
</p>

#### Long Term Context Injection
The local attention state (Ad) & memory retrieved content (Am, i.e. from the previous segments) are aggregated using a hyperparameter, β : 
<br>
A = sigmoid(β) * Am + (1 - sigmoid(β)) * Ad
<br>
The hyperparameter, β is determined by a [learnable trade-off](https://arxiv.org/abs/2203.08913) between the long-term and local information flows in the model.
<br>

#### Complexity Analysis
Infini-Transformer has a **constant memory complexity** of **dkey × dvalue + dkey** for storing compressed context in **Ms** and **zs** for each head in a single layer. On the other hand, for the other models, their complexity increases along with the sequence dimension. In **Transformer-XL, Compressive Transformers & Memorizing Transformers**, the memory complexity depends on the cache size. In case of **RTM** & **AutoCompressors**, it depends on the **prompt size**.

#### References
Refer to [Associative Matrix](https://arxiv.org/abs/1910.06611) to get to know more about compressive memory. <br>
Refer [here](https://arxiv.org/abs/2006.16236) for more on the working of the update rule & retrieval mechanism.

### Future Work
1) Scale the architecture to multiple Infini attention layers
2) LLM pre-training on large datasets
3) Perform the Book Summarization task by finetuning the LLM on the BookSum dataset
