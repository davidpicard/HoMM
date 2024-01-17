# High order Moment Models

We propose an alternative to classical attention that scales linearly with the number of tokens and is based on high order moments.

![homm scheme](https://github.com/davidpicard/HoMM/blob/master/misc/homm.png?raw=true)

The HoMM scheme is as follows: Having a query token $x_q$ and a set of context tokens $x_c$, we first use a projection $ho$ to map each token $x_c$ to a high dimension space, where the high-order moments are computed recursively (by chunking and performing element-wise product, and then averaging over the tokens). $x_q$ is projected into the same high dimensional space with a projection $s$. The element-wise product of the two corresponds to $x_q$ _selecting_ the information it needs in the high-order moments of $x_c$. The results is then projected back to the same space as $x_q$ and added to the original tokens via a residual connection.

Currently testing on
- Vision: ImageNet classification
- Vision: Masked Auto Encoder pretraining

TODO:
- Vision: diffusion model
- NLP: sentence embedding
- NLP: next token prediction
- Graphs?
