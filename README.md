## Nano GPT-jax
An implementation of nanogpt in jax from scratch ( Other than Optax for optimization and Equinox for handling PyTrees ) based on Andrej Karpathy's <a href="https://www.youtube.com/watch?v=kCc8FmEb1nY">Let's build GPT</a> Lecture.

## Usage
- The Shakespeare dataset is in `data` folder. You only need to configure hyper-parameters in `nanogpt-jax/train.py` as per your test settings and then run :
```
$ python train.py
```

## TODOS

- [x] Write DropOut Layers. 
- [x] LayerNorm.
- [x] Apply weight initializers.
- [ ] Implement Adam.

## References
- Andrej Karpathy's <a href="https://www.youtube.com/watch?v=kCc8FmEb1nY">Let's build GPT</a> Lecture.
- <a href="https://sjmielke.com/jax-purify.htm"> From PyTorch to JAX: towards neural net frameworks that purify stateful code </a>.
- For my usecase I did not want to use Haiku or Flax. I wanted something very mimimal. And I found Equinox suitable. I got introduced to Equinox through this <a href="https://github.com/lucidrains/PaLM-jax"> Repo </a>  by Phil Wang.
