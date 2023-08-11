## Example commands

To use offline data, MMD_KL loss and RBF with lengthscale as 10
```shell
python offline_encode_gp_prior.py loss=mmd_kl mmd.lengthscale=10.
```

To use offline data, MMD_KL loss and RationalQuadratic with lengthscale as 10 and alpha as 2
```shell
python offline_encode_gp_prior.py loss=mmd_kl kernel@mmd=rational_quadratic mmd.lengthscale=4. mmd.alpha=6
```
KL scaling changed:
```shell
python encode_gp_prior.py loss=mmd_kl kernel@mmd=rational_quadratic mmd.lengthscale=1. mmd.alpha=1 loss.kl_scaling=1e-2
```

To use discrete lengthscales and on the fly data generation
```shell
python encode_gp_prior.py "discrete_ls=[0.1, 0.4, 0.8, 1.0]"
```

TwoHeadDecoder with NLLAndKL loss
```shell
loss=nll_kl model/decoder=mlp_decoder_two_heads
```