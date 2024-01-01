# CACm
Chaotic Amplitude Control with momentum - curator: Timothee Leleu

# Installation

## local install

python -m venv CACm \\
source CACm/bin/activate \\
git clone https://github.com/NTTRI-PHI-Algorithms/CACm.git ./ \\
pip install . \\
python ./CACm/models_pyTorch.py \\


# Description

## Introduction

Chaotic Amplitude Control is an algorithm used to find the low-energy states of the Ising Hamiltonian $\mathcal{H}(\boldsymbol{\sigma}) = \frac{1}{2} \boldsymbol{\sigma}^T \Omega \boldsymbol{\sigma}$ with $\boldsymbol{\sigma} \in \{-1,1\}^N$ and $\Omega$ the instance-dependent Ising coupling. The algorithm is based on a relaxation of the binary spins $\boldsymbol{\sigma}$ to soft spins $\boldsymbol{x}$. The dynamics of the soft spins are composed of a gradient descent $\nabla_{\boldsymbol{x}} \mathcal{H}$ of the potential $\mathcal{H}(\boldsymbol{x})$ and the correction of amplitude heterogeneity imposed by the auxiliary variables $\boldsymbol{e}$ (see [1,2]). In addition, a momentum term is added in this version.

## Benchmark

The algorithm is tested on Wishart planted instances. The evaluation metric is the time-to-solution (TTS) defined as TTS = $\frac{\text{log}(1-0.99)}{\text{log}(1-p_0)} T$ where $p_0$ is the probability of finding the ground-state.

## Model

$$
\boldsymbol{x} (t+1) = \boldsymbol{x} (t) + \Delta t (- \beta(t) \boldsymbol{x} (t) + \alpha \boldsymbol{e} (t) \circ (\Omega \boldsymbol{\phi}(t)) + \gamma (\boldsymbol{x} (t)-\boldsymbol{x} (t-1)) ),\\
\boldsymbol{e} (t+1) = \boldsymbol{e} (t) - \xi \boldsymbol{e} (t) \circ (\boldsymbol{x} (t)^2-1)
$$

with

$$
\frac{1}{N} \sum_i e_i(t) = 1, \forall t,\\
\beta = \beta_1 + \frac{t}{T}(\beta_2-\beta_1),
$$

and 

$$
\phi_i(t) = \text{tanh}(x_i(t)), \forall i.
$$

## Parameters

| Parameter | Interpretation |
| --------------- | --------------- |
| $T$          | Number of time steps         |
| $\beta_1$         | Initial decay rate          |
| $\beta_2$         | Final decay rate          |
| $\alpha$         | Coupling strength          |
| $\gamma$         | Momentum term strength          |
| $\xi$         | Rate of change of auxiliary variables          |
| $\Delta$         | Time step size          |

## Pseudo-code

```
FOR t IN 0..T-1
    Set beta to beta1 + t/T*(beta2-beta1)   
    Set xpp to xp
    Set xp to x
    Set y to tanh(xp)
    Set mu to w @ y           #Matrix-matrix multiplication
    Set x to xp + Dt*( -beta*xp + alpha*e*mu + gamma*(xp-xpp)  )
    Set e to e - (xp**2-1.0)*e*xi
    Set e to e/mean(e)
    Set s to sign(x)
    Set mu0 to w @ s
    Set H to -0.5 sum(s * mu0)
END FOR
```

## References

[1] Leleu, Timothée, et al. "Destabilization of local minima in analog spin systems by correction of amplitude heterogeneity." Physical review letters 122.4 (2019): 040607.

[2] Leleu, Timothée, et al. "Scaling advantage of chaotic amplitude control for high-performance combinatorial optimization." Communications Physics 4.1 (2021): 266.
