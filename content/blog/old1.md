---
title: "(old post, test) Sampling from Dirichlet Distribution using Gamma Distributed Samples"
date: 2016-03-13
author: "Eliezer de Souza da Silva"
tags: ["old blog"]
categories: ["blog"]
math: true
---


There is an algorithm to generate Dirichlet samples using a sampler for Gamma distribution for any \\( \alpha > 0 \\) and \\( \beta > 0 \\). We will generate Gamma distributed variables \\( z_k \sim \text{gamma}(\alpha_k,1) \\), for \\( k \in \{1,\cdots,d\} \\), and do the following variable transformation to get Dirichlet samples \\( x_k = \frac{z_k}{\sum_k z_k} \\). First, we should demonstrate that this transformation results in Dirichlet distributed samples.

Consider the following transformation \\( (z_1,\cdots,z_d) \leftarrow (x_1,\cdots,x_d,v) \\), where \\( x_k = \frac{z_k}{\sum_k z_k} \\) and \\( v = {\sum_k z_k} \\). We can rewrite this transformation as \\( (x_1,\cdots,x_d,v)=h(z_1,\cdots,z_d) \\), where \\( x_k = \frac{z_k}{v} \\) and \\( v = {\sum_k z_k} \\). Also, we can immediately calculate the inverse transformation \\( (z_1,\cdots,z_d)=h^{-1}(x_1,\cdots,x_d,v) \\), with \\( z_k=v x_k \\). From the transformation definition, we know that \\( {\sum_{k=1}^d x_k=1} \\), implying that \\( x_d = 1-\sum_{k=1}^{d-1} x_k \\) and \\( z_d=v(1-\sum_{k=1}^{d-1}x_k) \\).

Now we can apply the formula for transformation of variables to \\( z_{1:d} \\), \\( x_{1:(d-1)} \\) and \\( v \\) (using the notation that \\( f_X \\) is the pdf of the random variables \\( (x_{1:(d-1)},v) \\) and \\( f_Z \\) is the pdf of the random variables \\( z_{1:d} \\)):

$$
f_{Z}(z_{1:d}) =\prod_{k=1}^d \frac{z_k^{\alpha_k-1} e^{-z_k}}{\Gamma(\alpha_k)}
$$

$$
f_{X}(x_{1:(d-1)},v) =f_{Z}(z_{1:d})J(z_{1:d})
$$

$$
f_{Z}(z_{1:d})=f_{Z}(h^{-1}(x_{1:(d-1)},v))\left|\frac{\partial z_{1:d}}{\partial (x_{1:(d-1)},v)}\right|
$$

So we need to compute the determinant of the Jacobian \\( J(z_{1:d}) \\) of the transformation.

$$
\begin{aligned}
J(z_{1:d}) &= \det \left( \begin{bmatrix}
\frac{\partial z_1}{\partial x_1} & \cdots & \frac{\partial z_1}{\partial x_{d-1}} & \frac{\partial z_1}{\partial v} \\\\
\frac{\partial z_2}{\partial x_1} & \cdots & \frac{\partial z_2}{\partial x_{d-1}} & \frac{\partial z_2}{\partial v} \\\\
\vdots & \ddots & \cdots & \vdots \\\\
\frac{\partial z_d}{\partial x_1} & \cdots & \frac{\partial z_d}{\partial x_{d-1}} & \frac{\partial z_d}{\partial v}
\end{bmatrix} \right)
\end{aligned}
$$

Computing the individual terms, for \\( k<d \\) and \\( i=k \\),

$$
\begin{aligned}
\frac{\partial z_k}{\partial x_i} &= \frac{\partial vx_k}{\partial x_k}\\\\
&=v \\\\
\frac{\partial z_k}{\partial v} &= \frac{\partial vx_k}{\partial v} \\\\
&=x_k \\\\
\text{for } k<d \text{ and } i \neq k\text{, }\frac{\partial z_k}{\partial x_i} &= 0 \\\\
\text{for } k=d \text{, }\frac{\partial z_d}{\partial x_i} &= \frac{\partial v(1-\sum_{k=1}^{d-1}x_k)}{\partial x_i} \\\\
&=-v \\\\
\frac{\partial z_d}{\partial v} &= \frac{\partial v(1-\sum_{k=1}^{d-1}x_k)}{\partial v} \\\\
&=1-\sum_{k=1}^{d-1}x_k
\end{aligned}
$$

Now with this result from the determinant of the Jacobian, we will apply it in the transformation of random variables formula for the pdf and marginalize \\( v \\) to get the pdf of \\( x_{1:(d-1)} \\).

$$
\begin{aligned}
f_{X}(x_{1:(d-1)},v) &= f_{Z}(z_{1:d}) v^{d-1}=v^{d-1} \prod_{k=1}^d \frac{{(vx_k)}^{\alpha_k-1} e^{-vx_k}}{\Gamma(\alpha_k)} \\\\
f_{X}(x_{1:(d-1)}) &= \int_v f_{X}(x_{1:(d-1)},v) dv \\\\
&= \prod_{k=1}^d \frac{x_k^{\alpha_k-1}}{\Gamma(\alpha_k) } \int_v v^{d-1} \prod_{k=1}^d v^{\alpha_k-1} e^{-vx_k} dv \\\\
&= \prod_{k=1}^d \frac{x_k^{\alpha_k-1}}{\Gamma(\alpha_k) } \int_v v^{d-1-d+\sum_k \alpha_k} e^{-v \sum_k x_k} dv \\\\
&= \prod_{k=1}^d \frac{x_k^{\alpha_k-1}}{\Gamma(\alpha_k) } \underbrace{\int_v v^{-1+\sum_k \alpha_k} e^{-v} dv}_{\Gamma(\sum_k \alpha_k)} \\\\
\end{aligned}
$$

$$
f_{X}(x_{1:(d-1)}) = \Gamma(\sum_k \alpha_k) \prod_{k=1}^d \frac{x_k^{\alpha_k-1}}{\Gamma(\alpha_k)} 
$$

This finishes our demonstration that the given transformation of Gamma distributed random variables results in Dirichlet distributed random variables. Notice that \\( x_d \\) is totally determined by \\( x_{1:(d-1)} \\) (\\( \sum_k x_k = 1 \\)), so we can actually express the right side of the equation as a formula of only \\( x_{1:(d-1)} \\).

If drawing \\( d \\) samples from a Gamma distribution can be performed with complexity \\( O(g(d)) \\), then \\( d \\) samples from a Dirichlet distribution can be computed with the same \\( O(g(d)) \\) complexity.
