# SVM from Scratch

A from-scratch implementation of Support Vector Machines using quadratic programming via [`cvxopt`](https://cvxopt.org/), covering both **hard-margin** and **soft-margin** variants with a linear kernel.

## Contents

| Notebook | Description |
|----------|-------------|
| `SVM-hard_margin(scratch).ipynb` | Hard-margin SVM for linearly separable data. |
| `SVM-soft_margin(scratch).ipynb` | Soft-margin SVM with regularization parameter `C`, tolerating some misclassification. |

## The Problem

An SVM finds the optimal hyperplane `w·x + b = 0` that separates two classes with the maximum margin. This is solved through the **dual formulation**:

$$
\begin{aligned}
\text{maximize} \quad & \sum_i \alpha_i - \tfrac{1}{2}\sum_i\sum_j \alpha_i \alpha_j y_i y_j (x_i \cdot x_j) \\
\text{subject to} \quad & \sum_i \alpha_i y_i = 0 \\
& \alpha_i \ge 0 && \text{(hard margin)} \\
& 0 \le \alpha_i \le C && \text{(soft margin)}
\end{aligned}
$$

Once the $\alpha_i$ are found, the model parameters are:

$$
w = \sum_i \alpha_i y_i x_i, \qquad b = y_s - w \cdot x_s
$$

for any support vector $x_s$. Data points with $\alpha_i > 0$ are the **support vectors**.

## Implementation

The dual is passed to `cvxopt.solvers.qp`, which solves problems of the form:

$$
\begin{aligned}
\text{minimize} \quad & \tfrac{1}{2} x^\top P x + q^\top x \\
\text{subject to} \quad & Gx \le h \\
& Ax = b
\end{aligned}
$$

The notebooks map the SVM dual onto these matrices:

| CVXOPT | SVM dual |
|--------|----------|
| `P` | `outer(y, y) * K` (K = linear kernel matrix) |
| `q` | `-1` vector |
| `A`, `b` | `yᵀ`, `0` (equality constraint) |
| `G`, `h` | `-I, 0` (hard) or `[-I; I], [0; C]` (soft) |

## Requirements

```bash
pip install numpy matplotlib cvxopt
