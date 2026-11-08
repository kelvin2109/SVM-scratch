# SVM from Scratch

A from-scratch implementation of **Support Vector Machines (SVM)** using quadratic programming via the [`cvxopt`](https://cvxopt.org/) library. This repository demonstrates both **hard-margin** and **soft-margin** SVMs with a linear kernel, built step-by-step so the math behind the dual optimization problem is transparent.

---

## 📁 Repository Structure

| File | Description |
|------|-------------|
| `SVM-hard_margin(scratch).ipynb` | Hard-margin SVM for linearly separable data (no slack variables). |
| `SVM-soft_margin(scratch).ipynb` | Soft-margin SVM with slack variables and regularization parameter `C`, allowing some misclassification. |

---

## 🧠 Background

An SVM finds the optimal hyperplane `w·x + b = 0` that separates two classes with the maximum margin. This is solved through the **dual formulation**:
$$
\begin{aligned}
\text{maximize} \quad & \sum_i \alpha_i - \tfrac{1}{2}\sum_i\sum_j \alpha_i \alpha_j y_i y_j (x_i \cdot x_j) \\
\text{subject to} \quad & \sum_i \alpha_i y_i = 0 \\
& \alpha_i \ge 0 && \text{(hard margin)} \\
& 0 \le \alpha_i \le C && \text{(soft margin)}
\end{aligned}
$$
