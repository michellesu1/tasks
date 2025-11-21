import torch

def layernorm_ref(x, w, eps=1e-6):
    """
    Implement a reference version of Layer Normalization for a 2D tensor.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor of shape (N, D). Layer normalization should be applied
        independently to each row (along dim=1). The computation should be
        performed in float32 for numerical stability, regardless of the
        input tensor's dtype.
    w : torch.Tensor
        Learnable scaling weights of shape (D,). These should also be cast
        to float32 before use.
    eps : float, optional
        Small constant added to the variance for numerical stability.
        Defaults to 1e-6.

    Returns
    -------
    torch.Tensor
        The normalized tensor, scaled elementwise by `w`, and cast back to
        the original dtype of `x`. No bias term should be applied.

    Notes
    -----
    - You may not use `torch.nn.LayerNorm` the normalization must be computed
      manually using basic tensor operations.
    - The mean and variance should be computed across `dim=1` with
      `keepdim=True` so that broadcasting works correctly.


    
    """

    og_type = x.dtype
    x_float = x.to(torch.float32)
    w_float = w.to(torch.float32)

    mean = x_float.mean(dim=1, keepdim=True)
    pop_var = x_float.var(dim=1, unbiased=False, keepdim=True) # /n

    x_norm = (x_float - mean) / torch.sqrt(pop_var + eps)
    res = x_norm * w_float

    return res.to(og_type)

if __name__ == "__main__":


    x = torch.tensor([[1.5, 3.0, 5.0],
                      [2.0, 4.5, 6.0]])
    w = torch.tensor([1.0, 1.0, 1.0])

    res = layernorm_ref(x, w)

    ln = torch.nn.LayerNorm(3, elementwise_affine=True) #not learnable
    ln.weight.data = w.clone()
    ln.bias.data = torch.zeros(3) #no bias
    exp = ln(x)

    same = torch.allclose(res, exp, atol=1e-5)
    print(same)





    


