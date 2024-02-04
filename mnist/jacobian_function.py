import torch

def jacobian(output_, input_):
    """
    Explicitly compute the full Jacobian matrix.

    Args:
        output (torch.Tensor): A model output with gradient attached
        input (torch.Tensor): A model input with gradient attached

    Returns:
    
    torch.Tensor: The Jacobian matrix, of dimensions torch.Size([len(output), len(input)])

    Code from https://gist.github.com/lyndond/ce29865d34e9c041a2701652260f2f32
    """

    #Compute the Jacobian using automatic differentiation. J is a tensor attached to a graph
    J = torch.stack([torch.autograd.grad([output_[i].sum()], [input_], retain_graph=True, create_graph=True)[0] for i in range(
        output_.size(0))], dim=-1).squeeze().t()
    
    #Return a 1x1 tensor even if the dimension of the output is 1
    if output_.shape[0] == 1: 
        J = J.unsqueeze(0)
    
    #Return a new Tensor detached from the current graph.
    return J.detach()
