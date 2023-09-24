
import torch

#------------------------------------------------------------------------------

class NthLayer(torch.nn.Module):
    """
    Wrap any model to get the response of an intermediate layer.
    Helper class for the function layerwise_jacobian.

    Code from https://gist.github.com/lyndond/ce29865d34e9c041a2701652260f2f32
    """
    def __init__(self, model, layer=None):
        """
        Args:
            model: A PyTorch model
            layer (int): The id of the layer
        """
        super().__init__()
        features = list(model.modules())[1:]
        self.features = nn.ModuleList(features).eval()

        if layer is None:
            layer = len(self.features)
        self.layer = layer

    def forward(self, x):
        """
        Propagates input through each layer of model until self.layer, at which point it returns that layer's output.

        Args:
            x (torch.Tensor) : A tensor containing the input of the model.
        
        Returns:
            torch.Tensor : A tensor containing the output of a layer.

        Code from https://gist.github.com/lyndond/ce29865d34e9c041a2701652260f2f32
        """
        for ii, mdl in enumerate(self.features):
            x = mdl(x)
            if ii == self.layer:
                return x

#------------------------------------------------------------------------------

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

#------------------------------------------------------------------------------

def layerwise_jacobian(model, input_, layer):
    """
    Creates a flat tensor w/ grad, reshapes to feed in as input, passes through model until specified nth layer, returned output is used to compute gradient
    Code from https://gist.github.com/lyndond/ce29865d34e9c041a2701652260f2f32
    """

    shape = input_.shape
    #Attach gradient to flattened tensor
    x_flat = input_.flatten().requires_grad_()  
    #And reshape it to input into model
    x_input = x_flat.reshape(shape) 

    mdl_layer = NthLayer(model, layer=layer)  # specify nth layer
    y = mdl_layer(x_input)
    #y_reduce = y.mean(dim=[2, 3])  # collapse [b, c, h, w] --> [b, c]
    #y_flat = y_reduce.flatten()
    y_flat = y
    print('input dim', x_flat.shape)
    print(f'layer {layer:d} output', y_flat.shape)  # should be 5D then 9D

    jac = jacobian(y_flat, x_flat)  # store this
    print(f'layer {layer:d} jac shape', jac.shape)  # should be 5x147 then 9x147


#-----------------------------------------------------------------------------------------