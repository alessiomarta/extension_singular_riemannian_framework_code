import torch
from jacobian_function import jacobian

def simec(model,  input_, g, steps, epsilon = 1e-5, delta = 1e-4):
    x1 = input_
    for i in range(steps):

        #Attach gradient to flattened tensor and reshape it to input into model
        shape = x1.shape
        
        #x_flat = x1.flatten().requires_grad_() 
        x_input = x1.requires_grad_()
        
        #Compute the output of the network
        y_predict = model.forward(x_input)
        
        #Compute the Jacobian of the network
        jac = jacobian(y_predict, x1)[0]
        jac_t = torch.transpose(jac, 0, 1)
        
        #Compute the pullback metric
        tmp = torch.mm(g,jac)
        pullback_metric = torch.mm(jac_t,tmp).type(torch.float64)
        

        #Compute eigenvalues and eigenvectors of the pullback metric
        eigenvalues, eigenvectors = torch.linalg.eigh(pullback_metric, UPLO="U")
        
        #num_null = (torch.sum(eigenvalues < epsilon)).item()
        num_null = torch.sum(eigenvalues < epsilon)

        rng = (torch.randint(num_null,(1,))).item()

        direction = rng % 2 - 0.5
        null = eigenvectors[:,rng]*direction*delta
        x1 = x1+null
        

        x1 = torch.clamp(x1, min=0, max=1).type(torch.float32)
    return x1

