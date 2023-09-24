import torch
from torchvision.utils import save_image
from jacobian_functions import jacobian
import matplotlib
import matplotlib.pyplot as plt
matplotlib.style.use('ggplot')

def simec_dec(model, device, input_, g, steps, epsilon = 1e-3, delta = 1e-2):
    x1 = input_
    for i in range(steps):
        if (i%100 == 0):
            print(i)
        with torch.cuda.device(device):
            
            #Attach gradient to flattened tensor and reshape it to input into model
            shape = x1.shape
            
            x_flat = x1.flatten().requires_grad_() 
            x_input = x_flat.reshape(shape) 
            
            #Compute the output of the network
            y_flat = model.forward(x_input)

            #Compute the Jacobian of the network
            jac = jacobian(y_flat, x_flat)
            jac_t = torch.transpose(jac, 0, 1)
            
            #Compute the pullback metric
            tmp = torch.mm(g,jac)
            pullback_metric = torch.mm(jac_t,tmp)

            #Compute eigenvalues and eigenvectors of the pullback metric
            eigenvalues, eigenvectors = torch.linalg.eigh(pullback_metric, UPLO="U")

            #Take one of the null eigenvectors
            #num_null = (torch.sum(eigenvalues < epsilon)).item()
            #rng = (torch.randint(num_null,(1,))).item()
            null = eigenvectors[0,:]
            null = null.flatten()
            x1 = x1+null*delta
            
            itop = y_flat.view([28,28])
            if (i%100 == 0):
                j = i // 100
                itop = y_flat.view([28,28])
                save_image(itop.cpu(), "outputs/simec_"+str(j)+".png")
    return x1


def simec_enc(model, device, input_, g, steps, epsilon = 1e-3, delta = 1e-4):
    x1 = input_
    for i in range(steps):
        if (i%100 == 0):
            print(i)
        with torch.cuda.device(device):
            
            #Attach gradient to flattened tensor and reshape it to input into model
            shape = x1.shape
            
            x_flat = x1.flatten().requires_grad_() 
            x_input = x_flat.reshape(shape) 
            
            #Compute the output of the network
            y_flat = model.forward(x_input)

            #Compute the Jacobian of the network
            jac = jacobian(y_flat, x_flat)
            jac_t = torch.transpose(jac, 0, 1)
            
            #Compute the pullback metric
            tmp = torch.mm(g,jac)
            pullback_metric = torch.mm(jac_t,tmp)

            #Compute eigenvalues and eigenvectors of the pullback metric
            eigenvalues, eigenvectors = torch.linalg.eigh(pullback_metric, UPLO="U")

            #Take one of the null eigenvectors
            num_null = (torch.sum(eigenvalues < epsilon)).item()
            rng = (torch.randint(num_null,(1,))).item()
            null = eigenvectors[rng,:]
            null = null.flatten()
            x1 = x1+null*delta
            x1 = torch.clamp(x1, min=0, max=1)
            itop = x1.view([28,28])
            if (i%100 == 0):
                j = i // 100
                itop = x1.view([28,28])*255
                save_image(itop.cpu(), "outputs/simec_"+str(j)+".png")
    return x1