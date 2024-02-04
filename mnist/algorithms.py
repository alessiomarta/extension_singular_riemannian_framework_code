import torch
from jacobian_function import jacobian
import copy

def simec(network,  input_, g, steps, save_every_n_steps = 1, delta = 1e-3):

    imgs_list = []   
    pseudolength = 0.
    x1 = input_
    
    for i in range(steps):
        if (i%100 == 0):
            print("Step:", i)
            print("Peudolength of the curve:", pseudolength)

        #Initialize the input coordinates

        #Compute the corresponding output (needed to compute the gradient in pytorch)
        tensor = network(x1)
        
        #Split the tensor containing the output in one tensor per component
        u = torch.split(tensor, 1, dim=1)
        
        #Compute the gradient for each component of the output
        du = []
        for j in range(10):
            du.append(torch.autograd.grad(u[j], x1, retain_graph=True )[0].detach().flatten())
        
        #Build the Hessian matrix
        jacobian = torch.vstack(du)
        
        #Compute the pullback
        g_jac = torch.mm(g,jacobian)
        jac_t = torch.transpose(jacobian,0,1)
        pullback_metric = torch.mm(jac_t,g_jac).type(torch.float64)
        
        
        #Compute the eigenvectors and the eigenvalues of the pullback metric
        L,V = torch.linalg.eigh(pullback_metric,UPLO="L")
        
        #Randomly select a null eigenvector
        idx = torch.randint(0, 774, (1,))[0]
        null_eigen_vector = V[:,idx]

        #Compute an approximation of the pseudolength
        pseudolength = pseudolength + torch.abs(L[idx])*delta
   
        #Flatten the eigenvector and the input
        null_eigen_vector = null_eigen_vector.flatten()     
        original_shape = input_.shape
        tmp = x1.flatten()
        
        direction = idx % 2 - 0.5

        #Proceed along a null eigenvector
        tmp = tmp+null_eigen_vector*direction*delta

        #The coordinates of the features space are normalized with min=-.42 and max=2.8
        #Cut values under MIN and MAX to remain inside meaningful data
        tmp = torch.clamp(tmp, min=-.42, max=2.8)

        #Reshape the result to the original shape (for the next iteration)
        x1 = tmp.reshape(original_shape)
        
        x1 = x1.type(torch.float32)

        #Add image to the results
        if (i % save_every_n_steps == 0):
            imgs_list.append(copy.deepcopy(x1.detach().cpu()))

    return imgs_list

#--------------------------------------------------------------------------------------------

def simec_simexp(network,  input_, g, steps, save_every_n_steps = 1, delta = 1e-3, delta_simexp = 1e-3):

    imgs_list = []   
    pseudolength = 0.
    
    #Initialize the input coordinates
    x1 = input_
 
    for i in range(steps):
        if (i%100 == 0):
            print("Step:", i)
            print("Pseudolength of the curve:", pseudolength)

        #Compute the corresponding output (needed to compute the gradient in pytorch)
        tensor = network(x1)
        
        #Split the tensor containing the output in one tensor per component
        u = torch.split(tensor, 1, dim=1)
        
        #Compute the gradient for each component of the output
        du = []
        for j in range(10):
            du.append(torch.autograd.grad(u[j], x1, retain_graph=True )[0].detach().flatten())
        
        #Build the Hessian matrix
        jacobian = torch.vstack(du)
        
        #Compute the pullback
        g_jac = torch.mm(g,jacobian)
        jac_t = torch.transpose(jacobian,0,1)
        pullback_metric = torch.mm(jac_t,g_jac).type(torch.float64)
     
        #Compute the eigenvectors and the eigenvalues of the pullback metric
        L,V = torch.linalg.eigh(pullback_metric,UPLO="L")
        
        #Flatten the eigenvector and the input  
        original_shape = x1.shape
        tmp = x1.flatten()
        
        #Run a SiMEC step
        idx = torch.randint(0, 774, (1,))[0]
        null_eigen_vector = V[:,idx]
        null_eigen_vector = null_eigen_vector.flatten()  
         
        pseudolength = pseudolength + torch.abs(L[idx])*delta
        direction = idx % 2 - 0.5
        tmp = tmp+null_eigen_vector*direction*delta       
        
        # Run a SiMEXP step
        idx = torch.randint(774, 784, (1,))[0]
        non_null_eigen_vector = V[:,idx]
        non_null_eigen_vector = non_null_eigen_vector.flatten()      
        
        pseudolength = pseudolength + torch.abs(L[idx])*delta_simexp
        direction = idx % 2 - 0.5
        tmp = tmp+non_null_eigen_vector*direction*delta_simexp
        
        #The coordinates of the features space are normalized with min=-.42 and max=2.8
        #Cut values under MIN and MAX to remain inside meaningful data
        tmp = torch.clamp(tmp, min=-.42, max=2.8)

        #Reshape the result to the original shape (for the next iteration)
        x1 = tmp.reshape(original_shape)   
        x1 = x1.type(torch.float32)

        #Add image to the results
        if (i % save_every_n_steps == 0):
            imgs_list.append(copy.deepcopy(x1.detach().cpu()))

    return imgs_list

