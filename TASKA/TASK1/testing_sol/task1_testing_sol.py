import numpy as np
import matplotlib.pyplot as plt
import torch.optim as optim
import torch
from torch.utils.data import DataLoader
from common import NeuralNet, MultiVariatePoly
import os

import time
import datetime

torch.set_num_threads(4)

torch.autograd.set_detect_anomaly(True)
torch.manual_seed(12)

class Pinns_sol:
    def __init__(self, n_int_, n_sb_, n_tb_, To, Th):
        self.n_int = n_int_
        self.n_sb = n_sb_
        self.n_tb = n_tb_
        self.dir = os.path.dirname(os.path.realpath(__file__))
        self.iteration = 0

        self.To = To
        self.Th = Th

        self.domain_extrema = torch.tensor([[0, 1],  # Time dimension
                                            [0, 1]])  # Space dimension

        self.space_dimensions = 1


        self.approximate_solution = NeuralNet(input_dimension=self.domain_extrema.shape[0], output_dimension=2,
                                              n_hidden_layers=8,
                                              neurons=64,
                                              regularization_param=0.,
                                              regularization_exp=2.,
                                              retrain_seed=42)
        '''self.approximate_solution = MultiVariatePoly(self.domain_extrema.shape[0], 3)'''

        self.soboleng = torch.quasirandom.SobolEngine(dimension=self.domain_extrema.shape[0])

        self.training_set_sb_0, self.training_set_sb_L, self.training_set_tb, self.training_set_int = self.assemble_datasets()


################################################################################################
################################################################################################
    

    def convert(self, tens):
        assert (tens.shape[1] == self.domain_extrema.shape[0])   #num colonne == num righe di domain_extrema altrimenti lancia errore
        return tens * (self.domain_extrema[:, 1] - self.domain_extrema[:, 0]) + self.domain_extrema[:, 0]    # (seconda colonna - prima colonna) + prima colonna -> normalizza 

    def initial_condition_Tf(self, x):
        return torch.sin(2*np.pi*x)
    def initial_condition_Ts(self, x):
        return torch.cos(2*np.pi*x)
    
    def initial_spatial_condition(self, t):
        return torch.zeros_like(t)

    def exact_sol_Tf(self, inputs):
        t = inputs[:,0]
        x = inputs[:,1]

        Tf = torch.exp(-t)*torch.sin(2*np.pi*x)

        return Tf #Tf.shape [n]

    def exact_sol_Ts(self, inputs):
        t = inputs[:,0]
        x = inputs[:,1]

        Ts = torch.exp(-t)*torch.cos(2*np.pi*x)

        return Ts
    

################################################################################################
################################################################################################

 
    def add_temporal_boundary_points(self):

        t0 = self.domain_extrema[0, 0] 
        input_tb = self.convert(self.soboleng.draw(self.n_tb))   
        input_tb[:, 0] = torch.full(input_tb[:, 0].shape, t0)   
        output_tb_Tf = self.initial_condition_Tf(input_tb[:, 1]).reshape(-1,1)  
        output_tb_Ts = self.initial_condition_Ts(input_tb[:, 1]).reshape(-1,1) 

        return input_tb, output_tb_Tf, output_tb_Ts    #input_tb [nx2], output_tb_Tf [nx1], output_tb_Ts [nx1]


    def add_spatial_boundary_points(self):

        x0 = self.domain_extrema[1, 0]         
        xL = self.domain_extrema[1, 1]         

        input_sb = self.convert(self.soboleng.draw(self.n_sb))

        input_sb_0 = torch.clone(input_sb)
        input_sb_0[:, 1] = torch.full(input_sb_0[:, 1].shape, x0)

        input_sb_L = torch.clone(input_sb)
        input_sb_L[:, 1] = torch.full(input_sb_L[:, 1].shape, xL)

        output_sb_0_Tf = self.initial_spatial_condition(input_sb_0[:,0]).reshape(-1,1)
        output_sb_L_Tf = torch.zeros_like(output_sb_0_Tf)                                                                        
        
        output_sb_0_Ts = torch.zeros_like(output_sb_0_Tf)
        output_sb_L_Ts = torch.zeros_like(output_sb_0_Tf)                      

        return input_sb_0, input_sb_L, output_sb_0_Tf, output_sb_L_Tf, output_sb_0_Ts, output_sb_L_Ts  #input_sb_0 [nx2]  output_sb_0_Tf  [nx1]
    

    def add_interior_points(self):

        input_int = self.convert(self.soboleng.draw(self.n_int))
        input_int = input_int[~((input_int[:,0] == 0) | (input_int[:,1] == 0) | (input_int[:,1] == 1))]

        output_int_Tf = torch.zeros((input_int.shape[0], 1)).reshape(-1,1)   
        output_int_Ts =  torch.clone(output_int_Tf)   

        return input_int, output_int_Tf, output_int_Ts  #input_int [nx2] output_int_Tf [nx1] 


    def assemble_datasets(self):

        input_sb_0, input_sb_L, output_sb_0_Tf, output_sb_L_Tf, output_sb_0_Ts, output_sb_L_Ts = self.add_spatial_boundary_points()   
        input_tb, output_tb_Tf, output_tb_Ts = self.add_temporal_boundary_points()  
        input_int, output_int_Tf, output_int_Ts = self.add_interior_points()         

        print("°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°")
        print("ASSEMBLE DATASET")
        print("input_sb_0, input_sb_L, output_sb_0_Tf, output_sb_L_Tf, output_sb_0_Ts, output_sb_L_Ts")
        print(input_sb_0.size(), input_sb_L.size(), output_sb_0_Tf.size(), output_sb_L_Tf.size(), output_sb_0_Ts.size(), output_sb_L_Ts.size())
        print("input_tb, output_tb_Tf")
        print(input_tb.size(), output_tb_Tf.size())
        print("input_int, output_int_Tf, output_int_Ts")
        print(input_int.size(), output_int_Tf.size(), output_int_Ts.size())

        print("°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°")
        print("°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°")

        print("input_sb_0 , input_sb_L | output_sb_0_Tf , output_sb_L_Tf | output_sb_0_Ts, output_sb_L_Ts")
        for i, j in enumerate(input_sb_0[:,0].detach().numpy()):
            print(input_sb_0.detach().numpy()[i], " , ", input_sb_L.detach().numpy()[i], " | ",  output_sb_0_Tf.reshape(-1,).detach().numpy()[i], " , ", output_sb_L_Tf.reshape(-1,).detach().numpy()[i], " | ", output_sb_0_Ts.reshape(-1,).detach().numpy()[i], " , ", output_sb_L_Ts.reshape(-1,).detach().numpy()[i])

        print("°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°")

        print("input_tb | output_tb_Tf")
        for i, j in enumerate(input_tb[:,0].detach().numpy()):
            print(input_tb.detach().numpy()[i], " | ",  output_tb_Tf.reshape(-1,).detach().numpy()[i])

        print("°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°")

        print("input_int | output_int_Tf | output_int_Ts")
        for i, j in enumerate(input_tb[:,0].detach().numpy()):
            print(input_int.detach().numpy()[i], " | ",  output_int_Tf.reshape(-1,).detach().numpy()[i], " | ", output_int_Ts.reshape(-1,).detach().numpy()[i])

        print("°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°")


        training_set_sb_0 = DataLoader(torch.utils.data.TensorDataset(input_sb_0, output_sb_0_Tf, output_sb_0_Ts), batch_size=self.n_sb, shuffle=False)      #batch_size????? chi moe e quando?
        training_set_sb_L = DataLoader(torch.utils.data.TensorDataset(input_sb_L, output_sb_L_Tf, output_sb_L_Ts), batch_size=self.n_sb, shuffle=False) 
        training_set_tb = DataLoader(torch.utils.data.TensorDataset(input_tb, output_tb_Tf, output_tb_Ts), batch_size=self.n_tb, shuffle=False)
        training_set_int = DataLoader(torch.utils.data.TensorDataset(input_int, output_int_Tf, output_int_Ts), batch_size=self.n_int, shuffle=False)
        
        return training_set_sb_0, training_set_sb_L, training_set_tb, training_set_int

################################################################################################
################################################################################################
################################################################################################

    def apply_initial_condition(self, input_tb):
        pred_tb = self.approximate_solution(input_tb)
        return pred_tb   #pred_tb [nx2]

    def apply_boundary_conditions(self, input_sb):
        
        input_sb.requires_grad = True
        pred_sb = self.approximate_solution(input_sb)
        grad_Tf_sb = torch.autograd.grad(Tf_pred_sb.sum(), input_sb, create_graph=True)[0]


        lambdas.requires_grad = True
        phi_prediction = rcwa_solve(lambdas)
        derivarive = torch.autograd.grad(phi_prediction, lambdas, create_graph=True)



        Tf_pred_sb = pred_sb[:,0].reshape(-1,1)   
        Ts_pred_sb = pred_sb[:,1].reshape(-1,1)

        grad_Tf_sb = torch.autograd.grad(Tf_pred_sb.sum(), input_sb, create_graph=True)[0]
        grad_Ts_sb = torch.autograd.grad(Ts_pred_sb.sum(), input_sb, create_graph=True)[0]

        grad_Tf_x_sb = grad_Tf_sb[:, 1]
        grad_Ts_x_sb = grad_Ts_sb[:, 1]    

        return pred_sb, grad_Tf_x_sb, grad_Ts_x_sb  #pred_sb [nx2],  grad_Tf_x_sb [n]


    def compute_pde_residual(self, input_int):

        input_int.requires_grad = True
        approx_sol = self.approximate_solution(input_int)
        
        Tf = approx_sol[:,0]
        Ts = approx_sol[:,1]
        
        grad_Tf = torch.autograd.grad(Tf.sum(), input_int, create_graph=True)[0]                     
        grad_Ts = torch.autograd.grad(Ts.sum(), input_int, create_graph=True)[0]
        grad_Tf_t = grad_Tf[:, 0]
        grad_Tf_x = grad_Tf[:, 1]
        grad_Ts_t = grad_Ts[:, 0]
        grad_Ts_x = grad_Ts[:, 1]
        grad_Tf_xx = torch.autograd.grad(grad_Tf_x.sum(), input_int, create_graph=True)[0][:, 1]
        grad_Ts_xx = torch.autograd.grad(grad_Ts_x.sum(), input_int, create_graph=True)[0][:, 1]

        assert((Tf.shape[0] == Ts.shape[0]) & (Ts.shape[0] == grad_Tf_t.shape[0]) & (grad_Tf_t.shape[0] == grad_Tf_x.shape[0]) & (grad_Tf_x.shape[0] == grad_Ts_t.shape[0]) & (grad_Ts_t.shape[0] == grad_Ts_x.shape[0]) & (grad_Ts_x.shape[0] == grad_Tf_xx.shape[0]) & (grad_Tf_xx.shape[0] == grad_Ts_xx.shape[0]))
        #assert((Tf.shape[1] == Ts.shape[1]) & (Ts.shape[1] == grad_Tf_t.shape[1]) & (grad_Tf_t.shape[1] == grad_Tf_x.shape[1]) & (grad_Tf_x.shape[1] == grad_Ts_t.shape[1]) & (grad_Ts_t.shape[1] == grad_Ts_x.shape[1]) & (grad_Ts_x.shape[1] == grad_Tf_xx.shape[1]) & (grad_Tf_xx.shape[1] == grad_Ts_xx.shape[1]))


        residual1 = (grad_Tf_t + grad_Tf_x - 0.05 * grad_Tf_xx + 5 * (Tf - Ts)) - ((0.05*4*np.pi*np.pi+5-1)*Tf + (2*np.pi-5)*Ts)
        residual2 = (grad_Ts_t - 0.08 * grad_Ts_xx - 6 * (Tf - Ts)) - ((0.08*4*np.pi*np.pi+6-1)*Ts - 6*Tf) 
        #residual = (grad_Tf_t + grad_Tf_x - 0.05 * grad_Tf_xx + 5 * (Tf.reshape(-1,1) - Ts.reshape(-1,1))) + (grad_Ts_t - 0.08 * grad_Ts_xx - 6 * (Tf.reshape(-1,1) - Ts.reshape(-1,1))) - ((0.05*4*np.pi*np.pi+5-1)*Tf.reshape(-1,1) + (2*np.pi-5)*Ts.reshape(-1,1) +(0.08*4*np.pi*np.pi+6-1)*Ts.reshape(-1,1) - 6*Tf.reshape(-1,1)) 

        return residual1, residual2


    def compute_loss(self, inp_train_sb_0, output_sb_0_Tf, output_sb_0_Ts, inp_train_sb_L, output_sb_L_Tf, output_sb_L_Ts, inp_train_tb, output_tb_Tf, output_tb_Ts, inp_train_int, output_int_Tf, output_int_Ts, verbose=True):
           
        pred_sb_0, grad_Tf_x_sb_0, grad_Ts_x_sb_0 = self.apply_boundary_conditions(inp_train_sb_0)
        pred_sb_L, grad_Tf_x_sb_L, grad_Ts_x_sb_L = self.apply_boundary_conditions(inp_train_sb_L)

        Tf_pred_sb_0 = pred_sb_0[:,0].reshape(-1,1)    #[nx1]
        Ts_pred_sb_0 = pred_sb_0[:,1].reshape(-1,1)    #[nx1]

        Tf_pred_sb_L = pred_sb_L[:,0].reshape(-1,1)    #[nx1]
        Ts_pred_sb_L = pred_sb_L[:,1].reshape(-1,1)    #[nx1]

        pred_tb = self.apply_initial_condition(inp_train_tb)     
        Tf_pred_tb = pred_tb[:,0].reshape(-1,1)        #[nx1]
        Ts_pred_tb = pred_tb[:,1].reshape(-1,1)        #[nx1]

        zeros = torch.zeros_like(inp_train_sb_0[:,0])

        assert (Tf_pred_sb_0.shape[1] == output_sb_0_Tf.shape[1])          
        assert (Ts_pred_sb_0.shape[1] == output_sb_0_Tf.shape[1])          
        assert (Tf_pred_tb.shape[1] == output_tb_Tf.shape[1])  
        assert (Ts_pred_tb.shape[1] == output_tb_Ts.shape[1])  

        assert (Tf_pred_sb_0.shape[0] == output_sb_0_Tf.shape[0])          
        assert (Ts_pred_sb_0.shape[0] == output_sb_0_Tf.shape[0])          
        assert (Tf_pred_tb.shape[0] == output_tb_Tf.shape[0])  
        assert (Ts_pred_tb.shape[0] == output_tb_Ts.shape[0])  

        assert(grad_Tf_x_sb_0.shape[0] == zeros.shape[0])
        assert(grad_Ts_x_sb_0.shape[0] == zeros.shape[0])
        assert(grad_Tf_x_sb_L.shape[0] == zeros.shape[0])
        assert(grad_Ts_x_sb_L.shape[0] == zeros.shape[0])
       
################################################

        r_deriv_Ts_0 = grad_Ts_x_sb_0
        r_deriv_Ts_L = grad_Ts_x_sb_L
        r_deriv_Tf_L = grad_Tf_x_sb_L - 2*np.pi*torch.exp(-inp_train_sb_L[:,0])

        loss_deriv_Ts_0 = torch.mean(abs(r_deriv_Ts_0) ** 2)
        loss_deriv_Ts_L = torch.mean(abs(r_deriv_Ts_L) ** 2)
        loss_deriv_Tf_L = torch.mean(abs(r_deriv_Tf_L) ** 2)

        loss_deriv = loss_deriv_Ts_0 + loss_deriv_Ts_L + loss_deriv_Tf_L

#################################################

        r_int1, r_int2 = self.compute_pde_residual(inp_train_int)            
        loss_int = torch.mean(abs(r_int1) ** 2) + torch.mean(abs(r_int2) ** 2)

##################################################

        r_sb_Tf_0 = Tf_pred_sb_0
        loss_sb_Tf = torch.mean(abs(r_sb_Tf_0) ** 2)

##################################################

        r_tb_Tf = output_tb_Tf - Tf_pred_tb     
        r_tb_Ts = output_tb_Ts - Ts_pred_tb     

        loss_tb_Tf = torch.mean(abs(r_tb_Tf) ** 2)
        loss_tb_Ts = torch.mean(abs(r_tb_Ts) ** 2) 

        loss_tb = loss_tb_Tf + loss_tb_Ts

##################################################

        loss_tsb = loss_sb_Tf + loss_tb + loss_deriv

        loss = torch.log10(loss_tsb + loss_int)
        
        if verbose: print("Iteration:", self.iteration, "||", "Total loss: ", round(loss.item(), 4), "| Boundary Loss: ", round(torch.log10(loss_tsb).item(), 4), "| Internal Loss: ", round(torch.log10(loss_int).item(), 4), "| Derivative Loss: ", round(torch.log10(loss_deriv).item(), 4))
        self.iteration = self.iteration +1
        return loss, torch.log10(loss_tb), torch.log10(loss_deriv), torch.log10(loss_sb_Tf), torch.log10(loss_int)

################################################################################################

    def fit(self, num_epochs, optimizer, verbose=True):
        history = list()
        history_tb = list()
        history_deriv = list()
        history_sb_Tf = list()
        history_int = list()


        # Loop over epochs
        for epoch in range(num_epochs):
            if verbose: print("################################ ", epoch, " ################################")

            for j, ((inp_train_sb_0, output_sb_0_Tf, output_sb_0_Ts), (inp_train_sb_L, output_sb_L_Tf, output_sb_L_Ts), (inp_train_tb, output_tb_Tf, output_tb_Ts), (inp_train_int, output_int_Tf, output_int_Ts)) in enumerate(zip(self.training_set_sb_0, self.training_set_sb_L, self.training_set_tb, self.training_set_int)):
                def closure():
                    optimizer.zero_grad()

                    loss, loss_tb, loss_deriv, loss_sb_Tf, loss_int = self.compute_loss(inp_train_sb_0, output_sb_0_Tf, output_sb_0_Ts, inp_train_sb_L, output_sb_L_Tf, output_sb_L_Ts, inp_train_tb, output_tb_Tf, output_tb_Ts, inp_train_int, output_int_Tf, output_int_Ts, verbose=verbose)
                    
                    loss.backward()

                    history.append(loss.item())
                    history_tb.append(loss_tb.item())
                    history_deriv.append(loss_deriv.item())
                    history_sb_Tf.append(loss_sb_Tf.item())
                    history_int.append(loss_int.item())

                    return loss

                optimizer.step(closure=closure)

        print('Final Loss: ', history[-1])

        return history, history_tb, history_deriv, history_sb_Tf, history_int

################################################################################################
################################################################################################
################################################################################################
 
    def plotting_assemble_datasets(self):

        input_sb_0, input_sb_L, output_sb_0_Tf, output_sb_L_Tf, output_sb_0_Ts, output_sb_L_Ts = self.add_spatial_boundary_points()  
        input_tb, output_tb_Tf, output_tb_Ts = self.add_temporal_boundary_points()  
        input_int, output_int_Tf, output_int_Ts = self.add_interior_points()  

        inputs =  torch.cat([input_sb_0, input_sb_L, input_tb, input_int ], 0) 
        
        outputs_Tf = torch.cat([output_sb_0_Tf, output_sb_L_Tf, output_tb_Tf, output_int_Tf], 0)
        outputs_Ts = torch.cat([output_sb_0_Ts, output_sb_L_Ts, output_tb_Ts, output_int_Ts], 0)


        fig, axs = plt.subplots(1, 2, figsize=(16, 8), dpi=150)
        fig.suptitle("Dataset")
        im1 = axs[0].scatter(inputs[:, 1].detach().numpy(), inputs[:, 0].detach().numpy(), c=outputs_Tf.detach().numpy(), cmap="jet")
        axs[0].set_xlabel("x")
        axs[0].set_ylabel("t")
        plt.colorbar(im1, ax=axs[0])
        axs[0].grid(True, which="both", ls=":")

        im2 = axs[1].scatter(inputs[:, 1].detach().numpy(), inputs[:, 0].detach().numpy(), c=outputs_Ts.detach().numpy(), cmap="jet")
        axs[1].set_xlabel("x")
        axs[1].set_ylabel("t")
        plt.colorbar(im2, ax=axs[1])
        axs[1].grid(True, which="both", ls=":")
        axs[0].set_title("Tf")
        axs[1].set_title("Ts")

        plt.savefig(self.dir + "/assemble_dataset_" + f"model_{datetime.datetime.now()}" + ".png")


        input_sb_0_np_sorted, output_sb_Tf_0_np_sorted = zip(*sorted(zip(input_sb_0[:, 0].detach().numpy(), output_sb_0_Tf.detach().numpy())))


        fig2, axs2 = plt.subplots(1, 3, figsize=(16, 8), dpi=150)
        fig2.suptitle("Dataset BC")
        axs2[0].plot(input_sb_0_np_sorted, output_sb_Tf_0_np_sorted, label="SC Tf at x=0")
        axs2[0].set_xlabel("t")
        axs2[0].set_ylabel("T")
        axs2[0].legend(loc='best')
        axs2[0].grid(True, which="both", ls=":")

        input_tb_np_sorted_Tf, output_tb_np_sorted_Tf = zip(*sorted(zip(input_tb[:, 1].detach().numpy(),output_tb_Tf.detach().numpy())))
        input_tb_np_sorted_Ts, output_tb_np_sorted_Ts = zip(*sorted(zip(input_tb[:, 1].detach().numpy(),output_tb_Ts.detach().numpy())))


        axs2[1].plot(input_tb_np_sorted_Tf, output_tb_np_sorted_Tf, label="IC Tf at t=0")
        axs2[1].set_xlabel("x")
        axs2[1].grid(True, which="both", ls=":")

        axs2[2].plot(input_tb_np_sorted_Ts, output_tb_np_sorted_Ts, label="IC Ts at t=0")
        axs2[2].set_xlabel("x")
        axs2[2].grid(True, which="both", ls=":")



        axs2[0].set_title("BC Tf")
        axs2[1].set_title("IC Tf")
        axs2[2].set_title("IC Ts")

        plt.savefig(self.dir + "assemble_dataset_BC_" + f"model_{datetime.datetime.now()}" + ".png")

    
    def plot_exact_solutions(self):

        inputs = self.soboleng.draw(10000)
        inputs = self.convert(inputs)

        output_Tf = self.exact_sol_Tf(inputs)
        output_Ts = self.exact_sol_Ts(inputs)


        fig, axs = plt.subplots(1, 2, figsize=(16, 8), dpi=150)
        fig.suptitle("Exact solution")
        im1 = axs[0].scatter(inputs[:, 1].detach(), inputs[:, 0].detach(), c=output_Tf.detach(), cmap="jet")
        axs[0].set_xlabel("x")
        axs[0].set_ylabel("t")
        plt.colorbar(im1, ax=axs[0])
        axs[0].grid(True, which="both", ls=":")
        im2 = axs[1].scatter(inputs[:, 1].detach(), inputs[:, 0].detach(), c=output_Ts.detach(), cmap="jet")
        axs[1].set_xlabel("x")
        axs[1].set_ylabel("t")
        plt.colorbar(im2, ax=axs[1])
        axs[1].grid(True, which="both", ls=":")
        axs[0].set_title("Tf")
        axs[1].set_title("Ts")

        plt.savefig(self.dir + "/exact_solutions_" + f"model_{datetime.datetime.now()}" + ".png")



    def plotting_sol(self):
        inputs = self.soboleng.draw(10000)
        inputs = self.convert(inputs)

        output_Tf = self.approximate_solution(inputs)[:,0]
        output_Ts = self.approximate_solution(inputs)[:,1]

        exact_Tf = self.exact_sol_Tf(inputs)
        exact_Ts = self.exact_sol_Ts(inputs)

        errTf = (torch.mean((output_Tf - exact_Tf) ** 2) / torch.mean(exact_Tf ** 2)) ** 0.5 * 100
        print("L2 Relative Error Norm: ", errTf.item(), "%")
        errTs = (torch.mean((output_Ts - exact_Ts) ** 2) / torch.mean(exact_Ts ** 2)) ** 0.5 * 100
        print("L2 Relative Error Norm: ", errTs.item(), "%")




        fig, axs = plt.subplots(1, 2, figsize=(16, 8), dpi=150)
        fig.suptitle("NN full solution")
        im1 = axs[0].scatter(inputs[:, 1].detach(), inputs[:, 0].detach(), c=output_Tf.detach(), cmap="jet")
        axs[0].set_xlabel("x")
        axs[0].set_ylabel("t")
        plt.colorbar(im1, ax=axs[0])
        axs[0].grid(True, which="both", ls=":")
        im2 = axs[1].scatter(inputs[:, 1].detach(), inputs[:, 0].detach(), c=output_Ts.detach(), cmap="jet")
        axs[1].set_xlabel("x")
        axs[1].set_ylabel("t")
        plt.colorbar(im2, ax=axs[1])
        axs[1].grid(True, which="both", ls=":")
        axs[0].set_title("Tf: " + str(round(errTf.item(), 2)) + "%")
        axs[1].set_title("Ts: " + str(round(errTs.item(), 2)) + "%")


        plt.savefig(self.dir + "/solutions_" + f"model_{datetime.datetime.now()}" + ".png")


    def plottting_bc(self):

        inputs = self.convert(self.soboleng.draw(10000))

        t0 = self.domain_extrema[0, 0]
        x0 = self.domain_extrema[1, 0]
        xL = self.domain_extrema[1, 1]

        input_tb = torch.clone(inputs)
        input_tb[:, 0] = torch.full(input_tb[:, 0].shape, t0)
        
        input_sb_0 = torch.clone(inputs)
        input_sb_0[:, 1] = torch.full(input_sb_0[:, 1].shape, x0)

        input_sb_L = torch.clone(inputs)
        input_sb_L[:, 1] = torch.full(input_sb_L[:, 1].shape, xL)

        
        output_tb_Tf = self.approximate_solution(input_tb)[:,0]
        output_tb_Ts = self.approximate_solution(input_tb)[:,1]

        output_sb_0_Tf = self.approximate_solution(input_sb_0)[:,0]
        output_sb_L_Tf = self.approximate_solution(input_sb_L)[:,0]

        output_sb_0_Ts = self.approximate_solution(input_sb_0)[:,1]
        output_sb_L_Ts = self.approximate_solution(input_sb_L)[:,1]


        input_tb_np_sorted_Tf, output_tb_np_sorted_Tf = zip(*sorted(zip(input_tb[:, 1].detach().numpy(), output_tb_Tf.detach().numpy())))
        input_tb_np_sorted_Ts, output_tb_np_sorted_Ts = zip(*sorted(zip(input_tb[:, 1].detach().numpy(), output_tb_Ts.detach().numpy())))

        input_sb_0_np_sorted_Tf, output_sb_0_np_sorted_Tf = zip(*sorted(zip(input_sb_0[:, 0].detach().numpy(), output_sb_0_Tf.detach().numpy())))

        exact_sb_Tf_0 = self.initial_spatial_condition(input_sb_0[:, 0])
        exact_tb_Tf = self.initial_condition_Tf(input_tb[:, 1])
        exact_tb_Ts = self.initial_condition_Ts(input_tb[:, 1])



        err_sb_Tf_0 = (torch.mean((output_sb_0_Tf) ** 2)) ** 0.5 * 100
        print("L2 Relative Error Norm sb_Tf_0: ", err_sb_Tf_0.item(), "%")
        err_tb_Tf = (torch.mean((exact_tb_Tf-output_tb_Tf) ** 2) / torch.mean(exact_tb_Tf.float() ** 2)) ** 0.5 * 100
        print("L2 Relative Error Norm sb_Tf_0: ", err_tb_Tf.item(), "%")
        err_tb_Ts = (torch.mean((exact_tb_Ts-output_tb_Ts) ** 2) / torch.mean(exact_tb_Ts.float() ** 2)) ** 0.5 * 100
        print("L2 Relative Error Norm sb_Tf_0: ", err_tb_Ts.item(), "%")


        fig2, axs2 = plt.subplots(1, 3, figsize=(16, 8), dpi=150)
        fig2.suptitle("NN Boundary Prediction")
        axs2[0].plot(input_sb_0_np_sorted_Tf, output_sb_0_np_sorted_Tf, label="SC Tf at x=0")
        axs2[0].set_xlabel("t")
        axs2[0].set_ylabel("T")
        axs2[0].legend(loc='best')
        axs2[0].grid(True, which="both", ls=":")


        axs2[1].plot(input_tb_np_sorted_Tf, output_tb_np_sorted_Tf, label="IC Tf at t=0")
        axs2[1].set_xlabel("x")
        axs2[1].grid(True, which="both", ls=":")


        axs2[2].plot(input_tb_np_sorted_Ts, output_tb_np_sorted_Ts, label="IC Ts at t=0")
        axs2[2].set_xlabel("x")
        axs2[2].grid(True, which="both", ls=":")


        axs2[0].set_title("BC Tf: " + str(round(err_sb_Tf_0.item(), 2)) + "%")
        axs2[1].set_title("IC Tf: " + str(round(err_tb_Tf.item(), 2)) + "%")
        axs2[2].set_title("IC Ts: " + str(round(err_tb_Ts.item(), 2)) + "%")

        plt.savefig(self.dir + "/BC_solutions_" + f"model_{datetime.datetime.now()}" + ".png")
        

    def plotting_loss(self, history, history_tb, history_deriv, history_sb_Tf, history_int):

        fig, axs = plt.subplots(1, 2, figsize=(16, 8), dpi=150)
        axs[0].plot(np.arange(1, len(history) + 1), history, label="Train Loss")
        axs[0].legend(loc='best')
        axs[0].grid(True, which="both", ls=":")

        axs[1].plot(np.arange(1, len(history_tb) + 1), history_tb, label="Train Loss TB")
        axs[1].plot(np.arange(1, len(history_deriv) + 1), history_deriv, label="Train Loss SB Derivatives")
        axs[1].plot(np.arange(1, len(history_sb_Tf) + 1), history_sb_Tf, label="Train Loss SB Tf")
        axs[1].plot(np.arange(1, len(history_int) + 1), history_int, label="Train Loss INT")

        axs[1].legend(loc='best')
        axs[1].grid(True, which="both", ls=":")
        plt.savefig(self.dir + "Losses_" + f"model_{datetime.datetime.now()}" + ".png")



    def plotting_derivatives(self):


        x0 = self.domain_extrema[1, 0]
        xL = self.domain_extrema[1, 1]

        input_sb = self.convert(self.soboleng.draw(10000))

        input_sb_0 = torch.clone(input_sb)
        input_sb_0[:, 1] = torch.full(input_sb_0[:, 1].shape, x0)

        input_sb_L = torch.clone(input_sb)
        input_sb_L[:, 1] = torch.full(input_sb_L[:, 1].shape, xL)

        input_sb_0.requires_grad = True
        input_sb_L.requires_grad = True

        Tf_L = self.approximate_solution(input_sb_L)[:,0]
        Ts_0 = self.approximate_solution(input_sb_0)[:,1]
        Ts_L = self.approximate_solution(input_sb_L)[:,1]


        grad_Tf_L = torch.autograd.grad(Tf_L.sum(), input_sb_L, create_graph=True)[0]
        grad_Ts_0 = torch.autograd.grad(Ts_0.sum(), input_sb_0, create_graph=True)[0]
        grad_Ts_L = torch.autograd.grad(Ts_L.sum(), input_sb_L, create_graph=True)[0]

        grad_Tf_x_L = grad_Tf_L[:, 1]
        grad_Ts_x_0 = grad_Ts_0[:, 1]
        grad_Ts_x_L = grad_Ts_L[:, 1]
      
        input_sb_Tf_L_np_sorted, grad_Tf_x_L_np_sorted = zip(*sorted(zip(input_sb_L[:, 0].detach().numpy(), grad_Tf_x_L.detach().numpy())))
        input_sb_Ts_0_np_sorted, grad_Ts_x_0_np_sorted = zip(*sorted(zip(input_sb_0[:, 0].detach().numpy(), grad_Ts_x_0.detach().numpy())))
        input_sb_Ts_L_np_sorted, grad_Ts_x_L_np_sorted = zip(*sorted(zip(input_sb_L[:, 0].detach().numpy(), grad_Ts_x_L.detach().numpy())))


        exact_grad_Tf_x_L = 2*np.pi*torch.exp(-input_sb_L[:,0])


        err_grad_Tf_x_L = (torch.mean((grad_Tf_x_L - exact_grad_Tf_x_L) ** 2) / torch.mean(exact_grad_Tf_x_L ** 2)) ** 0.5 * 100
        print("L2 Error grad_Tf_x_L: ", err_grad_Tf_x_L.item(), "%")
        err_grad_Ts_x_0 = (torch.mean((grad_Ts_x_0) ** 2)) ** 0.5 * 100
        print("L2 Error grad_Ts_x_0: ", err_grad_Ts_x_0.item(), "%")
        err_grad_Ts_x_L = (torch.mean((grad_Ts_x_L) ** 2)) ** 0.5 * 100
        print("L2 Error grad_Ts_x_L: ", err_grad_Ts_x_L.item(), "%")




        fig, axs = plt.subplots(1, 3, figsize=(16, 8), dpi=150)
        im1 = axs[0].plot(input_sb_Tf_L_np_sorted, grad_Tf_x_L_np_sorted)
        axs[0].set_xlabel("t")
        axs[0].set_ylabel("Derivata")
        axs[0].grid(True, which="both", ls=":")

        im2 = axs[1].plot(input_sb_Ts_0_np_sorted, grad_Ts_x_0_np_sorted)
        axs[1].set_xlabel("t")
        axs[1].grid(True, which="both", ls=":")

        im3 = axs[2].plot(input_sb_Ts_L_np_sorted, grad_Ts_x_L_np_sorted)
        axs[2].set_xlabel("t")
        axs[2].grid(True, which="both", ls=":")

        axs[2].set_xlabel("t")
        axs[2].grid(True, which="both", ls=":")


        axs[0].set_title("Derivata_x Tf at x=L: " + str(round(err_grad_Tf_x_L.item(),2)) + "%")
        axs[1].set_title("Derivata_x Ts at x=0: " + str(round(err_grad_Ts_x_0.item(),2)) + "%")
        axs[2].set_title("Derivata_x Ts at x=L: " + str(round(err_grad_Ts_x_L.item(),2)) + "%")

        plt.savefig(self.dir + "/Der_solutions_" + f"model_{datetime.datetime.now()}" + ".png")







factor = 20

n_int = int(512 *factor)
n_sb = int(128 *factor)
n_tb = int(128 *factor)
To = 1
Th = 4

pinn = Pinns_sol(n_int, n_sb, n_tb, To, Th)


pinn.plotting_assemble_datasets()

pinn.plot_exact_solutions()

print("------")
print(pinn.dir)
print("------")

n_epochs = 10
optimizer_LBFGS = optim.LBFGS(pinn.approximate_solution.parameters(),
                              lr=float(0.5),
                              max_iter=50000,
                              max_eval=50000,
                              history_size=150,
                              line_search_fn="strong_wolfe",
                              tolerance_change=1.0 * np.finfo(float).eps)

optimizer_ADAM = optim.Adam(pinn.approximate_solution.parameters(), lr=float(0.001))


history, history_tb, history_deriv, history_sb_Tf, history_int = pinn.fit(num_epochs=n_epochs, optimizer=optimizer_LBFGS, verbose=True)


pinn.plotting_loss(history, history_tb, history_deriv, history_sb_Tf, history_int)


pinn.plotting_sol()

pinn.plottting_bc()

pinn.plotting_derivatives()

torch.save(pinn.approximate_solution.state_dict(), pinn.dir + "/models/" + f"model_{datetime.datetime.now()}")













################################################

#BASE_PATH = Path("/home/luchito/Desktop/DL/Tasks/Task2/model_checkpoints")
#model_path = BASE_PATH / "model_03_06_12_44.pt"

## Loading model
#pinn = Pinns(n_int, n_sb, n_tb, To, Th, Tc)
#pinn.approximate_solution.load_state_dict(torch.load(PATH))

## Saving model
#torch.save(pinn.approximate_solution.state_dict(), BASE_PATH / f"model_{datetime.datetime.now()}_{j}")

