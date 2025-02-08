import numpy as np
import matplotlib.pyplot as plt
import torch.optim as optim
import torch
import time
from torch.utils.data import DataLoader
from common import NeuralNet, MultiVariatePoly
from scipy.interpolate import LinearNDInterpolator
import random
import os 
from pathlib import Path

import datetime


torch.set_num_threads(8)

torch.autograd.set_detect_anomaly(True)
torch.manual_seed(12)                                                                                        

phase_type = "i"
name = phase_type


########################################################################################
########################################################################################
########################################################################################

class Pinns:
    def __init__(self, n_int_, n_sb_, n_tb_, To, Th, Tc):
        self.n_int = n_int_
        self.n_sb = n_sb_
        self.n_tb = n_tb_
        self.dir = os.path.dirname(os.path.realpath(__file__))
        self.iteration = 0

        if phase_type == "i":
            self.v = 0

        elif phase_type == "c":
            self.v = 1

        elif phase_type == "d":
            self.v = -1

        self.n_meas = 100
        self.t_meas = 100
        
        self.To = To
        self.Th = Th
        self.Tc = Tc

        self.domain_extrema = torch.tensor([[0, 1],  # Time dimension
                                            [0, 1]])  # Space dimension


        self.lambda_tsb = 0.5
        self.lambda_int = 0.2
        self.lambda_meas = 1

        self.approximate_solution = NeuralNet(input_dimension=self.domain_extrema.shape[0], output_dimension=2,
                                              n_hidden_layers=8,
                                              neurons=64,
                                              regularization_param=0.,
                                              regularization_exp=2.,
                                              retrain_seed=42)                                                                                  #ANY PARAMETERS HERE HAS TO CHANGE??????

        self.soboleng = torch.quasirandom.SobolEngine(dimension=self.domain_extrema.shape[0])
        
        self.training_set_sb_0, self.training_set_sb_L, self.training_set_int, self.training_set_m = self.assemble_datasets()

        self.train_meas_input, self.train_meas_Tf, self.train_meas_Ts = self.get_measurement_data()



########################################################################################
########################################################################################


    # Function to linearly transform a tensor whose value are between 0 and 1 to a tensor whose values are between the domain extrema
    def convert(self, tens):
        assert (tens.shape[1] == self.domain_extrema.shape[0])
        return tens * (self.domain_extrema[:, 1] - self.domain_extrema[:, 0]) + self.domain_extrema[:, 0]
    
    # Function to normalize a vector [n], [nx1] or a vector [nxm] between 0-1 according to its max and min
    def normalize(self, tensor):
        min_vals, _ = torch.min(tensor, dim=0)
        max_vals, _ = torch.max(tensor, dim=0)
        normalized_tensor = (tensor - min_vals) / (max_vals - min_vals)
        return normalized_tensor
    
    def normalize_np(self, tensor):
        min_vals = np.min(tensor, axis=0)
        max_vals = np.max(tensor, axis=0)
        normalized_tensor = (tensor - min_vals) / (max_vals - min_vals)
        return normalized_tensor

    def normalize_cost(self, tens):
        return tens*0.1

    def initial_condition(self, x):
        return torch.full(x.shape, self.To)

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

########################################################################################
########################################################################################



    def add_spatial_boundary_points(self):
        x0 = self.domain_extrema[1, 0]
        xL = self.domain_extrema[1, 1]

        input_sb = self.normalize(self.convert(self.soboleng.draw(self.n_sb)))

        input_sb_0 = torch.clone(input_sb)
        input_sb_0[:, 1] = torch.full(input_sb_0[:, 1].shape, x0)

        input_sb_L = torch.clone(input_sb)
        input_sb_L[:, 1] = torch.full(input_sb_L[:, 1].shape, xL)

        output_sb_Tf_0 = torch.full(input_sb_0[:,0].shape, 0)
        output_sb_Tf_L = torch.full(input_sb_L[:,0].shape, 0)

        return input_sb_0, input_sb_L, output_sb_Tf_0.reshape(-1, 1), output_sb_Tf_L.reshape(-1, 1)  #size input_sb [nx2], size output_sb_Tf [n]


    def add_interior_points(self):
        input_int = self.normalize(self.convert(self.soboleng.draw(self.n_int)))
        input_int = input_int[~((input_int[:,1] == 0) | (input_int[:,1] == 1))]       #get rid of points on boundary

        output_int = torch.zeros((input_int.shape[0], 1))
        return input_int, output_int.reshape(-1, 1)  #size input_sb [nx2], size output_sb_Tf [nx1]
    
    def get_measurement_data(self):
        torch.random.manual_seed(42)
        # take measurments every 0.001 sec on a set of randomly placed (in space) sensors


        t = torch.linspace(0, self.domain_extrema[0, 1], self.t_meas)
        x = torch.linspace(self.domain_extrema[1, 0], self.domain_extrema[1, 1], self.n_meas)



        input_meas = self.normalize(torch.cartesian_prod(t, x))

        output_meas_Tf = self.exact_sol_Tf(input_meas).reshape(-1,1)
        noise = 0.01*torch.randn_like(output_meas_Tf)
        output_meas_Tf = output_meas_Tf + noise

        output_meas_Ts = self.exact_sol_Ts(input_meas).reshape(-1,1)
        noise = 0.01*torch.randn_like(output_meas_Ts)
        output_meas_Ts = output_meas_Ts + noise

        return input_meas, output_meas_Tf, output_meas_Tf



    def assemble_datasets(self):

        input_sb_0, input_sb_L, output_sb_Tf_0, output_sb_Tf_L = self.add_spatial_boundary_points()  
        input_int, output_int = self.add_interior_points()  
        input_m, Tf_m, Ts_m = self.get_measurement_data()                                                                                    #???????????????? Do I do a DataLoader su 40000000 ???

        print("°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°")
        print("ASSEMBLE DATASET")
        print("input_sb_0, input_sb_L, output_sb_Tf_0, output_sb_Tf_L")
        print(input_sb_0.size(), input_sb_L.size(), output_sb_Tf_0.size(), output_sb_Tf_L.size())
        print("input_int, output_int")
        print(input_int.size(), output_int.size())
        print("input_m, Tf_m")
        print(input_m.size(), Tf_m.size())

        print("°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°")
        print("°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°")

        print("input_sb_0 , input_sb_L | output_sb_Tf_0 , output_sb_Tf_L")
        for i, j in enumerate(input_sb_0[:,0].detach().numpy()):
            print(input_sb_0.detach().numpy()[i], " , ", input_sb_L.detach().numpy()[i], " | ",  output_sb_Tf_0.reshape(-1,).detach().numpy()[i], " , ", output_sb_Tf_L.reshape(-1,).detach().numpy()[i])

        print("°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°")

        print("input_int | output_int")
        for i, j in enumerate(input_int[:,0].detach().numpy()):
            print(input_int.detach().numpy()[i], " | ",  output_int.reshape(-1,).detach().numpy()[i])

        print("°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°")

        print("input_m | Tf_m")
        for i, j in enumerate(input_m[:,0].detach().numpy()):
            print(input_m.detach().numpy()[i], " | ",  Tf_m.reshape(-1,).detach().numpy()[i])

        print("°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°")
        print("°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°")


        training_set_sb_0 = DataLoader(torch.utils.data.TensorDataset(input_sb_0, output_sb_Tf_0), batch_size=self.n_sb, shuffle=False)                                        #ANY PARAMETERS (BATCH) HERE HAS TO CHANGE??????
        training_set_sb_L = DataLoader(torch.utils.data.TensorDataset(input_sb_L, output_sb_Tf_L), batch_size=self.n_sb, shuffle=False)
        training_set_int = DataLoader(torch.utils.data.TensorDataset(input_int, output_int), batch_size=self.n_int, shuffle=False)
        training_set_m = DataLoader(torch.utils.data.TensorDataset(input_m, Tf_m), batch_size=self.n_meas, shuffle=False)                                                        


        return training_set_sb_0, training_set_sb_L, training_set_int, training_set_m


########################################################################################
########################################################################################

    

    def apply_boundary_conditions(self, input_sb):

        input_sb.requires_grad = True
        pred_sb_Tf = self.approximate_solution(input_sb)[:,0]
        pred_sb_Ts = self.approximate_solution(input_sb)[:,1]

        grad_Tf_sb = torch.autograd.grad(pred_sb_Tf.sum(), input_sb, create_graph=True)[0]
        grad_Ts_sb = torch.autograd.grad(pred_sb_Ts.sum(), input_sb, create_graph=True)[0]

        grad_Tf_x_sb = grad_Tf_sb[:, 1]     
        grad_Ts_x_sb = grad_Ts_sb[:, 1]     

        return pred_sb_Tf.reshape(-1,1), grad_Tf_x_sb, grad_Ts_x_sb  #size pred_sb_Tf [nx1], size grad_Tf_x_sb [n]


    def compute_pde_residual(self, input_int):
        input_int.requires_grad = True

        Tf = self.approximate_solution(input_int)[:,0]
        Ts = self.approximate_solution(input_int)[:,1]

        grad_Tf = torch.autograd.grad(Tf.sum(), input_int, create_graph=True)[0]
        grad_Tf_t = grad_Tf[:, 0]
        grad_Tf_x = grad_Tf[:, 1]
        grad_Tf_xx = torch.autograd.grad(grad_Tf_x.sum(), input_int, create_graph=True)[0][:, 1]

        if phase_type == "i":
            residual = grad_Tf_t + self.v*grad_Tf_x - 0.005 * grad_Tf_xx + 5 * (Tf - Ts) - (-Tf + self.v*2*np.pi*Ts + 4*np.pi*0.005*Tf + 5*(Tf - Ts))

        elif phase_type == "c":
            residual = grad_Tf_t + self.v*grad_Tf_x - 0.005 * grad_Tf_xx + 5 * (Tf - Ts) - (-Tf + self.v*2*np.pi*Ts + 4*np.pi*0.005*Tf + 5*(Tf - Ts))

        elif phase_type == "d":
            residual = grad_Tf_t + self.v*grad_Tf_x - 0.005 * grad_Tf_xx + 5 * (Tf - Ts) - (-Tf + self.v*2*np.pi*Ts + 4*np.pi*0.005*Tf + 5*(Tf - Ts))

        return residual.reshape(-1)

    def compute_loss(self, input_sb_0, output_sb_Tf_0, input_sb_L, output_sb_Tf_L, input_int, output_int, input_m, Tf_m, verbose=True):

        pred_sb_Tf_0, grad_Tf_x_sb_0, grad_Ts_x_sb_0 = self.apply_boundary_conditions(input_sb_0) #size pred_sb_Tf [nx1], size grad_Tf_x_sb [n]
        pred_sb_Tf_L, grad_Tf_x_sb_L, grad_Ts_x_sb_L = self.apply_boundary_conditions(input_sb_L) #size pred_sb_Tf [nx1], size grad_Tf_x_sb [n]
        
            
        #INIZIALIZZANDO I DATI IN __init__
        train_meas_input, train_meas_Tf = self.train_meas_input_Tf, self.train_meas_Tf
        pred_Tf_m = self.approximate_solution(train_meas_input)[:,0].reshape(-1,1)

        """
        #WITH DATALOADER
        pred_Tf_m = self.approximate_solution_Tf(input_m)  #size pred_Tf_m [nx1]

        #AS PinnINV TUTORIAL
        train_meas_input, train_meas_Tf, train_meas_n_points = self.get_measurement_data()
        pred_Tf_m = self.approximate_solution_Tf(train_meas_input).reshape(-1,)
        """

        #DEBUG PRINT

        """
        print("||||||||||||||||||||||||||||||||||||||||||||||||||||")

        print("Predictions and gradients ")
        print("pred_sb_Tf_0, grad_Tf_x_sb_0, grad_Ts_x_sb_0")
        print(pred_sb_Tf_0.size(), grad_Tf_x_sb_0.size(), grad_Ts_x_sb_0.size() )
        print("pred_sb_Tf_L, grad_Tf_x_sb_L, grad_Ts_x_sb_L")
        print(pred_sb_Tf_L.size(), grad_Tf_x_sb_L.size(), grad_Ts_x_sb_L.size())
        print("pred_tb_Tf, pred_tb_Ts")
        print(pred_tb_Tf.size(), pred_tb_Ts.size())
        print("pred_Tf_m")
        print(pred_Tf_m.size())

        print("||||||||||||||||||||||||||||||||||||||||||||||||||||")
        """


        assert (pred_sb_Tf_0.shape == output_sb_Tf_0.shape)
        assert (pred_sb_Tf_L.shape == output_sb_Tf_L.shape)
        assert (pred_Tf_m.shape == train_meas_Tf.shape)

        assert (grad_Tf_x_sb_0.shape[0] == input_sb_0.shape[0])
        assert (grad_Ts_x_sb_0.shape[0] == input_sb_0.shape[0])
        assert (grad_Tf_x_sb_L.shape[0] == input_sb_L.shape[0])
        assert (grad_Ts_x_sb_L.shape[0] == input_sb_L.shape[0])
        

########################################################################################

        if phase_type == "i":
            r_sb_Tf_0 = grad_Tf_x_sb_0 - 2*np.pi*torch.exp(-input_sb_0[:,0])
            r_sb_Tf_L = grad_Tf_x_sb_L - 2*np.pi*torch.exp(-input_sb_L[:,0])

        elif phase_type == "c":
            r_sb_Tf_0 = pred_sb_Tf_0 
            r_sb_Tf_L = grad_Tf_x_sb_L - 2*np.pi*torch.exp(-input_sb_L[:,0])

        elif phase_type == "d":
            r_sb_Tf_0 = grad_Tf_x_sb_0 - 2*np.pi*torch.exp(-input_sb_0[:,0])
            r_sb_Tf_L = pred_sb_Tf_L
        

        loss_sb_Tf_0_Neu = torch.mean(abs(r_sb_Tf_0) ** 2)
        loss_sb_Tf_L_Neu = torch.mean(abs(r_sb_Tf_L) ** 2)

        loss_sb_Tf = loss_sb_Tf_0_Neu  + loss_sb_Tf_L_Neu

########################################################################################
   
        r_sb_Ts_0 = grad_Ts_x_sb_0
        r_sb_Ts_L = grad_Ts_x_sb_L

        loss_sb_Ts_0 = torch.mean(abs(r_sb_Ts_0)**2)
        loss_sb_Ts_L = torch.mean(abs(r_sb_Ts_L)**2)

        loss_sb_Ts = loss_sb_Ts_0 + loss_sb_Ts_L
        


########################################################################################


        r_meas = pred_Tf_m - train_meas_Tf
        
        loss_meas = torch.mean(abs(r_meas) ** 2)


########################################################################################


        r_int = self.compute_pde_residual(input_int)

        loss_int = torch.mean(abs(r_int) ** 2)


########################################################################################


        loss_tsb = loss_sb_Tf 


        loss = torch.log10( 0.2*loss_int + loss_tsb + loss_meas )

        if verbose: print("Iteration:", self.iteration, "||", "Total loss: ", round(loss.item(), 4), "| Loss_meas: ", round(torch.log10(loss_meas).item(), 4), "| Loss_int: ", round(torch.log10(loss_int).item(), 4), "| Loss_sb_Tf: ", round(torch.log10(loss_sb_Tf).item(), 4), "| Loss_sb_Ts: ", round(torch.log10(loss_sb_Ts).item(), 4))
        self.iteration = self.iteration +1

        #if verbose: print("Total loss: ", round(loss.item(), 4), "| Loss_meas: ", round(torch.log10(loss_meas).item(), 4), "| Loss_int: ", round(torch.log10(loss_int).item(), 4), "| Loss_sb_Tf: ", round(torch.log10(loss_sb_Tf).item(), 4), "| Loss_sb_Ts: ", round(torch.log10(loss_sb_Ts).item(), 4), "| Loss_int_m: ", round(torch.log10(loss_int_m).item(), 4),)
        
    
        return loss, torch.log10(loss_sb_Tf), torch.log10(loss_sb_Ts), torch.log10(loss_meas), torch.log10(loss_int)

########################################################################################

    def fit(self, num_epochs, optimizer, verbose=True):
        history = list()

        history_sb_Tf = list()
        history_sb_Ts = list()

        history_meas = list()
        history_int = list()

        for epoch in range(num_epochs):
            if verbose: print("################################ ", epoch, " ################################")

            for j, ((input_sb_0, output_sb_Tf_0), (input_sb_L, output_sb_Tf_L), (input_int, output_int), (input_m, Tf_m)) in enumerate(zip(self.training_set_sb_0, self.training_set_sb_L, self.training_set_int, self.training_set_m)):
                def closure():
                    optimizer.zero_grad()
                    loss, loss_sb_Tf, loss_sb_Ts, loss_meas, loss_int = self.compute_loss(input_sb_0, output_sb_Tf_0, input_sb_L, output_sb_Tf_L, input_int, output_int, input_m, Tf_m, verbose=verbose)
                    loss.backward()


                    history.append(loss.item())
                    
                    history_sb_Tf.append(loss_sb_Tf.item())
                    history_sb_Ts.append(loss_sb_Ts.item())

                    history_meas.append(loss_meas.item())
                    history_int .append(loss_int.item())

                    return loss

                optimizer.step(closure=closure)

        print('Final Loss: ', history[-1])
        

        return history, history_sb_Tf, history_sb_Ts, history_meas, history_int


########################################################################################
########################################################################################
########################################################################################
########################################################################################


    def plotting_assemble_datasets(self):

        
        input_sb_0, input_sb_L, output_sb_Tf_0, output_sb_Tf_L = self.add_spatial_boundary_points()  
        input_int, output_int = self.add_interior_points()  
        input_m, Tf_m, Ts_m = self.get_measurement_data()

        inputs =  torch.cat([input_sb_0, input_sb_L, input_int ], 0) 
        
        outputs = torch.cat([output_sb_Tf_0, output_sb_Tf_L, output_int], 0)


        fig, axs = plt.subplots(1, 3, figsize=(16, 8), dpi=150)
        im1 = axs[0].scatter(inputs[:, 1].detach().numpy(), inputs[:, 0].detach().numpy(), c=outputs.detach().numpy(), cmap="jet")
        axs[0].set_xlabel("x")
        axs[0].set_ylabel("t")
        plt.colorbar(im1, ax=axs[0])
        axs[0].grid(True, which="both", ls=":")
        axs[0].set_title("Tf")


        im2 = axs[1].scatter(input_m[:, 1].detach().numpy(), input_m[:, 0].detach().numpy(), c=Tf_m, cmap="jet")
        axs[1].set_xlabel("x")
        axs[1].grid(True, which="both", ls=":")
        axs[1].set_title("Tf MEAS")

        im3 = axs[2].scatter(input_m[:, 1].detach().numpy(), input_m[:, 0].detach().numpy(), c=Ts_m, cmap="jet")
        axs[2].set_xlabel("x")
        plt.colorbar(im2, ax=axs[2])
        axs[2].grid(True, which="both", ls=":")
        axs[2].set_title("Ts MEAS")


        plt.savefig(self.dir + "/assemble_dataset_" + name + ".png")




    def plotting_solutions(self):

        train_meas_input, train_meas_Tf, train_meas_Ts = self.train_meas_input, self.train_meas_Tf, self.train_meas_Ts


        output = self.approximate_solution(train_meas_input)
        output_Tf = output[:,0]
        output_Ts = output[:,1]

        err_Tf = (torch.mean((output_Tf - train_meas_Tf.reshape(-1,)) ** 2) / torch.mean(train_meas_Tf.reshape(-1,) ** 2)) ** 0.5 * 100
        print("L2 Relative Error Norm Tf: ", err_Tf.item(), "%")
        err_Ts = (torch.mean((output_Ts - train_meas_Ts.reshape(-1,)) ** 2) / torch.mean(train_meas_Ts.reshape(-1,) ** 2)) ** 0.5 * 100
        print("L2 Relative Error Norm Ts: ", err_Ts.item(), "%")

        fig, axs = plt.subplots(1, 2, figsize=(16, 8), dpi=150)
        im1 = axs[0].scatter(train_meas_input[:, 1].detach(), train_meas_input[:, 0].detach(), c=output_Tf.detach(), cmap="jet")
        axs[0].set_xlabel("x")
        axs[0].set_ylabel("t")
        plt.colorbar(im1, ax=axs[0])
        axs[0].grid(True, which="both", ls=":")
        im2 = axs[1].scatter(train_meas_input[:, 1].detach(), train_meas_input[:, 0].detach(), c=output_Ts.detach(), cmap="jet")
        axs[1].set_xlabel("x")
        axs[1].set_ylabel("t")
        plt.colorbar(im2, ax=axs[1])
        axs[1].grid(True, which="both", ls=":")
        axs[0].set_title("Tf Relative Error Norm to MEAS: " + str(round(err_Tf.item(),2)) + " % " )
        axs[1].set_title("Ts Relative Error Norm to MEAS: " + str(round(err_Ts.item(),2)) + " % " )

        plt.savefig(self.dir + "/solutions_" + name  + ".png")


    def plottting_bc(self):

        input_sb_0, input_sb_L, exact_sb_Tf_0, exact_sb_Tf_L = self.add_spatial_boundary_points()  



        t0 = self.domain_extrema[0, 0]
        x0 = self.domain_extrema[1, 0]
        xL = self.domain_extrema[1, 1]


        output_sb_Tf_0 = self.approximate_solution(input_sb_0)[:,0]
        output_sb_Tf_L = self.approximate_solution(input_sb_L)[:,0]


        input_sb_0_np_sorted, output_sb_Tf_0_np_sorted = zip(*sorted(zip(input_sb_0[:, 0].detach().numpy(), output_sb_Tf_0.detach().numpy())))
        input_sb_L_np_sorted, output_sb_Tf_L_np_sorted = zip(*sorted(zip(input_sb_L[:, 0].detach().numpy(), output_sb_Tf_L.detach().numpy())))

        if phase_type == "c" or phase_type == "d": 
            err_sb_Tf_0 = (torch.mean((output_sb_Tf_0 - exact_sb_Tf_0.reshape(-1)) ** 2)) ** 0.5 * 100
            print("L2 Relative Error Norm sb x=0 Tf: ", err_sb_Tf_0.item(), "%")
            err_sb_Tf_L = (torch.mean((output_sb_Tf_L - exact_sb_Tf_L.reshape(-1)) ** 2)) ** 0.5 * 100
            print("L2 Relative Error Norm sb x=L Tf: ", err_sb_Tf_L.item(), "%")

        elif: 

            err_sb_Tf_0 = (torch.mean((output_sb_Tf_0 - exact_sb_Tf_0.reshape(-1)) ** 2) / torch.mean(exact_sb_Tf_0.reshape(-1).float()** 2)) ** 0.5 * 100
            print("L2 Relative Error Norm sb x=0 Tf: ", err_sb_Tf_0.item(), "%")
            err_sb_Tf_L = (torch.mean((output_sb_Tf_L - exact_sb_Tf_L.reshape(-1)) ** 2) / torch.mean(exact_sb_Tf_L.reshape(-1).float()** 2)) ** 0.5 * 100
            print("L2 Relative Error Norm sb x=L Tf: ", err_sb_Tf_L.item(), "%")


        fig, axs = plt.subplots(1, 2, figsize=(16, 8), dpi=150)
        axs[0].plot(input_sb_0_np_sorted, output_sb_Tf_0_np_sorted)
        axs[0].set_xlabel("t")
        axs[0].set_ylabel("Tf")
        axs[0].grid(True, which="both", ls=":")

        axs[1].plot(input_sb_L_np_sorted, output_sb_Tf_L_np_sorted)
        axs[1].set_xlabel("t")
        axs[1].set_ylabel("Tf")
        axs[1].grid(True, which="both", ls=":")

        axs[0].set_title("SB Tf at x = 0; " + str(round(err_sb_Tf_0.item(),2)) + "%")
        axs[1].set_title("SB Tf at x = L; " + str(round(err_sb_Tf_L.item(),2)) + "%")


        if phase_type == "c" or phase_type == "d":
            plt.savefig(self.dir + "/SBC_solutions_" + name  + ".png")




    def plotting_derivatives(self):


        x0 = self.domain_extrema[1, 0]
        xL = self.domain_extrema[1, 1]

        input_sb_0, input_sb_L, exact_sb_Tf_0, exact_sb_Tf_L = self.add_spatial_boundary_points()  

        input_sb_0.requires_grad = True
        input_sb_L.requires_grad = True

        Tf_0 = self.approximate_solution(input_sb_0)[:,0]
        Tf_L = self.approximate_solution(input_sb_L)[:,0]
        Ts_0 = self.approximate_solution(input_sb_0)[:,1]
        Ts_L = self.approximate_solution(input_sb_L)[:,1]    

        grad_Tf_L = torch.autograd.grad(Tf_L.sum(), input_sb_L, create_graph=True)[0]
        grad_Tf_0 = torch.autograd.grad(Tf_0.sum(), input_sb_0, create_graph=True)[0]
        grad_Ts_0 = torch.autograd.grad(Ts_0.sum(), input_sb_0, create_graph=True)[0]
        grad_Ts_L = torch.autograd.grad(Ts_L.sum(), input_sb_L, create_graph=True)[0]

        grad_Tf_t_0 = grad_Tf_0[:, 0]
        grad_Tf_x_0 = grad_Tf_0[:, 1]
        grad_Tf_t_L = grad_Tf_L[:, 0]
        grad_Tf_x_L = grad_Tf_L[:, 1]

        grad_Ts_t_0 = grad_Ts_0[:, 0]
        grad_Ts_x_0 = grad_Ts_0[:, 1]
        grad_Ts_t_L = grad_Ts_L[:, 0]
        grad_Ts_x_L = grad_Ts_L[:, 1]


        input_sb_Tf_0_np_sorted, grad_Tf_x_0_np_sorted = zip(*sorted(zip(input_sb_0[:, 0].detach().numpy(), grad_Tf_x_0.detach().numpy())))
        input_sb_Tf_0_np_sorted, grad_Tf_t_0_np_sorted = zip(*sorted(zip(input_sb_0[:, 0].detach().numpy(), grad_Tf_t_0.detach().numpy())))
      
        input_sb_Tf_L_np_sorted, grad_Tf_x_L_np_sorted = zip(*sorted(zip(input_sb_L[:, 0].detach().numpy(), grad_Tf_x_L.detach().numpy())))
        input_sb_Tf_L_np_sorted, grad_Tf_t_L_np_sorted = zip(*sorted(zip(input_sb_L[:, 0].detach().numpy(), grad_Tf_t_L.detach().numpy())))


        input_sb_Ts_0_np_sorted, grad_Ts_x_0_np_sorted = zip(*sorted(zip(input_sb_0[:, 0].detach().numpy(), grad_Ts_x_0.detach().numpy())))
        input_sb_Ts_0_np_sorted, grad_Ts_t_0_np_sorted = zip(*sorted(zip(input_sb_0[:, 0].detach().numpy(), grad_Ts_t_0.detach().numpy())))

        input_sb_Ts_L_np_sorted, grad_Ts_x_L_np_sorted = zip(*sorted(zip(input_sb_L[:, 0].detach().numpy(), grad_Ts_x_L.detach().numpy())))
        input_sb_Ts_L_np_sorted, grad_Ts_t_L_np_sorted = zip(*sorted(zip(input_sb_L[:, 0].detach().numpy(), grad_Ts_t_L.detach().numpy())))



        err_der_Tf_0 = (torch.mean((grad_Tf_x_0) ** 2)) ** 0.5 * 100
        print("L2 Error derivatice sb x=0 Tf: ", err_der_Tf_0.item(), "%")
        err_der_Tf_L = (torch.mean((grad_Tf_x_L) ** 2)) ** 0.5 * 100
        print("L2 Error derivative sb x=L Tf: ", err_der_Tf_L.item(), "%")

        err_der_Ts_0 = (torch.mean((grad_Ts_x_0) ** 2)) ** 0.5 * 100
        print("L2 Error derivative sb x=0 Ts: ", err_der_Ts_0.item(), "%")
        err_der_Ts_L = (torch.mean((grad_Ts_x_L) ** 2)) ** 0.5 * 100
        print("L2 Error derivative sb x=L Ts: ", err_der_Ts_L.item(), "%")


        fig, axs = plt.subplots(2, 2, figsize=(16, 8), dpi=150)

        im1 = axs[0,0].plot(input_sb_Ts_0_np_sorted, grad_Ts_x_0_np_sorted)
        axs[0,0].set_ylabel("Derivata")
        axs[0,0].grid(True, which="both", ls=":")

        im2 = axs[1,0].plot(input_sb_Ts_L_np_sorted, grad_Ts_x_L_np_sorted)
        axs[1,0].set_xlabel("t")
        axs[1,0].set_ylabel("Derivata")
        axs[1,0].grid(True, which="both", ls=":")


        im3 = axs[0,1].plot(input_sb_Tf_0_np_sorted, grad_Tf_x_0_np_sorted)
        axs[0,1].set_ylabel("Derivata")
        axs[0,1].grid(True, which="both", ls=":")


        im4 = axs[1,1].plot(input_sb_Tf_L_np_sorted, grad_Tf_x_L_np_sorted)
        axs[1,1].set_xlabel("t")
        axs[1,1].set_ylabel("Derivata")
        axs[1,1].grid(True, which="both", ls=":")


        axs[0,0].set_title("Derivata Ts per x a 0; " + str(round(err_der_Ts_0.item(),2))+"%")
        axs[1,0].set_title("Derivata Ts per x a L; " + str(round(err_der_Ts_L.item(),2))+"%")
        axs[0,1].set_title("Derivata Tf per x a 0; " + str(round(err_der_Tf_0.item(),2))+"%")
        axs[1,1].set_title("Derivata Tf per x a L; " + str(round(err_der_Tf_L.item(),2))+"%")

        #plt.savefig(self.dir + "/der_solutions_" + name + f"_{datetime.datetime.now()}" + ".png")
        plt.savefig(self.dir + "/der_solutions_" + name + ".png")


    def plotting_losses(self, history, history_sb_Tf, history_sb_Ts, history_meas, history_int):

        fig, axs = plt.subplots(1, 2, figsize=(16, 8), dpi=150)
        axs[0].plot(np.arange(1, len(history) + 1), history, label="Train Loss")
        axs[0].legend(loc='best')
        axs[0].grid(True, which="both", ls=":")

        axs[1].plot(np.arange(1, len(history_sb_Tf) + 1), history_sb_Tf, label="Train Loss SB Tf")
        axs[1].plot(np.arange(1, len(history_sb_Ts) + 1), history_sb_Ts, label="Train Loss SB Ts")

        axs[1].plot(np.arange(1, len(history_meas) + 1), history_meas, label="Train Loss MEAS Tf")
        axs[1].plot(np.arange(1, len(history_int) + 1), history_int, label="Train Loss INT")

        axs[1].legend(loc='best')
        axs[1].grid(True, which="both", ls=":")

        #plt.savefig(self.dir + "/losses_" + f"2i_{datetime.datetime.now()}" + ".png")
        plt.savefig(self.dir + "/losses_" + name + ".png")








################################################################################################################################################################################
################################################################################################################################################################################
################################################################################################################################################################################
################################################################################################################################################################################
################################################################################################################################################################################
################################################################################################################################################################################     

factor = 1      
        
n_int = 512*factor
n_sb = 128*factor
n_tb = 128*factor
To = 1
Th = 4
Tc = 1

pinn = Pinns(n_int, n_sb, n_tb, To, Th, Tc)


pinn.plotting_assemble_datasets()

print("------")
print(pinn.dir)
print("------")


n_epochs = 1
optimizer_LBFGS = optim.LBFGS(list(pinn.approximate_solution.parameters()),
                              lr=float(0.5),
                              max_iter=10,
                              max_eval=100000,
                              history_size=150,
                              line_search_fn="strong_wolfe",
                              tolerance_change=1.0 * np.finfo(float).eps)             

optimizer_ADAM = optim.Adam(list(pinn.approximate_solution.parameters()),
                            lr=float(0.005),)
                            
history, history_sb_Tf, history_sb_Ts, history_meas, history_int = pinn.fit(num_epochs=n_epochs,
                optimizer=optimizer_LBFGS,
                verbose=True)

#torch.save(pinn.approximate_solution.state_dict(), pinn.dir + "/models/" + name + f"_model_{datetime.datetime.now()}")


pinn.plotting_losses(history, history_sb_Tf, history_sb_Ts, history_meas, history_int)

pinn.plotting_solutions()

pinn.plotting_derivatives()

pinn.plottting_bc()
