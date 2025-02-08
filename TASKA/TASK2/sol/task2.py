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

        if name == "1c":
            self.t_min = 0
            self.t_max = 1
        elif name == "2i":
            self.t_min = 1
            self.t_max = 2
        elif name == "3d":
            self.t_min = 2
            self.t_max = 3
        elif name == "4i":
            self.t_min = 3
            self.t_max = 4
        elif name == "5c":
            self.t_min = 4
            self.t_max = 5
        elif name == "6i":
            self.t_min = 5
            self.t_max = 6
        elif name == "7d":
            self.t_min = 6
            self.t_max = 7
        elif name == "8i":
            self.t_min = 7
            self.t_max = 8

        
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
        
        self.training_set_sb_0, self.training_set_sb_L, self.training_set_tb, self.training_set_int, self.training_set_m = self.assemble_datasets()

        self.train_meas_input, self.train_meas_Tf, self.train_meas_points = self.get_measurement_data()



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

    def initial_condition(self, x):
        return torch.full(x.shape, self.To)
    

########################################################################################
########################################################################################

    def add_temporal_boundary_points(self):
        t0 = self.domain_extrema[0, 0]
        input_tb = self.normalize(self.convert(self.soboleng.draw(self.n_tb)))
        input_tb[:, 0] = torch.full(input_tb[:, 0].shape, t0)

        input_tb = input_tb[~(((input_tb[:,0] == 0.) & (input_tb[:,1] == 0))) ]         #only for charge 1 i get rid of 0,0

        output_tb_Tf = self.initial_condition(input_tb[:, 1]).reshape(-1,1)
        output_tb_Ts = torch.clone(output_tb_Tf)

        return input_tb, output_tb_Tf, output_tb_Ts   #size input_tb [nx2], size output_tb_Tf [nx1]


    def add_spatial_boundary_points(self):
        x0 = self.domain_extrema[1, 0]
        xL = self.domain_extrema[1, 1]

        input_sb = self.normalize(self.convert(self.soboleng.draw(self.n_sb)))

        input_sb_0 = torch.clone(input_sb)
        input_sb_0[:, 1] = torch.full(input_sb_0[:, 1].shape, x0)

        if name == "1c":
            input_sb_0 = input_sb_0[~(((input_sb_0[:,0] == 0.) & (input_sb_0[:,1] == 0))) ]         #only for charge 1 i get rid of 0,0



        input_sb_L = torch.clone(input_sb)
        input_sb_L[:, 1] = torch.full(input_sb_L[:, 1].shape, xL)

        output_sb_Tf_0 = torch.full(input_sb_0[:,0].shape, self.Th)
        output_sb_Tf_L = torch.full(input_sb_L[:,0].shape, self.Tc)

        return input_sb_0, input_sb_L, output_sb_Tf_0.reshape(-1, 1), output_sb_Tf_L.reshape(-1, 1)  #size input_sb [nx2], size output_sb_Tf [n]


    def add_interior_points(self):
        input_int = self.normalize(self.convert(self.soboleng.draw(self.n_int)))
        input_int = input_int[~((input_int[:,1] == 0) | (input_int[:,1] == 1))]       #get rid of points on boundary

        output_int = torch.zeros((input_int.shape[0], 1))
        return input_int, output_int.reshape(-1, 1)  #size input_sb [nx2], size output_sb_Tf [nx1]
    

    def get_measurement_data(self):

        file = self.dir + "/DataSolution.txt"

        data = np.genfromtxt(file, delimiter=",", skip_header=1)

        data = data[~(((data[:,0] == 8.0) & (data[:,1] == 0.02)) | ((data[:,0] == 0.0) & (data[:,1] == 0.02)))]

        #data = data[np.random.choice(data.shape[0], 10000, replace=False), :]

        if phase_type == "i":
            phase = (self.t_min < data[:,0]) & (data[:,0] < self.t_max)

        elif phase_type == "c":
            phase = (self.t_min <= data[:,0]) & (data[:,0] <= self.t_max)

        elif phase_type == "d":
            phase = (self.t_min <= data[:,0]) & (data[:,0] <= self.t_max)
 


        n_points_m = data.shape[0]

        t = torch.tensor(data[:,0][phase], dtype=torch.float32)
        x = torch.tensor(data[:,1][phase], dtype=torch.float32)
        Tf_m = torch.tensor(data[:,2][phase], dtype=torch.float32)  

        input_m = self.normalize(torch.stack([t, x], 1))

        return input_m, Tf_m.reshape(-1, 1), n_points_m  #size input_m [nx2], size Tf_m [nx1]


    def assemble_datasets(self):

        input_sb_0, input_sb_L, output_sb_Tf_0, output_sb_Tf_L = self.add_spatial_boundary_points()  
        input_tb, output_tb_Tf, output_tb_Ts = self.add_temporal_boundary_points()  
        input_int, output_int = self.add_interior_points()  
        input_m, Tf_m, n_points_m = self.get_measurement_data()                                                                                    #???????????????? Do I do a DataLoader su 40000000 ???

        """
        print("°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°")
        print("ASSEMBLE DATASET")
        print("input_sb_0, input_sb_L, output_sb_Tf_0, output_sb_Tf_L")
        print(input_sb_0.size(), input_sb_L.size(), output_sb_Tf_0.size(), output_sb_Tf_L.size())
        print("input_tb, output_tb_Tf, output_tb_Ts")
        print(input_tb.size(), output_tb_Tf.size(), output_tb_Ts.size() )
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
        
        print("input_tb | output_tb_Tf | output_tb_Ts")
        for i, j in enumerate(input_tb[:,0].detach().numpy()):
            print(input_tb.detach().numpy()[i], " | ",  output_tb_Tf.reshape(-1,).detach().numpy()[i], " | ",  output_tb_Ts.reshape(-1,).detach().numpy()[i])

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
        """

        training_set_sb_0 = DataLoader(torch.utils.data.TensorDataset(input_sb_0, output_sb_Tf_0), batch_size=self.n_sb, shuffle=False)                                        #ANY PARAMETERS (BATCH) HERE HAS TO CHANGE??????
        training_set_sb_L = DataLoader(torch.utils.data.TensorDataset(input_sb_L, output_sb_Tf_L), batch_size=self.n_sb, shuffle=False)
        training_set_tb = DataLoader(torch.utils.data.TensorDataset(input_tb, output_tb_Tf, output_tb_Ts), batch_size=self.n_tb, shuffle=False)
        training_set_int = DataLoader(torch.utils.data.TensorDataset(input_int, output_int), batch_size=self.n_int, shuffle=False)
        training_set_m = DataLoader(torch.utils.data.TensorDataset(input_m, Tf_m), batch_size=n_points_m, shuffle=False)                                                        


        return training_set_sb_0, training_set_sb_L, training_set_tb, training_set_int, training_set_m


########################################################################################
########################################################################################

    def apply_initial_condition(self, input_tb):
        pred_tb_Tf = self.approximate_solution(input_tb)[:,0]
        pred_tb_Ts = self.approximate_solution(input_tb)[:,1]
        return pred_tb_Tf.reshape(-1,1), pred_tb_Ts.reshape(-1,1)   #size u_pred_tb [nx1]
    

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
            residual = grad_Tf_t + self.v*grad_Tf_x - 0.005 * grad_Tf_xx + 5 * (Tf - Ts)

        elif phase_type == "c":
            residual = grad_Tf_t + self.v*grad_Tf_x - 0.005 * grad_Tf_xx + 5 * (Tf - Ts)

        elif phase_type == "d":
            residual = grad_Tf_t + self.v*grad_Tf_x - 0.005 * grad_Tf_xx + 5 * (Tf - Ts)

        return residual.reshape(-1)

    def compute_loss(self, input_sb_0, output_sb_Tf_0, input_sb_L, output_sb_Tf_L, input_tb, output_tb_Tf, output_tb_Ts, input_int, output_int, input_m, Tf_m, verbose=True):

        pred_sb_Tf_0, grad_Tf_x_sb_0, grad_Ts_x_sb_0 = self.apply_boundary_conditions(input_sb_0) #size pred_sb_Tf [nx1], size grad_Tf_x_sb [n]
        pred_sb_Tf_L, grad_Tf_x_sb_L, grad_Ts_x_sb_L = self.apply_boundary_conditions(input_sb_L) #size pred_sb_Tf [nx1], size grad_Tf_x_sb [n]
        
        pred_tb_Tf, pred_tb_Ts = self.apply_initial_condition(input_tb) #size pred_tb_Tf [nx1]
            
        #INIZIALIZZANDO I DATI IN __init__
        train_meas_input, train_meas_Tf = self.train_meas_input, self.train_meas_Tf
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
        assert (pred_tb_Tf.shape == output_tb_Tf.shape)
        assert (pred_tb_Ts.shape == output_tb_Ts.shape)

        assert (grad_Tf_x_sb_0.shape[0] == input_sb_0.shape[0])
        assert (grad_Ts_x_sb_0.shape[0] == input_sb_0.shape[0])
        assert (grad_Tf_x_sb_L.shape[0] == input_sb_L.shape[0])
        assert (grad_Ts_x_sb_L.shape[0] == input_sb_L.shape[0])
        

########################################################################################

        if phase_type == "i":
            r_sb_Tf_0 = grad_Tf_x_sb_0
            r_sb_Tf_L = grad_Tf_x_sb_L

        elif phase_type == "c":
            r_sb_Tf_0 = pred_sb_Tf_0 - output_sb_Tf_0
            r_sb_Tf_L = grad_Tf_x_sb_L

        elif phase_type == "d":
            r_sb_Tf_0 = grad_Tf_x_sb_0
            r_sb_Tf_L = pred_sb_Tf_L - output_sb_Tf_L
        

        loss_sb_Tf_0 = torch.mean(abs(r_sb_Tf_0) ** 2)
        loss_sb_Tf_L = torch.mean(abs(r_sb_Tf_L) ** 2)

        loss_sb_Tf = loss_sb_Tf_0  + loss_sb_Tf_L



########################################################################################

        
        r_sb_Ts_0 = grad_Ts_x_sb_0
        r_sb_Ts_L = grad_Ts_x_sb_L

        loss_sb_Ts_0 = torch.mean(abs(r_sb_Ts_0)**2)
        loss_sb_Ts_L = torch.mean(abs(r_sb_Ts_L)**2)

        loss_sb_Ts = loss_sb_Ts_0 + loss_sb_Ts_L
                

########################################################################################

        
        r_tb_Tf = output_tb_Tf - pred_tb_Tf
        loss_tb_Tf = torch.mean(abs(r_tb_Tf) ** 2)
        
        r_tb_Ts = output_tb_Ts - pred_tb_Ts
        loss_tb_Ts = torch.mean(abs(r_tb_Ts) ** 2)
        
        loss_tb = loss_tb_Tf
        

########################################################################################


        r_meas = pred_Tf_m - train_meas_Tf
        
        loss_meas = torch.mean(abs(r_meas) ** 2)


########################################################################################


        r_int = self.compute_pde_residual(input_int)

        loss_int = torch.mean(abs(r_int) ** 2)


########################################################################################

        if name == "1c":
            loss_tsb = loss_sb_Tf + loss_tb

        else:
            loss_tsb = loss_sb_Tf

        loss = torch.log10( 0.3*loss_int  + loss_meas )
        if verbose: print("Iteration:", name, self.iteration, "||", "Total loss: ", round(loss.item(), 4), "| Loss_meas: ", round(torch.log10(loss_meas).item(), 4), "| Loss_int: ", round(torch.log10(loss_int).item(), 4), "| Loss_sb_Tf: ", round(torch.log10(loss_sb_Tf).item(), 4), "| Loss_sb_Ts: ", round(torch.log10(loss_sb_Ts).item(), 4), "| Loss_tb: ", round(torch.log10(loss_tb).item(), 4))
        self.iteration = self.iteration +1

        return loss, torch.log10(loss_tb), torch.log10(loss_sb_Tf), torch.log10(loss_sb_Ts), torch.log10(loss_meas), torch.log10(loss_int)




########################################################################################

    def fit(self, num_epochs, optimizer, verbose=True):
        history = list()

        history_tb = list()

        history_sb_Tf = list()
        history_sb_Ts = list()

        history_meas = list()
        history_int = list()

        for epoch in range(num_epochs):
            if verbose: print("################################ ", epoch, " ################################")

            for j, ((input_sb_0, output_sb_Tf_0), (input_sb_L, output_sb_Tf_L), (input_tb, output_tb_Tf, output_tb_Ts), (input_int, output_int), (input_m, Tf_m)) in enumerate(zip(self.training_set_sb_0, self.training_set_sb_L, self.training_set_tb, self.training_set_int, self.training_set_m)):
                def closure():
                    optimizer.zero_grad()
                    loss, loss_tb, loss_sb_Tf, loss_sb_Ts, loss_meas, loss_int = self.compute_loss(input_sb_0, output_sb_Tf_0, input_sb_L, output_sb_Tf_L, input_tb, output_tb_Tf, output_tb_Ts, input_int, output_int, input_m, Tf_m, verbose=verbose)
                    loss.backward()


                    history.append(loss.item())
                    
                    history_tb.append(loss_tb.item())
                    history_sb_Tf.append(loss_sb_Tf.item())
                    history_sb_Ts.append(loss_sb_Ts.item())

                    history_meas.append(loss_meas.item())
                    history_int .append(loss_int.item())

                    return loss

                optimizer.step(closure=closure)

        print('Final Loss: ', history[-1])
        

        return history, history_tb, history_sb_Tf, history_sb_Ts, history_meas, history_int


########################################################################################
########################################################################################
########################################################################################
########################################################################################


    def plotting_assemble_datasets(self):

        
        input_sb_0, input_sb_L, output_sb_Tf_0, output_sb_Tf_L = self.add_spatial_boundary_points()  
        input_tb, output_tb_Tf, output_tb_Ts = self.add_temporal_boundary_points()  

        input_int, output_int = self.add_interior_points()  
        input_m, Tf_m, n_points_m = self.get_measurement_data()

        inputs =  torch.cat([input_sb_0, input_sb_L, input_int ], 0) 
        
        outputs = torch.cat([output_sb_Tf_0, output_sb_Tf_L, output_int], 0)


        fig, axs = plt.subplots(1, 2, figsize=(16, 8), dpi=150)
        im1 = axs[0].scatter(inputs[:, 1].detach().numpy(), inputs[:, 0].detach().numpy(), c=outputs.detach().numpy(), cmap="jet")
        axs[0].set_xlabel("x")
        axs[0].set_ylabel("t")
        plt.colorbar(im1, ax=axs[0])
        axs[0].grid(True, which="both", ls=":")

        im2 = axs[1].scatter(input_m[:, 1].detach().numpy(), input_m[:, 0].detach().numpy(), c=Tf_m, cmap="jet")
        axs[1].set_xlabel("x")
        axs[1].set_ylabel("t")
        plt.colorbar(im2, ax=axs[1])
        axs[1].grid(True, which="both", ls=":")
        axs[0].set_title("Tf")
        axs[1].set_title("Tf MEAS")

        plt.savefig(self.dir + "/plots/assemble_dataset_" + name + f"_{datetime.datetime.now()}" + ".png")


        input_sb_0_np_sorted, output_sb_Tf_0_np_sorted = zip(*sorted(zip(input_sb_0[:, 0].detach().numpy(), output_sb_Tf_0.detach().numpy())))
        input_sb_L_np_sorted, output_sb_Tf_L_np_sorted = zip(*sorted(zip(input_sb_L[:, 0].detach().numpy(), output_sb_Tf_L.detach().numpy())))


        fig2, axs2 = plt.subplots(1, 2, figsize=(16, 8), dpi=150)
        axs2[0].plot(input_sb_0_np_sorted, output_sb_Tf_0_np_sorted, label="BC at x=0")
        axs2[0].plot(input_sb_L_np_sorted, output_sb_Tf_L_np_sorted, label="BC at x=L")
        axs2[0].set_xlabel("t")
        axs2[0].set_ylabel("T")
        axs2[0].legend(loc='best')
        axs2[0].grid(True, which="both", ls=":")

        input_tb_np_sorted, output_tb_np_sorted = zip(*sorted(zip(input_tb[:, 1].detach().numpy(),output_tb_Tf.detach().numpy())))


        axs2[1].plot(input_tb_np_sorted, output_tb_np_sorted, label="IC at t=0")
        axs2[1].set_xlabel("x")
        axs2[1].set_ylabel("T")
        axs2[1].grid(True, which="both", ls=":")

        axs2[0].set_title("BC")
        axs2[1].set_title("IC")



    def plotting_solutions(self):

        train_meas_input, train_meas_Tf = self.train_meas_input, self.train_meas_Tf


        output = self.approximate_solution(train_meas_input)
        output_Tf = output[:,0]
        output_Ts = output[:,1]

        err = (torch.mean((self.approximate_solution(train_meas_input)[:,0] - train_meas_Tf.reshape(-1,)) ** 2) / torch.mean(train_meas_Tf.reshape(-1,) ** 2)) ** 0.5 * 100
        print("L2 Relative Error Norm Tf: ", err.item(), "%")

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
        axs[0].set_title("Tf Relative Error Norm to MEAS: " + str(round(err.item(),2)) + " % " + f" | {datetime.datetime.now()}")
        axs[1].set_title("Ts")

        plt.savefig(self.dir + "/plots/solutions_" + name + f"_{datetime.datetime.now()}" + ".png")


    def plottting_bc(self):

        input_sb_0, input_sb_L, exact_sb_Tf_0, exact_sb_Tf_L = self.add_spatial_boundary_points()  



        t0 = self.domain_extrema[0, 0]
        x0 = self.domain_extrema[1, 0]
        xL = self.domain_extrema[1, 1]


        output_sb_Tf_0 = self.approximate_solution(input_sb_0)[:,0]
        output_sb_Tf_L = self.approximate_solution(input_sb_L)[:,0]


        input_sb_0_np_sorted, output_sb_Tf_0_np_sorted = zip(*sorted(zip(input_sb_0[:, 0].detach().numpy(), output_sb_Tf_0.detach().numpy())))
        input_sb_L_np_sorted, output_sb_Tf_L_np_sorted = zip(*sorted(zip(input_sb_L[:, 0].detach().numpy(), output_sb_Tf_L.detach().numpy())))


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
            plt.savefig(self.dir + "/plots/SBC_solutions_" + name  + f"_{datetime.datetime.now()}" + ".png")


    def plotting_ic(self):
        input_tb, exact_tb_Tf, exact_tb_Ts = self.add_temporal_boundary_points()  

        inputs = self.normalize(self.convert(self.soboleng.draw(10000)))

        t0 = self.domain_extrema[0, 0]
        x0 = self.domain_extrema[1, 0]
        xL = self.domain_extrema[1, 1]

        output_tb_Tf = self.approximate_solution(input_tb)[:,0]
        output_tb_Ts = self.approximate_solution(input_tb)[:,1]


        err_tb_Tf = (torch.mean((output_tb_Tf - exact_tb_Tf) ** 2) / torch.mean(exact_tb_Tf.float() ** 2)) ** 0.5 * 100
        print("L2 Relative Error Norm tb Tf: ", err_tb_Tf.item(), "%")

        input_tb_Tf_np_sorted, output_tb_np_sorted = zip(*sorted(zip(input_tb[:, 1].detach().numpy(), output_tb_Tf.detach().numpy())))
        input_tb_Ts_np_sorted, output_tb_np_sorted = zip(*sorted(zip(input_tb[:, 1].detach().numpy(), output_tb_Ts.detach().numpy())))

        fig2, axs2 = plt.subplots(1, 2, figsize=(16, 8), dpi=150)
        axs2[0].plot(input_tb_Tf_np_sorted, output_tb_np_sorted)
        axs2[0].set_xlabel("x")
        axs2[0].set_ylabel("Tf")
        axs2[0].grid(True, which="both", ls=":")

        axs2[1].plot(input_tb_Ts_np_sorted, output_tb_np_sorted)
        axs2[1].set_xlabel("x")
        axs2[1].set_ylabel("Tf")
        axs2[1].grid(True, which="both", ls=":")

        axs2[0].set_title("IC Tf at t = 0; " + str(round(err_tb_Tf.item(),2)) + "%")
        axs2[1].set_title("IC Ts at t = 0")


        plt.savefig(self.dir + "/plots/IC_solution_" + name + f"_{datetime.datetime.now()}" + ".png")


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

        plt.savefig(self.dir + "/plots/der_solutions_" + name + f"_{datetime.datetime.now()}" + ".png")


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


        plt.savefig(self.dir + "/plots/losses_" + name + f"_{datetime.datetime.now()}" + ".png")

    def plotting_losses_1c(self, history, history_tb, history_sb_Tf, history_sb_Ts, history_meas, history_int):

        fig, axs = plt.subplots(1, 2, figsize=(16, 8), dpi=150)
        axs[0].plot(np.arange(1, len(history) + 1), history, label="Train Loss")
        axs[0].legend(loc='best')
        axs[0].grid(True, which="both", ls=":")

        axs[1].plot(np.arange(1, len(history_tb) + 1), history_tb, label="Train Loss TB")
        axs[1].plot(np.arange(1, len(history_sb_Tf) + 1), history_sb_Tf, label="Train Loss SB Tf")
        axs[1].plot(np.arange(1, len(history_sb_Ts) + 1), history_sb_Ts, label="Train Loss SB Ts")

        axs[1].plot(np.arange(1, len(history_meas) + 1), history_meas, label="Train Loss MEAS Tf")
        axs[1].plot(np.arange(1, len(history_int) + 1), history_int, label="Train Loss INT")

        axs[1].legend(loc='best')
        axs[1].grid(True, which="both", ls=":")

        plt.savefig(self.dir + "/plots/losses_" + name + f"_{datetime.datetime.now()}" + ".png")





########################################################################################

    def submission_file(self):


        file = self.dir + '/SubExample.txt'

        data = np.genfromtxt(file, delimiter=",", skip_header=1)

        phase = (self.t_min <= data[:,0]) & (data[:,0] <= self.t_max) 

        if phase_type == "i":
            phase = (self.t_min < data[:,0]) & (data[:,0] < self.t_max)

        elif phase_type == "8i":
            phase = (self.t_min < data[:,0]) & (data[:,0] <= self.t_max)

        elif phase_type == "c":
            phase = (self.t_min <= data[:,0]) & (data[:,0] <= self.t_max)

        elif phase_type == "d":
            phase = (self.t_min <= data[:,0]) & (data[:,0] <= self.t_max)


        t = torch.tensor(data[:,0][phase], dtype=torch.float32)
        x = torch.tensor(data[:,1][phase], dtype=torch.float32)

        inputs = self.normalize(torch.stack([t, x], 1))


        outputs_Ts = self.approximate_solution(inputs)[:,1]
        outputs_Ts = outputs_Ts.detach().numpy()

        t = data[:,0][phase]
        x = data[:,1][phase]

        combined_array = np.column_stack((t, x, outputs_Ts))

        np.savetxt(self.dir + "/submission/submissionTask2_LUCA_SACCHI_" + name + f"_{datetime.datetime.now()}" + ".txt", combined_array, delimiter=",", header="t,x,ts")
        np.savetxt(self.dir + "/submission/submissionTask2_LUCA_SACCHI_" + name + "_final.txt", combined_array, delimiter=",", header="t,x,ts")


    def save_model(self):
        torch.save(self.approximate_solution.state_dict(), self.dir + "/models/" + name + f"_model_{datetime.datetime.now()}")




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

n_epochs = 1

max_iter = 2000
max_eval = 100000


for i in range(8):
    num = i+1
    

    if num == 1:
        phase_type = "c"
        name = str(num) + phase_type

        pinn1c = Pinns(n_int, n_sb, n_tb, To, Th, Tc)

        pinn1c.plotting_assemble_datasets()

        optimizer_LBFGS_1c = optim.LBFGS(list(pinn1c.approximate_solution.parameters()),
                                      lr=float(0.5),
                                      max_iter=max_iter,
                                      max_eval=max_eval,
                                      history_size=150,
                                      line_search_fn="strong_wolfe",
                                      tolerance_change=1.0 * np.finfo(float).eps)             

                                    
        history, history_tb, history_sb_Tf, history_sb_Ts, history_meas, history_int = pinn1c.fit(num_epochs=n_epochs,
                        optimizer=optimizer_LBFGS_1c,
                        verbose=True)

        pinn1c.save_model()

        pinn1c.submission_file()

        pinn1c.plotting_losses_1c(history, history_tb, history_sb_Tf, history_sb_Ts, history_meas, history_int)

        pinn1c.plotting_solutions()

        pinn1c.plottting_bc()

        pinn1c.plotting_ic()

        pinn1c.plotting_derivatives()

    if num == 2:
        phase_type = "i"
        name = str(num) + phase_type

        pinn2i = Pinns(n_int, n_sb, n_tb, To, Th, Tc)

        pinn2i.plotting_assemble_datasets()

        optimizer_LBFGS_2i = optim.LBFGS(list(pinn2i.approximate_solution.parameters()),
                                      lr=float(0.5),
                                      max_iter=max_iter,
                                      max_eval=max_eval,
                                      history_size=150,
                                      line_search_fn="strong_wolfe",
                                      tolerance_change=1.0 * np.finfo(float).eps)             

                                    
        history, history_tb, history_sb_Tf, history_sb_Ts, history_meas, history_int = pinn2i.fit(num_epochs=n_epochs,
                        optimizer=optimizer_LBFGS_2i,
                        verbose=True)

        pinn2i.save_model()

        pinn2i.submission_file()

        pinn2i.plotting_losses(history, history_sb_Tf, history_sb_Ts, history_meas, history_int)

        pinn2i.plotting_solutions()

        pinn2i.plottting_bc()

        pinn2i.plotting_derivatives()

    elif num == 3:
        phase_type = "d"
        name = str(num) + phase_type

        pinn3d = Pinns(n_int, n_sb, n_tb, To, Th, Tc)

        pinn3d.plotting_assemble_datasets()

        optimizer_LBFGS_3d = optim.LBFGS(list(pinn3d.approximate_solution.parameters()),
                                      lr=float(0.5),
                                      max_iter=max_iter,
                                      max_eval=max_eval,
                                      history_size=150,
                                      line_search_fn="strong_wolfe",
                                      tolerance_change=1.0 * np.finfo(float).eps)             

                                    
        history, history_tb, history_sb_Tf, history_sb_Ts, history_meas, history_int = pinn3d.fit(num_epochs=n_epochs,
                        optimizer=optimizer_LBFGS_3d,
                        verbose=True)

        pinn3d.save_model()

        pinn3d.submission_file()

        pinn3d.plotting_losses(history, history_sb_Tf, history_sb_Ts, history_meas, history_int)

        pinn3d.plotting_solutions()

        pinn3d.plottting_bc()

        pinn3d.plotting_derivatives()

    elif num == 4:
        phase_type = "i"
        name = str(num) + phase_type

        pinn4i = Pinns(n_int, n_sb, n_tb, To, Th, Tc)

        pinn4i.plotting_assemble_datasets()

        optimizer_LBFGS_4i = optim.LBFGS(list(pinn4i.approximate_solution.parameters()),
                                      lr=float(0.5),
                                      max_iter=max_iter,
                                      max_eval=max_eval,
                                      history_size=150,
                                      line_search_fn="strong_wolfe",
                                      tolerance_change=1.0 * np.finfo(float).eps)             

                                    
        history, history_tb, history_sb_Tf, history_sb_Ts, history_meas, history_int = pinn4i.fit(num_epochs=n_epochs,
                        optimizer=optimizer_LBFGS_4i,
                        verbose=True)

        pinn4i.save_model()

        pinn4i.submission_file()

        pinn4i.plotting_losses(history, history_sb_Tf, history_sb_Ts, history_meas, history_int)

        pinn4i.plotting_solutions()

        pinn4i.plottting_bc()

        pinn4i.plotting_derivatives()


    elif num == 5:
        phase_type = "c"
        name = str(num) + phase_type

        pinn5c = Pinns(n_int, n_sb, n_tb, To, Th, Tc)

        pinn5c.plotting_assemble_datasets()

        optimizer_LBFGS_5c = optim.LBFGS(list(pinn5c.approximate_solution.parameters()),
                                      lr=float(0.5),
                                      max_iter=max_iter,
                                      max_eval=max_eval,
                                      history_size=150,
                                      line_search_fn="strong_wolfe",
                                      tolerance_change=1.0 * np.finfo(float).eps)             

                                    
        history, history_tb, history_sb_Tf, history_sb_Ts, history_meas, history_int = pinn5c.fit(num_epochs=n_epochs,
                        optimizer=optimizer_LBFGS_5c,
                        verbose=True)

        pinn5c.save_model()

        pinn5c.submission_file()

        pinn5c.plotting_losses(history, history_sb_Tf, history_sb_Ts, history_meas, history_int)

        pinn5c.plotting_solutions()

        pinn5c.plottting_bc()

        pinn5c.plotting_derivatives()

    elif num == 6:
        phase_type = "i"
        name = str(num) + phase_type

        pinn6i = Pinns(n_int, n_sb, n_tb, To, Th, Tc)

        pinn6i.plotting_assemble_datasets()

        optimizer_LBFGS_6i = optim.LBFGS(list(pinn6i.approximate_solution.parameters()),
                                      lr=float(0.5),
                                      max_iter=max_iter,
                                      max_eval=max_eval,
                                      history_size=150,
                                      line_search_fn="strong_wolfe",
                                      tolerance_change=1.0 * np.finfo(float).eps)             

                                    
        history, history_tb, history_sb_Tf, history_sb_Ts, history_meas, history_int = pinn6i.fit(num_epochs=n_epochs,
                        optimizer=optimizer_LBFGS_6i,
                        verbose=True)

        pinn6i.save_model()

        pinn6i.submission_file()

        pinn6i.plotting_losses(history, history_sb_Tf, history_sb_Ts, history_meas, history_int)

        pinn6i.plotting_solutions()

        pinn6i.plottting_bc()

        pinn6i.plotting_derivatives()

    elif num == 7:
        phase_type = "d"
        name = str(num) + phase_type

        pinn7d = Pinns(n_int, n_sb, n_tb, To, Th, Tc)

        pinn7d.plotting_assemble_datasets()

        optimizer_LBFGS_7d = optim.LBFGS(list(pinn7d.approximate_solution.parameters()),
                                      lr=float(0.5),
                                      max_iter=max_iter,
                                      max_eval=max_eval,
                                      history_size=150,
                                      line_search_fn="strong_wolfe",
                                      tolerance_change=1.0 * np.finfo(float).eps)             

                                    
        history, history_tb, history_sb_Tf, history_sb_Ts, history_meas, history_int = pinn7d.fit(num_epochs=n_epochs,
                        optimizer=optimizer_LBFGS_7d,
                        verbose=True)

        pinn7d.save_model()

        pinn7d.submission_file()

        pinn7d.plotting_losses(history, history_sb_Tf, history_sb_Ts, history_meas, history_int)

        pinn7d.plotting_solutions()

        pinn7d.plottting_bc()

        pinn7d.plotting_derivatives()

    elif num == 8:
        phase_type = "i"
        name = str(num) + phase_type

        pinn8i = Pinns(n_int, n_sb, n_tb, To, Th, Tc)

        pinn8i.plotting_assemble_datasets()

        optimizer_LBFGS_8i = optim.LBFGS(list(pinn8i.approximate_solution.parameters()),
                                      lr=float(0.5),
                                      max_iter=max_iter,
                                      max_eval=max_eval,
                                      history_size=150,
                                      line_search_fn="strong_wolfe",
                                      tolerance_change=1.0 * np.finfo(float).eps)             

                                    
        history, history_tb, history_sb_Tf, history_sb_Ts, history_meas, history_int = pinn8i.fit(num_epochs=n_epochs,
                        optimizer=optimizer_LBFGS_8i,
                        verbose=True)

        pinn8i.save_model()

        pinn8i.submission_file()

        pinn8i.plotting_losses(history, history_sb_Tf, history_sb_Ts, history_meas, history_int)

        pinn8i.plotting_solutions()

        pinn8i.plottting_bc()

        pinn8i.plotting_derivatives()

directory = os.path.dirname(os.path.realpath(__file__))
file = directory + '/SubExample.txt'

data = np.genfromtxt(file, delimiter=",", skip_header=1)

t = torch.tensor(data[:,0], dtype=torch.float32)
x = torch.tensor(data[:,1], dtype=torch.float32)

inputs = pinn1c.normalize(torch.stack([t, x], 1))

outputs_Ts = []

for i,j in enumerate(data[:,0]):
    print(i)
    if j <=1:
        outputs_Ts.append(pinn1c.approximate_solution(inputs[i])[1].item())
    elif 1 < j and j < 2:
        outputs_Ts.append(pinn2i.approximate_solution(inputs[i])[1].item())
    elif 2 <= j and j <=3:
        outputs_Ts.append(pinn3d.approximate_solution(inputs[i])[1].item())
    elif 3 < j and j < 4:
        outputs_Ts.append(pinn4i.approximate_solution(inputs[i])[1].item())
    elif 4 < j and j < 5:
        outputs_Ts.append(pinn5c.approximate_solution(inputs[i])[1].item())
    elif 5 < j and j < 6:
        outputs_Ts.append(pinn6i.approximate_solution(inputs[i])[1].item())
    elif 6 < j and j < 7:
        outputs_Ts.append(pinn7d.approximate_solution(inputs[i])[1].item())
    elif 7 < j and j <= 8:
        outputs_Ts.append(pinn8i.approximate_solution(inputs[i])[1].item())


outputs_Ts = np.array(outputs_Ts)


t = data[:,0]
x = data[:,1]

combined_array = np.column_stack((t, x, outputs_Ts))

np.savetxt(directory + "/submissionTask2_LUCA_SACCHI.txt", combined_array, delimiter=",", header="t,x,ts")


plt.scatter(t, x, c=outputs_Ts, cmap="jet", vmax=5)
plt.savefig(directory +  f"finalplot_{datetime.datetime.now()}" + ".png")