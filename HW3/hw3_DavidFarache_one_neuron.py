import random
import numpy as np
import matplotlib.pyplot as plt
import operator

seed = 0           
random.seed(seed)
np.random.seed(seed)

from ComputationalGraphPrimer import *

class ComputationGraphPrimerUpdate(ComputationalGraphPrimer):

    def run_training_loop_one_neuron_model(self, training_data, mu=0.7, beta1=0.9, beta2=0.99, epsilon=1e-6):
        """
        The training loop must first initialize the learnable parameters.  Remember, these are the 
        symbolic names in your input expressions for the neural layer that do not begin with the 
        letter 'x'.  In this case, we are initializing with random numbers from a uniform distribution 
        over the interval (0,1).
        """
        self.vals_for_learnable_params = {param: random.uniform(0,1) for param in self.learnable_params}

        self.bias = random.uniform(0,1)                   ## Adding the bias improves class discrimination.
                                                        ##   We initialize it to a random number.

        ###########################################Modified Code####################################################  
        # Add for SG Optimization
        self.mu = mu
        self.step = 0
        self.bias_change = 0
        
        # Added for Atom Optimization
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m_t = 0
        self.v_t = 0
        self.m_td =  0
        self.v_td = 0 
        ############################################################################################################

        class DataLoader:
            """
            To understand the logic of the dataloader, it would help if you first understand how 
            the training dataset is created.  Search for the following function in this file:

                            gen_training_data(self)
        
            As you will see in the implementation code for this method, the training dataset
            consists of a Python dict with two keys, 0 and 1, the former points to a list of 
            all Class 0 samples and the latter to a list of all Class 1 samples.  In each list,
            the data samples are drawn from a multi-dimensional Gaussian distribution.  The two
            classes have different means and variances.  The dimensionality of each data sample
            is set by the number of nodes in the input layer of the neural network.

            The data loader's job is to construct a batch of samples drawn randomly from the two
            lists mentioned above.  And it mush also associate the class label with each sample
            separately.
            """
            def __init__(self, training_data, batch_size):
                self.training_data = training_data
                self.batch_size = batch_size
                self.class_0_samples = [(item, 0) for item in self.training_data[0]]   ## Associate label 0 with each sample
                self.class_1_samples = [(item, 1) for item in self.training_data[1]]   ## Associate label 1 with each sample

            def __len__(self):
                return len(self.training_data[0]) + len(self.training_data[1])

            def _getitem(self):    
                cointoss = random.choice([0,1])                            ## When a batch is created by getbatch(), we want the
                                                                        ##   samples to be chosen randomly from the two lists
                if cointoss == 0:
                    return random.choice(self.class_0_samples)
                else:
                    return random.choice(self.class_1_samples)            

            def getbatch(self):
                batch_data,batch_labels = [],[]                            ## First list for samples, the second for labels
                maxval = 0.0                                               ## For approximate batch data normalization
                for _ in range(self.batch_size):
                    item = self._getitem()
                    if np.max(item[0]) > maxval: 
                        maxval = np.max(item[0])
                    batch_data.append(item[0])
                    batch_labels.append(item[1])
                batch_data = [item/maxval for item in batch_data]          ## Normalize batch data
                batch = [batch_data, batch_labels]
                return batch                


        data_loader = DataLoader(training_data, batch_size=self.batch_size)
        loss_running_record = []
        i = 0
        avg_loss_over_iterations = 0.0                                    ##  Average the loss over iterations for printing out 
                                                                        ##    every N iterations during the training loop.
        for i in range(self.training_iterations):
            data = data_loader.getbatch()
            data_tuples = data[0]
            class_labels = data[1]
            y_preds, deriv_sigmoids =  self.forward_prop_one_neuron_model(data_tuples)              ##  FORWARD PROP of data
            loss = sum([(abs(class_labels[i] - y_preds[i]))**2 for i in range(len(class_labels))])  ##  Find loss
            loss_avg = loss / float(len(class_labels))                                              ##  Average the loss over batch
            avg_loss_over_iterations += loss_avg                          
            if i%(self.display_loss_how_often) == 0: 
                avg_loss_over_iterations /= self.display_loss_how_often
                loss_running_record.append(avg_loss_over_iterations)
                print("[iter=%d]  loss = %.4f" %  (i+1, avg_loss_over_iterations))                 ## Display average loss
                avg_loss_over_iterations = 0.0                                                     ## Re-initialize avg loss
            y_errors = list(map(operator.sub, class_labels, y_preds))
            y_error_avg = sum(y_errors) / float(len(class_labels))
            deriv_sigmoid_avg = sum(deriv_sigmoids) / float(len(class_labels))
            data_tuple_avg = [sum(x) for x in zip(*data_tuples)]
            data_tuple_avg = list(map(operator.truediv, data_tuple_avg, 
                                    [float(len(class_labels))] * len(class_labels) ))
        ##########################################Modified Code###################################################
            if self.optimizer == 'SGD':
                self.backprop_and_update_params_one_neuron_model(y_error_avg, data_tuple_avg, deriv_sigmoid_avg)     ## BACKPROP loss
            if self.optimizer == 'SGD+':
                self.SGD_plus(y_error_avg, data_tuple_avg, deriv_sigmoid_avg)     ## BACKPROP loss
            if self.optimizer == 'Adam':
                self.Adam(y_error_avg, data_tuple_avg, deriv_sigmoid_avg, i + 1)     ## BACKPROP loss
        ##########################################################################################################
        #print(self.optimizer)

        # plt.figure()     
        # plt.plot(loss_running_record) 
        # plt.show()  
        #  
        return loss_running_record

    def forward_prop_one_neuron_model(self, data_tuples_in_batch):
        """
        Forward propagates the batch data through the neural network according to the equations on
        Slide 50 of my Week 3 slides.

        As the one-neuron model is characterized by a single expression, the main job of this function is
        to evaluate that expression for each data tuple in the incoming batch.  The resulting output is
        fed into the sigmoid activation function and the partial derivative of the sigmoid with respect
        to its input calculated.
        """
        output_vals = []
        deriv_sigmoids = []
        for vals_for_input_vars in data_tuples_in_batch:
            input_vars = self.independent_vars                   ## This is a list of vars for the input nodes. For the
                                                                ##   the One-Neuron example in the Examples directory
                                                                ##   this is just the list [xa, xb, xc, xd]
            vals_for_input_vars_dict =  dict(zip(input_vars, list(vals_for_input_vars)))   ## The current values at input

            exp_obj = self.exp_objects[0]                        ## To understand this, first see the definition of the
                                                                ##   Exp class (search for the string "class Exp").
                                                                ##   Each expression that defines the neural network is
                                                                ##   represented by one Exp instance by the parser.
            output_val = self.eval_expression(exp_obj.body , vals_for_input_vars_dict, self.vals_for_learnable_params)

            ## [Search for "self.bias" in this file.]  As mentioned earlier, adding bias improves class discrimination:
            output_val = output_val + self.bias

            output_val = 1.0 / (1.0 + np.exp(-1.0 * output_val))   ## Apply sigmoid activation (output confined to [0.0,1.0] interval) 

            deriv_sigmoid = output_val * (1.0 - output_val)        ## See Slide 59 for why we need partial deriv of Sigmoid at input point

            output_vals.append(output_val)                         ## Collect output values for different input samples in batch

            deriv_sigmoids.append(deriv_sigmoid)                   ## Collect the Sigmoid derivatives for each input sample in batch
                                                                ##   The derivatives that are saved during forward prop are shown on Slide 59.
        return output_vals, deriv_sigmoids

    def backprop_and_update_params_one_neuron_model(self, y_error, vals_for_input_vars, deriv_sigmoid):
        """
        As should be evident from the syntax used in the following call to backprop function,

        self.backprop_and_update_params_one_neuron_model( y_error_avg, data_tuple_avg, deriv_sigmoid_avg)
                                                                    ^^^             ^^^                ^^^
        the values fed to the backprop function for its three arguments are averaged over the training 
        samples in the batch.  This in keeping with the spirit of SGD that calls for averaging the 
        information retained in the forward propagation over the samples in a batch.

        See Slide 59 of my Week 3 slides for the math of back propagation for the One-Neuron network.
        """
        input_vars = self.independent_vars
        input_vars_to_param_map = self.var_to_var_param[self.output_vars[0]]
        param_to_vars_map = {param : var for var, param in input_vars_to_param_map.items()}
        vals_for_input_vars_dict =  dict(zip(input_vars, list(vals_for_input_vars)))
        vals_for_learnable_params = self.vals_for_learnable_params
        for i,param in enumerate(self.vals_for_learnable_params):
            ## Calculate the next step in the parameter hyperplane
#            step = self.learning_rate * y_error * vals_for_input_vars_dict[input_vars[i]] * deriv_sigmoid   
            step = self.learning_rate * y_error * vals_for_input_vars_dict[param_to_vars_map[param]] * deriv_sigmoid    
            ## Update the learnable parameters
            self.vals_for_learnable_params[param] += step
        self.bias += self.learning_rate * y_error * deriv_sigmoid    ## Update the bias

    def SGD_plus(self, y_error, vals_for_input_vars, deriv_sigmoid):
        """
        As should be evident from the syntax used in the following call to backprop function,

        self.backprop_and_update_params_one_neuron_model( y_error_avg, data_tuple_avg, deriv_sigmoid_avg)
                                                                    ^^^             ^^^                ^^^
        the values fed to the backprop function for its three arguments are averaged over the training 
        samples in the batch.  This in keeping with the spirit of SGD that calls for averaging the 
        information retained in the forward propagation over the samples in a batch.

        See Slide 59 of my Week 3 slides for the math of back propagation for the One-Neuron network.
        """
        input_vars = self.independent_vars
        vals_for_input_vars_dict =  dict(zip(input_vars, list(vals_for_input_vars)))
        vals_for_learnable_params = self.vals_for_learnable_params

        for i,param in enumerate(self.vals_for_learnable_params):
            ## Calculate the next step in the parameter hyperplane
        ###########################################Modified Code###############################################################################################  
            self.step = (self.mu * self.step) + (self.learning_rate * y_error * vals_for_input_vars_dict[input_vars[i]] * deriv_sigmoid)
            ## Update the learnable parameters
            self.vals_for_learnable_params[param] += self.step

        self.bias_change = (self.mu * self.bias_change) + (self.learning_rate * y_error * deriv_sigmoid)   ## Update the bias
        self.bias += self.bias_change
        ###########################################Modified Code###############################################################################################

    def Adam(self, y_error, vals_for_input_vars, deriv_sigmoid, iter):
        """
        As should be evident from the syntax used in the following call to backprop function,

            self.backprop_and_update_params_one_neuron_model( y_error_avg, data_tuple_avg, deriv_sigmoid_avg)
                                                                        ^^^             ^^^                ^^^
        the values fed to the backprop function for its three arguments are averaged over the training 
        samples in the batch.  This in keeping with the spirit of SGD that calls for averaging the 
        information retained in the forward propagation over the samples in a batch.

        See Slide 59 of my Week 3 slides for the math of back propagation for the One-Neuron network.
        """
        input_vars = self.independent_vars
        vals_for_input_vars_dict =  dict(zip(input_vars, list(vals_for_input_vars)))
        vals_for_learnable_params = self.vals_for_learnable_params
        for i,param in enumerate(self.vals_for_learnable_params):
        ###########################################Modified Code###############################################################################################
        # Code refrence from: https://towardsdatascience.com/how-to-implement-an-adam-optimizer-from-scratch-76e7b217f1cc

            ## Calculate momentum gradient
            self.m_td =  (self.beta1 * self.m_td) + (1 - self.beta1) * (self.learning_rate * y_error * vals_for_input_vars_dict[input_vars[i]] * deriv_sigmoid)
            self.v_td =  (self.beta2 * self.v_td) + (1 - self.beta2) * (self.learning_rate * y_error * vals_for_input_vars_dict[input_vars[i]] * deriv_sigmoid)**2

            mk_hat = self.m_td / (1 - self.beta1 ** iter) 
            vk_hat = self.v_td / (1 - self.beta2 ** iter)

            ## Update the learnable parameters
            self.vals_for_learnable_params[param] += mk_hat / np.sqrt(vk_hat + self.epsilon)
        
        ## Calculate momentum 1 and 2
        self.m_t = (self.beta1 * self.m_t) + (1 - self.beta1) * (self.learning_rate * y_error * deriv_sigmoid)
        self.v_t = (self.beta2 * self.v_t) + (1 - self.beta2) * (self.learning_rate * y_error * deriv_sigmoid) ** 2

        ## Get values to update bias
        m_hat = self.m_t / (1 - self.beta1 ** iter) 
        v_hat = self.v_t / (1 - self.beta2 ** iter) 

        self.bias += m_hat / np.sqrt(v_hat + self.epsilon) ## Update the bias 
        ################################################################################################################################################
       

######################################################################################################

def runningOpt(optName):
    loss_iteration_list = []
    for opt in optName:
        cgp = ComputationGraphPrimerUpdate(
                one_neuron_model = True,
                expressions = ['xw=ab*xa+bc*xb+cd*xc+ac*xd'],
                output_vars = ['xw'],
                dataset_size = 5000,
                learning_rate = 1e-4, #1e-4
    #               learning_rate = 5 * 1e-2,
                training_iterations = 40000,
                batch_size = 8,
                display_loss_how_often = 100,
                debug = True,
                optimizer = opt
        )

        cgp.parse_expressions()
        # cgp.display_one_neuron_network()      

        training_data = cgp.gen_training_data()
        loss_per_iteration = cgp.run_training_loop_one_neuron_model( training_data )
        #print(loss_per_iteration)
        loss_iteration_list.append(loss_per_iteration)
    return loss_iteration_list

optLoss = runningOpt(['SGD', 'SGD+', 'Adam'])

plt.plot(range(len(optLoss[0])), optLoss[0], label="SGD loss")
plt.plot(range(len(optLoss[1])), optLoss[1], label="SGD+ loss")
plt.plot(range(len(optLoss[2])), optLoss[2], label="Adam loss")

plt.title("One Neuron Optimization Methods")
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.legend(loc="upper right")

plt.show()