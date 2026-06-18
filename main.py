
import numpy as np

Inputs_lst=[[1,0],[0,1],[0,0],[1,1]]
Outputs_lst=[[0],[0],[0],[1]]
Inputs_Array=np.array(Inputs_lst)
Outputs_Array=np.array(Outputs_lst)

Weights=np.zeros([2,1])
Bias=np.zeros([1,1])

def Neuron_Output(Weights,Bias,Inputs):
  z= (np.dot(Inputs,Weights))+Bias
  return z

def Sigmoid(z):
  return 1/(1+(np.exp(-z)))

def Sigmoid_derivative(Predictions):
  return predictions*(1-predictions)

def Error_Calculation(Prediction,Outputs):
  return Outputs-Prediction

learning_rate=0.1
for epoch in range(5000):

  # Forward Propagation
  A=Neuron_Output(Weights,Bias,Inputs_Array)
  predictions=Sigmoid(A)

  # Loss Calculation
  Error=Error_Calculation(predictions,Outputs_Array)
  loss=np.mean(np.square(Error))

  # Back Propagation
  Output_delta=Error*Sigmoid_derivative(predictions)
  weight_gradient=np.dot(Inputs_Array.T,Output_delta)
  bias_gradient=np.sum(Output_delta)

  # Weight Update
  Weights+=learning_rate*weight_gradient
  Bias+=learning_rate*bias_gradient


  if(epoch%1000==0):
    print(f"Epoch={epoch:4d}---->Loss={loss:.6f}")
print("Training completed")

def Test(User_Inputs):
  Input_Array=np.array([User_Inputs])
  Neuron_out=Neuron_Output(Weights,Bias,User_Inputs)
  Prediction=Sigmoid(Neuron_out)
  answer=1 if Prediction>0.5 else 0
  print(answer)

Test([1,0])
Test([1,1])
Test([0,1])
Test([0,0])
