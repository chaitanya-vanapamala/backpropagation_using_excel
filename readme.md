[//]: ![formula](https://render.githubusercontent.com/render/math?math=\color{red}\Huge%20\frac{\partial%20f}{\partial%20x})

# Backpropagation through Excel - END2

In this, I will try to demonstrate the working of backpropagation in neural netwroks. Let's take a neural network with 1 hidden layer, and we will do a binary classification using L2 Loss function.

![Network Architecture](https://raw.githubusercontent.com/chaitanya-vanapamala/backpropagation_using_excel/main/network_arch.png)

The above is the network, we are going to compute the gradients and do backpropagation for this network. The initial weight values are shown in the above image.

Now let's write down all the equations for outputs. While h1, h2, o1, o2 are weighted sum of respective inputs, a_h1, a_h2, a_o1 and a_o2 are the activate output of weighted sums.

### **Hidden Layer equations**
#
![formula](https://render.githubusercontent.com/render/math?math=\color{Green}\large%20h_{1}%20=%20W_{1}%2Ai_{1}%20%2B%20W_{2}%2Ai_{2})

![formula](https://render.githubusercontent.com/render/math?math=\color{Green}\large%20h_{2}%20=%20W_{3}%2Ai_{1}%20%2B%20W_{4}%2Ai_{2})

![formula](https://render.githubusercontent.com/render/math?math=\color{Green}\large%20%5Ctext%7Ba_h%7D_1%20%3D%20%5Csigma%28h_1%29%20%3D%20%5Cfrac%7B1%7D%7B1%2Be%5E%7B-h_1%7D%7D)

![formula](https://render.githubusercontent.com/render/math?math=\color{Green}\large%20%5Ctext%7Ba_h%7D_2%20%3D%20%5Csigma%28h_2%29%20%3D%20%5Cfrac%7B1%7D%7B1%2Be%5E%7B-h_2%7D%7D)

### **Output Layer equations**
#
![formula](https://render.githubusercontent.com/render/math?math=\color{blue}\large%20o_1%20%3D%20W_5%2A%5Ctext%7Ba_h%7D_1%2BW_6%2A%5Ctext%7Ba_h%7D_2)

![formula](https://render.githubusercontent.com/render/math?math=\color{blue}\large%20o_2%20%3D%20W_7%2A%5Ctext%7Ba_h%7D_1%2BW_8%2A%5Ctext%7Ba_h%7D_2)

![formula](https://render.githubusercontent.com/render/math?math=\color{blue}\large%20%5Ctext%7Ba_o%7D_1%20%3D%20%5Csigma%28o_1%29%20%3D%20%5Cfrac%7B1%7D%7B1%2Be%5E%7B-o_1%7D%7D)

![formula](https://render.githubusercontent.com/render/math?math=\color{blue}\large%20%5Ctext%7Ba_o%7D_2%20%3D%20%5Csigma%28o_2%29%20%3D%20%5Cfrac%7B1%7D%7B1%2Be%5E%7B-o_2%7D%7D)

### **Loss equations**
#
The loss function we are going to use is L2 Loss function i.e, Mean Squared Error.

![formula](https://render.githubusercontent.com/render/math?math=\color{red}\large%20E_1%3D%5Cfrac%7B1%7D%7B2%7D%2A%28t_1-%5Ctext%7Ba_o%7D_1%29%5E2)

![formula](https://render.githubusercontent.com/render/math?math=\color{red}\large%20E_2%3D%5Cfrac%7B1%7D%7B2%7D%2A%28t_2-%5Ctext%7Ba_o%7D_2%29%5E2)

![formula](https://render.githubusercontent.com/render/math?math=\color{red}\large%20%5Ctext%7BE_Total%7D%3DE_1%2BE_2)

*Note:* **t1 and t2 are target outputs**

## **Finding Gradients**
#
Let's start computing the gradient of weights from output layer weights.

### **Loss Gradient with respect to W<sub>5</sub>**
![formula](https://render.githubusercontent.com/render/math?math=\color{red}\Large%20%5Cfrac%7B%5Cpartial%5Ctext%7B%20E_Total%7D%7D%7B%5Cpartial%20W_5%7D%3D%5Cfrac%7B%5Cpartial%20%28E_1%2BE_2%29%7D%7B%5Cpartial%20W_5%7D)

The **W<sub>5</sub>** weights is not connected to **E<sub>2</sub>** Node, but connected to **E<sub>1</sub>** So we can directly compute derivative for **E<sub>1</sub>** with respect to **W<sub>5</sub>**.

![\frac{\partial\text{ E_Total}}{\partial W_5}=\frac{\partial (E_1)}{\partial W_5}](https://render.githubusercontent.com/render/math?math=\color{red}\Large%20%5Cfrac%7B%5Cpartial%5Ctext%7B%20E_Total%7D%7D%7B%5Cpartial%20W_5%7D%3D%5Cfrac%7B%5Cpartial%20%28E_1%29%7D%7B%5Cpartial%20W_5%7D)

Now let's expand the derivative using chain rule.

![\frac{\partial E_1}{\partial W_5}=\frac{\partial E_1}{\partial \text{a_o}_1}*\frac{\partial \text{a_o}_1}{\partial o_1}*\frac{\partial o_1}{\partial W_5}](https://render.githubusercontent.com/render/math?math=\color{red}\Large%20%5Cfrac%7B%5Cpartial%20E_1%7D%7B%5Cpartial%20W_5%7D%3D%5Cfrac%7B%5Cpartial%20E_1%7D%7B%5Cpartial%20%5Ctext%7Ba_o%7D_1%7D%2A%5Cfrac%7B%5Cpartial%20%5Ctext%7Ba_o%7D_1%7D%7B%5Cpartial%20o_1%7D%2A%5Cfrac%7B%5Cpartial%20o_1%7D%7B%5Cpartial%20W_5%7D)

Let's compute the individual gradients from the above equation.

![\frac{\partial E_1}{\partial \text{a_o}_1}=\frac{\partial (\frac{1}{2}*(t_1-\text{a_o}_1)^2)}{\partial \text{a_o}_1}=\frac{1}{2}*\frac{\partial ((t_1-\text{a_o}_1)^2)}{\partial \text{a_o}_1}](https://render.githubusercontent.com/render/math?math=\color{red}\Large%20%5Cfrac%7B%5Cpartial%20E_1%7D%7B%5Cpartial%20%5Ctext%7Ba_o%7D_1%7D%3D%5Cfrac%7B%5Cpartial%20%28%5Cfrac%7B1%7D%7B2%7D%2A%28t_1-%5Ctext%7Ba_o%7D_1%29%5E2%29%7D%7B%5Cpartial%20%5Ctext%7Ba_o%7D_1%7D%3D%5Cfrac%7B1%7D%7B2%7D%2A%5Cfrac%7B%5Cpartial%20%28%28t_1-%5Ctext%7Ba_o%7D_1%29%5E2%29%7D%7B%5Cpartial%20%5Ctext%7Ba_o%7D_1%7D)

![=\frac{1}{2}*2*(t_1-\text{a_o}_1)*\frac{\partial (t_1-\text{a_o}_1)}{\partial \text{a_o}_1}=(t_1-\text{a_o}_1)*(-1)](https://render.githubusercontent.com/render/math?math=\color{red}\Large%20%3D%5Cfrac%7B1%7D%7B2%7D%2A2%2A%28t_1-%5Ctext%7Ba_o%7D_1%29%2A%5Cfrac%7B%5Cpartial%20%28t_1-%5Ctext%7Ba_o%7D_1%29%7D%7B%5Cpartial%20%5Ctext%7Ba_o%7D_1%7D%3D%28t_1-%5Ctext%7Ba_o%7D_1%29%2A%28-1%29)

![\frac{\partial E_1}{\partial \text{a_o}_1}=\text{a_o}_1-t_1 \qquad\qquad \cdots\cdots(1)](https://render.githubusercontent.com/render/math?math=\color{red}\Large%20%5Cfrac%7B%5Cpartial%20E_1%7D%7B%5Cpartial%20%5Ctext%7Ba_o%7D_1%7D%3D%5Ctext%7Ba_o%7D_1-t_1%20%5Cqquad%5Cqquad%20%5Ccdots%5Ccdots%281%29) 

Now let's see the other derivative terms.

![\frac{\partial \text{a_o}_1}{\partial o_1}=\frac{\partial \sigma(o_1)}{\partial o_1}](https://render.githubusercontent.com/render/math?math=\color{red}\Large%20%5Cfrac%7B%5Cpartial%20%5Ctext%7Ba_o%7D_1%7D%7B%5Cpartial%20o_1%7D%3D%5Cfrac%7B%5Cpartial%20%5Csigma%28o_1%29%7D%7B%5Cpartial%20o_1%7D)

As we know that the derivative of sigmoid from the following link [Derivation of sigmoid](https://math.stackexchange.com/questions/78575/derivative-of-sigmoid-function-sigma-x-frac11e-x), let's include it into the above equation.

![\frac{\partial \text{a_o}_1}{\partial o_1}=\frac{\partial \sigma(o_1)}{\partial o_1}=\sigma(o_1)*(1-\sigma(o_1))](https://render.githubusercontent.com/render/math?math=\color{red}\Large%20%5Cfrac%7B%5Cpartial%20%5Ctext%7Ba_o%7D_1%7D%7B%5Cpartial%20o_1%7D%3D%5Cfrac%7B%5Cpartial%20%5Csigma%28o_1%29%7D%7B%5Cpartial%20o_1%7D%3D%5Csigma%28o_1%29%2A%281-%5Csigma%28o_1%29%29)

From the forward equations

![\frac{\partial \text{a_o}_1}{\partial o_1}=\sigma(o_1)*(1-\sigma(o_1)) = \text{a_o}_1*(1-\text{a_o}_1)\qquad \cdots\cdots(2)](https://render.githubusercontent.com/render/math?math=\color{red}\Large%20%5Cfrac%7B%5Cpartial%20%5Ctext%7Ba_o%7D_1%7D%7B%5Cpartial%20o_1%7D%3D%5Csigma%28o_1%29%2A%281-%5Csigma%28o_1%29%29%20%3D%20%5Ctext%7Ba_o%7D_1%2A%281-%5Ctext%7Ba_o%7D_1%29%5Cqquad%20%5Ccdots%5Ccdots%282%29)

And final term in loss gradeint with respect to W5, let's expand the o<sub>1</sub> term from forward equations.

![\frac{\partial o_1}{\partial W_5}=\frac{\partial (W_5*\text{a_h}_1+W6*\text{a_h}_2)}{\partial W_5}](https://render.githubusercontent.com/render/math?math=\color{red}\Large%20%5Cfrac%7B%5Cpartial%20o_1%7D%7B%5Cpartial%20W_5%7D%3D%5Cfrac%7B%5Cpartial%20%28W_5%2A%5Ctext%7Ba_h%7D_1%2BW6%2A%5Ctext%7Ba_h%7D_2%29%7D%7B%5Cpartial%20W_5%7D)

![\frac{\partial o_1}{\partial W_5}=\text{a_h}_1\qquad \cdots\cdots(3)](https://render.githubusercontent.com/render/math?math=\color{red}\Large%20%5Cfrac%7B%5Cpartial%20o_1%7D%7B%5Cpartial%20W_5%7D%3D%5Ctext%7Ba_h%7D_1%5Cqquad%20%5Ccdots%5Ccdots%283%29)

Let's put equations 1,2 and 3 together.

![\frac{\partial E_1}{\partial W_5}=(\text{a_o}_1-t_1)*\text{a_o}_1*(1-\text{a_o}_1)*\text{a_h}_1](https://render.githubusercontent.com/render/math?math=\color{RED}\huge%20%5Cfrac%7B%5Cpartial%20E_1%7D%7B%5Cpartial%20W_5%7D%3D%28%5Ctext%7Ba_o%7D_1-t_1%29%2A%5Ctext%7Ba_o%7D_1%2A%281-%5Ctext%7Ba_o%7D_1%29%2A%5Ctext%7Ba_h%7D_1)

### **Loss Gradient with respect to W<sub>6</sub>**
Same as **W<sub>5</sub>** weights **W<sub>6</sub>** also not connected to **E<sub>2</sub>** Node, but connected to **E<sub>1</sub>** So we can directly compute derivative for **E<sub>1</sub>** with respect to **W<sub>6</sub>**.

![\frac{\partial\text{ E_Total}}{\partial W_6}=\frac{\partial (E_1)}{\partial W_6}](https://render.githubusercontent.com/render/math?math=\color{red}\Large%20%5Cfrac%7B%5Cpartial%5Ctext%7B%20E_Total%7D%7D%7B%5Cpartial%20W_6%7D%3D%5Cfrac%7B%5Cpartial%20%28E_1%29%7D%7B%5Cpartial%20W_6%7D)

Now let's expand it using chain rule.

![\frac{\partial E_1}{\partial W_6}=\frac{\partial E_1}{\partial \text{a_o}_1}\frac{\partial \text{a_o}_1}{\partial o_1}\frac{\partial o_1}{\partial W_6}](https://render.githubusercontent.com/render/math?math=\color{red}\Large%20%5Cfrac%7B%5Cpartial%20E_1%7D%7B%5Cpartial%20W_6%7D%3D%5Cfrac%7B%5Cpartial%20E_1%7D%7B%5Cpartial%20%5Ctext%7Ba_o%7D_1%7D%5Cfrac%7B%5Cpartial%20%5Ctext%7Ba_o%7D_1%7D%7B%5Cpartial%20o_1%7D%5Cfrac%7B%5Cpartial%20o_1%7D%7B%5Cpartial%20W_6%7D)

The first 2 terms in the above equation we have already found out and can be retrived from equation 1 and 2, let's find out 3rd derivative.

![\frac{\partial o_1}{\partial W_6}=\frac{\partial (W_5*\text{a_h}_1+W6*\text{a_h}_2)}{\partial W_6}](https://render.githubusercontent.com/render/math?math=\color{red}\Large%20%5Cfrac%7B%5Cpartial%20o_1%7D%7B%5Cpartial%20W_6%7D%3D%5Cfrac%7B%5Cpartial%20%28W_5%2A%5Ctext%7Ba_h%7D_1%2BW6%2A%5Ctext%7Ba_h%7D_2%29%7D%7B%5Cpartial%20W_6%7D)

![\frac{\partial o_1}{\partial W_6}=\text{a_h}_2\qquad \cdots\cdots(4)](https://render.githubusercontent.com/render/math?math=\color{red}\Large%20%5Cfrac%7B%5Cpartial%20o_1%7D%7B%5Cpartial%20W_6%7D%3D%5Ctext%7Ba_h%7D_2%5Cqquad%20%5Ccdots%5Ccdots%284%29)

And let's Combine 1,2, and 4 together.

![\frac{\partial E_1}{\partial W_6}=(\text{a_o}_1-t_1)*\text{a_o}_1*(1-\text{a_o}_1)*\text{a_h}_1](https://render.githubusercontent.com/render/math?math=\color{RED}\huge%20%5Cfrac%7B%5Cpartial%20E_1%7D%7B%5Cpartial%20W_6%7D%3D%28%5Ctext%7Ba_o%7D_1-t_1%29%2A%5Ctext%7Ba_o%7D_1%2A%281-%5Ctext%7Ba_o%7D_1%29%2A%5Ctext%7Ba_h%7D_2)

