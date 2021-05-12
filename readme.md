[//]: ![formula](https://render.githubusercontent.com/render/math?math=\color{red}\Huge%20\frac{\partial%20f}{\partial%20x})

# Backpropagation through Excel - END2

In this, I will try to demonstrate the working of backpropagation in neural netwroks. Let's take a neural network with 1 hidden layer, and we will do a binary classification using L2 Loss function.

![Network Architecture](https://raw.githubusercontent.com/chaitanya-vanapamala/backpropagation_using_excel/main/network_arch.png)

The above is the network, we are going to compute the gradients and do backpropagation for this network. The initial weight values are shown in the above image.

Now let's write down all the equations for outputs. While h1, h2, o1, o2 are weighted sum of respective inputs, a_h1, a_h2, a_o1 and a_o2 are the activate output of weighted sums.

### Hidden Layer equations
![formula](https://render.githubusercontent.com/render/math?math=\color{Green}\large%20h_{1}%20=%20W_{1}%2Ai_{1}%20%2B%20W_{2}%2Ai_{2})

![formula](https://render.githubusercontent.com/render/math?math=\color{Green}\large%20h_{2}%20=%20W_{3}%2Ai_{1}%20%2B%20W_{4}%2Ai_{2})

![formula](https://render.githubusercontent.com/render/math?math=\color{Green}\large%20%5Ctext%7Ba_h%7D_1%20%3D%20%5Csigma%28h_1%29%20%3D%20%5Cfrac%7B1%7D%7B1%2Be%5E%7B-h_1%7D%7D)

![formula](https://render.githubusercontent.com/render/math?math=\color{Green}\large%20%5Ctext%7Ba_h%7D_2%20%3D%20%5Csigma%28h_2%29%20%3D%20%5Cfrac%7B1%7D%7B1%2Be%5E%7B-h_2%7D%7D)

### **Output Layer equations**
#
![formula](https://render.githubusercontent.com/render/math?math=\color{Royalblue}\large%20o_1%20%3D%20W_5%2A%5Ctext%7Ba_h%7D_1%2BW_6%2A%5Ctext%7Ba_h%7D_2)

![formula](https://render.githubusercontent.com/render/math?math=\color{Royalblue}\large%20o_2%20%3D%20W_7%2A%5Ctext%7Ba_h%7D_1%2BW_8%2A%5Ctext%7Ba_h%7D_2)

![formula](https://render.githubusercontent.com/render/math?math=\color{Royalblue}\large%20%5Ctext%7Ba_o%7D_1%20%3D%20%5Csigma%28o_1%29%20%3D%20%5Cfrac%7B1%7D%7B1%2Be%5E%7B-o_1%7D%7D)

![formula](https://render.githubusercontent.com/render/math?math=\color{Royalblue}\large%20%5Ctext%7Ba_o%7D_2%20%3D%20%5Csigma%28o_2%29%20%3D%20%5Cfrac%7B1%7D%7B1%2Be%5E%7B-o_2%7D%7D)

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

## **Gradients of Output Layer Weights**
#
### **Loss Gradient with respect to W<sub>5</sub>**
![formula](https://render.githubusercontent.com/render/math?math=\color{red}\Large%20%5Cfrac%7B%5Cpartial%5Ctext%7B%20E_Total%7D%7D%7B%5Cpartial%20W_5%7D%3D%5Cfrac%7B%5Cpartial%20%28E_1%2BE_2%29%7D%7B%5Cpartial%20W_5%7D)

The **W<sub>5</sub>** weights is not connected to **E<sub>2</sub>** Node, but connected to **E<sub>1</sub>** So The partial derivative of **E<sub>2</sub>**  with respect to **W<sub>5</sub>** will be zero. and we will be left with only partial derivative of **E<sub>1</sub>** with respect to **W<sub>5</sub>**.

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

![\frac{\partial \text{ E_Total}}{\partial W_5}=(\text{a_o}_1-t_1)*\text{a_o}_1*(1-\text{a_o}_1)*\text{a_h}_1](https://render.githubusercontent.com/render/math?math=\color{purple}\huge%20%5Cfrac%7B%5Cpartial%5Ctext%7B%20E_Total%7D%7D%7B%5Cpartial%20W_5%7D%3D%28%5Ctext%7Ba_o%7D_1-t_1%29%2A%5Ctext%7Ba_o%7D_1%2A%281-%5Ctext%7Ba_o%7D_1%29%2A%5Ctext%7Ba_h%7D_1)

### **Loss Gradient with respect to W<sub>6</sub>**
Same as **W<sub>5</sub>** weights **W<sub>6</sub>** also not connected to **E<sub>2</sub>** Node, but connected to **E<sub>1</sub>** So we can directly compute derivative for **E<sub>1</sub>** with respect to **W<sub>6</sub>**.

![\frac{\partial\text{ E_Total}}{\partial W_6}=\frac{\partial (E_1)}{\partial W_6}](https://render.githubusercontent.com/render/math?math=\color{red}\Large%20%5Cfrac%7B%5Cpartial%5Ctext%7B%20E_Total%7D%7D%7B%5Cpartial%20W_6%7D%3D%5Cfrac%7B%5Cpartial%20%28E_1%29%7D%7B%5Cpartial%20W_6%7D)

Now let's expand it using chain rule.

![\frac{\partial E_1}{\partial W_6}=\frac{\partial E_1}{\partial \text{a_o}_1}\frac{\partial \text{a_o}_1}{\partial o_1}\frac{\partial o_1}{\partial W_6}](https://render.githubusercontent.com/render/math?math=\color{red}\Large%20%5Cfrac%7B%5Cpartial%20E_1%7D%7B%5Cpartial%20W_6%7D%3D%5Cfrac%7B%5Cpartial%20E_1%7D%7B%5Cpartial%20%5Ctext%7Ba_o%7D_1%7D%5Cfrac%7B%5Cpartial%20%5Ctext%7Ba_o%7D_1%7D%7B%5Cpartial%20o_1%7D%5Cfrac%7B%5Cpartial%20o_1%7D%7B%5Cpartial%20W_6%7D)

The first 2 terms in the above equation we have already found out and can be retrived from equation 1 and 2, let's find out 3rd derivative.

![\frac{\partial o_1}{\partial W_6}=\frac{\partial (W_5*\text{a_h}_1+W6*\text{a_h}_2)}{\partial W_6}](https://render.githubusercontent.com/render/math?math=\color{red}\Large%20%5Cfrac%7B%5Cpartial%20o_1%7D%7B%5Cpartial%20W_6%7D%3D%5Cfrac%7B%5Cpartial%20%28W_5%2A%5Ctext%7Ba_h%7D_1%2BW6%2A%5Ctext%7Ba_h%7D_2%29%7D%7B%5Cpartial%20W_6%7D)

![\frac{\partial o_1}{\partial W_6}=\text{a_h}_2\qquad \cdots\cdots(4)](https://render.githubusercontent.com/render/math?math=\color{red}\Large%20%5Cfrac%7B%5Cpartial%20o_1%7D%7B%5Cpartial%20W_6%7D%3D%5Ctext%7Ba_h%7D_2%5Cqquad%20%5Ccdots%5Ccdots%284%29)

And let's Combine 1,2, and 4 together.

![\frac{\partial \text{ E_Total}}{\partial W_6}=(\text{a_o}_1-t_1)*\text{a_o}_1*(1-\text{a_o}_1)*\text{a_h}_1](https://render.githubusercontent.com/render/math?math=\color{purple}\huge%20%5Cfrac%7B%5Cpartial%5Ctext%7B%20E_Total%7D%7D%7B%5Cpartial%20W_6%7D%3D%28%5Ctext%7Ba_o%7D_1-t_1%29%2A%5Ctext%7Ba_o%7D_1%2A%281-%5Ctext%7Ba_o%7D_1%29%2A%5Ctext%7Ba_h%7D_2)

Similarly let's find out W7 and W8 gradients which are connected to o2 node.


### **Loss Gradient with respect to W<sub>7</sub>**
![\frac{\partial\text{ E_Total}}{\partial W_7}=\frac{\partial (E_1+E_2)}{\partial W_7}](https://render.githubusercontent.com/render/math?math=\color{red}\Large%20%5Cfrac%7B%5Cpartial%5Ctext%7B%20E_Total%7D%7D%7B%5Cpartial%20W_7%7D%3D%5Cfrac%7B%5Cpartial%20%28E_1%2BE_2%29%7D%7B%5Cpartial%20W_7%7D)

The **W<sub>7</sub>** weight is not connected to **E<sub>1</sub>** branch, but connected to **E<sub>2</sub>**, So The partial derivative of **E<sub>1</sub>**  with respect to **W<sub>7</sub>** will be zero. and we will be left with only partial derivative of **E<sub>2</sub>** with respect to **W<sub>7</sub>**. And also apply let's apply the chain rule to the partial derivative.

![\frac{\partial E_2}{\partial W_7}=\frac{\partial E_2}{\partial \text{a_o}_2}*\frac{\partial \text{a_o}_2}{\partial o_2}*\frac{\partial o_2}{\partial W_7}](https://render.githubusercontent.com/render/math?math=\color{red}\Large%20%5Cfrac%7B%5Cpartial%20E_2%7D%7B%5Cpartial%20W_7%7D%3D%5Cfrac%7B%5Cpartial%20E_2%7D%7B%5Cpartial%20%5Ctext%7Ba_o%7D_2%7D%2A%5Cfrac%7B%5Cpartial%20%5Ctext%7Ba_o%7D_2%7D%7B%5Cpartial%20o_2%7D%2A%5Cfrac%7B%5Cpartial%20o_2%7D%7B%5Cpartial%20W_7%7D)

Now similar to **W<sub>5</sub>** and **W<sub>6</sub>**, let's find out the partial derivatives of each term individually.

![\frac{\partial E_2}{\partial \text{a_o}_2}=\frac{\partial (\frac{1}{2}*(t_2-\text{a_o}_2)^2)}{\partial \text{a_o}_2}](https://render.githubusercontent.com/render/math?math=\color{red}\Large%20%5Cfrac%7B%5Cpartial%20E_2%7D%7B%5Cpartial%20%5Ctext%7Ba_o%7D_2%7D%3D%5Cfrac%7B%5Cpartial%20%28%5Cfrac%7B1%7D%7B2%7D%2A%28t_2-%5Ctext%7Ba_o%7D_2%29%5E2%29%7D%7B%5Cpartial%20%5Ctext%7Ba_o%7D_2%7D)

![\frac{\partial E_2}{\partial \text{a_o}_2}=\text{a_o}_2-t_2\qquad \cdots(5)](https://render.githubusercontent.com/render/math?math=\color{red}\Large%20%5Cfrac%7B%5Cpartial%20E_2%7D%7B%5Cpartial%20%5Ctext%7Ba_o%7D_2%7D%3D%5Ctext%7Ba_o%7D_2-t_2%5Cqquad%20%5Ccdots%285%29)

![\frac{\partial \text{a_o}_2}{\partial o_2}=\frac{\partial \sigma(o_2)}{\partial o_2}=\sigma(o_2)*(1-\sigma(o_2))](https://render.githubusercontent.com/render/math?math=\color{red}\Large%20%5Cfrac%7B%5Cpartial%20%5Ctext%7Ba_o%7D_2%7D%7B%5Cpartial%20o_2%7D%3D%5Cfrac%7B%5Cpartial%20%5Csigma%28o_2%29%7D%7B%5Cpartial%20o_2%7D%3D%5Csigma%28o_2%29%2A%281-%5Csigma%28o_2%29%29)

![\frac{\partial \text{a_o}_2}{\partial o_2}=\text{a_o}_2*(1-\text{a_o}_2) \qquad \cdots(6)](https://render.githubusercontent.com/render/math?math=\color{red}\Large%20%5Cfrac%7B%5Cpartial%20%5Ctext%7Ba_o%7D_2%7D%7B%5Cpartial%20o_2%7D%3D%5Ctext%7Ba_o%7D_2%2A%281-%5Ctext%7Ba_o%7D_2%29%20%5Cqquad%20%5Ccdots%286%29)

and the final partial derivative, as per the feed forward equations we already know the equation for o2 let's insert into the derivative.

![\frac{\partial o_2}{\partial W_7}=\frac{\partial (W_7*\text{a_h}_1+W_8*\text{a_h}_2)}{\partial W_7}](https://render.githubusercontent.com/render/math?math=\color{red}\Large%20%5Cfrac%7B%5Cpartial%20o_2%7D%7B%5Cpartial%20W_7%7D%3D%5Cfrac%7B%5Cpartial%20%28W_7%2A%5Ctext%7Ba_h%7D_1%2BW_8%2A%5Ctext%7Ba_h%7D_2%29%7D%7B%5Cpartial%20W_7%7D)

![\frac{\partial o_2}{\partial W_7}=\text{a_h}_1 \qquad \cdots(7)](https://render.githubusercontent.com/render/math?math=\color{red}\Large%20%5Cfrac%7B%5Cpartial%20o_2%7D%7B%5Cpartial%20W_7%7D%3D%5Ctext%7Ba_h%7D_1%20%5Cqquad%20%5Ccdots%287%29)

Let's insert 5, 6 and 7 into Loss gradient with respect to W7.

![\frac{\partial\text{ E_Total}}{\partial W_7}=(\text{a_o}_2-t_2)*\text{a_o}_2*(1-\text{a_o}_2)*\text{a_h}_1](https://render.githubusercontent.com/render/math?math=\color{purple}\huge%20%5Cfrac%7B%5Cpartial%5Ctext%7B%20E_Total%7D%7D%7B%5Cpartial%20W_7%7D%3D%28%5Ctext%7Ba_o%7D_2-t_2%29%2A%5Ctext%7Ba_o%7D_2%2A%281-%5Ctext%7Ba_o%7D_2%29%2A%5Ctext%7Ba_h%7D_1)

### **Loss Gradient with respect to W<sub>8</sub>**
Similar to the W7 gradient, the W8 Gradient will be be including only E2 error.

![\frac{\partial\text{ E_Total}}{\partial W_8} = \frac{\partial E_2}{\partial \text{a_o}_2}*\frac{\partial \text{a_o}_2}{\partial o_2}*\frac{\partial o_2}{\partial W_8}](https://render.githubusercontent.com/render/math?math=\color{red}\Large%20%5Cfrac%7B%5Cpartial%5Ctext%7B%20E_Total%7D%7D%7B%5Cpartial%20W_8%7D%20%3D%20%5Cfrac%7B%5Cpartial%20E_2%7D%7B%5Cpartial%20%5Ctext%7Ba_o%7D_2%7D%2A%5Cfrac%7B%5Cpartial%20%5Ctext%7Ba_o%7D_2%7D%7B%5Cpartial%20o_2%7D%2A%5Cfrac%7B%5Cpartial%20o_2%7D%7B%5Cpartial%20W_8%7D)

from the equations 5 and 6.

![\frac{\partial\text{ E_Total}}{\partial W_8}=(\text{a_o}_2-t_2)*\text{a_o}_2*(1-\text{a_o}_2)*\text{a_h}_1](https://render.githubusercontent.com/render/math?math=\color{purple}\huge%20%5Cfrac%7B%5Cpartial%5Ctext%7B%20E_Total%7D%7D%7B%5Cpartial%20W_8%7D%3D%28%5Ctext%7Ba_o%7D_2-t_2%29%2A%5Ctext%7Ba_o%7D_2%2A%281-%5Ctext%7Ba_o%7D_2%29%2A%5Ctext%7Ba_h%7D_2)

## **Gradients of Hidden Layer Weights**
#
Before computing Gradients of W1, W2, W3, and W4, Let's first calculate the Loss gradient with respect to output of hidden layer nuerons(a_h1, a_h2), this will help us to easily find out the Weight Gradients.

![\frac{\partial\text{ E_Total}}{\partial \text{a_h}_1}=\frac{\partial(E_1+E_2)}{\partial \text{a_h}_1}](https://render.githubusercontent.com/render/math?math=\color{red}\Large%20%5Cfrac%7B%5Cpartial%5Ctext%7B%20E_Total%7D%7D%7B%5Cpartial%20%5Ctext%7Ba_h%7D_1%7D%3D%5Cfrac%7B%5Cpartial%28E_1%2BE_2%29%7D%7B%5Cpartial%20%5Ctext%7Ba_h%7D_1%7D)

Let's find out the derivates of E1 and E2 seperately and will combine them later.

![\frac{\partial E_1}{\partial \text{a_h}_1}=\frac{\partial E_1}{\partial \text{a_o}_1}*\frac{\partial \text{a_o}_1}{\partial o_1}*\frac{\partial o_1}{\partial \text{a_h}_1}](https://render.githubusercontent.com/render/math?math=\color{red}\Large%20%5Cfrac%7B%5Cpartial%20E_1%7D%7B%5Cpartial%20%5Ctext%7Ba_h%7D_1%7D%3D%5Cfrac%7B%5Cpartial%20E_1%7D%7B%5Cpartial%20%5Ctext%7Ba_o%7D_1%7D%2A%5Cfrac%7B%5Cpartial%20%5Ctext%7Ba_o%7D_1%7D%7B%5Cpartial%20o_1%7D%2A%5Cfrac%7B%5Cpartial%20o_1%7D%7B%5Cpartial%20%5Ctext%7Ba_h%7D_1%7D)

We already know the first two partial derivatives in the above equation from equation (1) and (2), so let's find out the last derivative and insert them into it.

![\frac{\partial o_1}{\partial \text{a_h}_1}=\frac{\partial(W_5*\text{a_h}_1+W_6*\text{a_h}_2)}{\partial \text{a_h}_1}](https://render.githubusercontent.com/render/math?math=\color{red}\Large%20%5Cfrac%7B%5Cpartial%20o_1%7D%7B%5Cpartial%20%5Ctext%7Ba_h%7D_1%7D%3D%5Cfrac%7B%5Cpartial%28W_5%2A%5Ctext%7Ba_h%7D_1%2BW_6%2A%5Ctext%7Ba_h%7D_2%29%7D%7B%5Cpartial%20%5Ctext%7Ba_h%7D_1%7D)

![\frac{\partial o_1}{\partial \text{a_h}_1}=W_5 \qquad \cdots(8)](https://render.githubusercontent.com/render/math?math=\color{red}\Large%20%5Cfrac%7B%5Cpartial%20o_1%7D%7B%5Cpartial%20%5Ctext%7Ba_h%7D_1%7D%3DW_5%20%5Cqquad%20%5Ccdots%288%29)

By inserting 1, 2 and 8 equations into E1 gradient we get

![\frac{\partial E_1}{\partial \text{a_h}_1}=(\text{a_o}_1-t_1)*\text{a_o}_1*(1-\text{a_o}_1)*W_5 \qquad \cdots(9)](https://render.githubusercontent.com/render/math?math=\color{red}\Large%20%5Cfrac%7B%5Cpartial%20E_1%7D%7B%5Cpartial%20%5Ctext%7Ba_h%7D_1%7D%3D%28%5Ctext%7Ba_o%7D_1-t_1%29%2A%5Ctext%7Ba_o%7D_1%2A%281-%5Ctext%7Ba_o%7D_1%29%2AW_5%20%5Cqquad%20%5Ccdots%289%29)

Now, let us see E2 Gradient.

![\frac{\partial E_2}{\partial \text{a_h}_1}=\frac{\partial E_2}{\partial \text{a_o}_2}*\frac{\partial \text{a_o}_2}{\partial o_2}*\frac{\partial o_2}{\partial \text{a_h}_1}](https://render.githubusercontent.com/render/math?math=\color{red}\Large%20%5Cfrac%7B%5Cpartial%20E_2%7D%7B%5Cpartial%20%5Ctext%7Ba_h%7D_1%7D%3D%5Cfrac%7B%5Cpartial%20E_2%7D%7B%5Cpartial%20%5Ctext%7Ba_o%7D_2%7D%2A%5Cfrac%7B%5Cpartial%20%5Ctext%7Ba_o%7D_2%7D%7B%5Cpartial%20o_2%7D%2A%5Cfrac%7B%5Cpartial%20o_2%7D%7B%5Cpartial%20%5Ctext%7Ba_h%7D_1%7D)

We already know the first two partial derivatives in the above equation from equation (5) and (6), so let's find out the last derivative and insert them into it.

![\frac{\partial o_2}{\partial \text{a_h}_1}=\frac{\partial(W_7*\text{a_h}_1+W_8*\text{a_h}_2)}{\partial \text{a_h}_1}](https://render.githubusercontent.com/render/math?math=\color{red}\Large%20%5Cfrac%7B%5Cpartial%20o_2%7D%7B%5Cpartial%20%5Ctext%7Ba_h%7D_1%7D%3D%5Cfrac%7B%5Cpartial%28W_7%2A%5Ctext%7Ba_h%7D_1%2BW_8%2A%5Ctext%7Ba_h%7D_2%29%7D%7B%5Cpartial%20%5Ctext%7Ba_h%7D_1%7D)

![\frac{\partial o_2}{\partial \text{a_h}_1}=W_7 \qquad \cdots(10)](https://render.githubusercontent.com/render/math?math=\color{red}\Large%20%5Cfrac%7B%5Cpartial%20o_2%7D%7B%5Cpartial%20%5Ctext%7Ba_h%7D_1%7D%3DW_7%20%5Cqquad%20%5Ccdots%2810%29)

By inserting 5, 6 and 10 equations into E2 gradient we get

![\frac{\partial E_2}{\partial \text{a_h}_1}=(\text{a_o}_2-t_2)*\text{a_o}_2*(1-\text{a_o}_2)* W_7 \qquad \cdots(11)](https://render.githubusercontent.com/render/math?math=\color{red}\Large%20%5Cfrac%7B%5Cpartial%20E_2%7D%7B%5Cpartial%20%5Ctext%7Ba_h%7D_1%7D%3D%28%5Ctext%7Ba_o%7D_2-t_2%29%2A%5Ctext%7Ba_o%7D_2%2A%281-%5Ctext%7Ba_o%7D_2%29%2A%20W_7%20%5Cqquad%20%5Ccdots%2811%29)

And finally adding both equation (9) and (11) we get the total loss with respect to a_h1.

![\frac{\partial\text{ E_Total}}{\partial \text{a_h}_1}=(\text{a_o}_1-t_1)*\text{a_o}_1*(1-\text{a_o}_1)*W_5+\\(\text{a_o}_2-t_2)*\text{a_o}_2*(1-\text{a_o}_2)* W_7 \qquad \cdots(12)](https://render.githubusercontent.com/render/math?math=\color{red}\Large%20%5Cfrac%7B%5Cpartial%5Ctext%7B%20E_Total%7D%7D%7B%5Cpartial%20%5Ctext%7Ba_h%7D_1%7D%3D%28%5Ctext%7Ba_o%7D_1-t_1%29%2A%5Ctext%7Ba_o%7D_1%2A%281-%5Ctext%7Ba_o%7D_1%29%2AW_5%2B%5C%5C%28%5Ctext%7Ba_o%7D_2-t_2%29%2A%5Ctext%7Ba_o%7D_2%2A%281-%5Ctext%7Ba_o%7D_2%29%2A%20W_7%20%5Cqquad%20%5Ccdots%2812%29)

Similarly when we apply the chain rule to Total erros gradient with respect to a_h2, we get

![\frac{\partial\text{ E_Total}}{\partial \text{a_h}_2}=(\text{a_o}_1-t_1)*\text{a_o}_1*(1-\text{a_o}_1)*W_6+\\(\text{a_o}_2-t_2)*\text{a_o}_2*(1-\text{a_o}_2)* W_8 \qquad \cdots(13)](https://render.githubusercontent.com/render/math?math=\color{red}\Large%20%5Cfrac%7B%5Cpartial%5Ctext%7B%20E_Total%7D%7D%7B%5Cpartial%20%5Ctext%7Ba_h%7D_2%7D%3D%28%5Ctext%7Ba_o%7D_1-t_1%29%2A%5Ctext%7Ba_o%7D_1%2A%281-%5Ctext%7Ba_o%7D_1%29%2AW_6%2B%5C%5C%28%5Ctext%7Ba_o%7D_2-t_2%29%2A%5Ctext%7Ba_o%7D_2%2A%281-%5Ctext%7Ba_o%7D_2%29%2A%20W_8%20%5Cqquad%20%5Ccdots%2813%29)

Now let's derive the Loss gradient with respect to W1, W2, W3 and W4.

Expanding the Partial Derivative with respect to W1 using chain rule will give us the following equation.

![\frac{\partial\text{ E_Total}}{\partial W_1}=\frac{\partial\text{ E_Total}}{\partial \text{a_h}_1}*\frac{{\partial \text{a_h}_1}}{\partial h_1}*\frac{\partial h_1}{\partial W_1}](https://render.githubusercontent.com/render/math?math=\color{red}\Large%20%5Cfrac%7B%5Cpartial%5Ctext%7B%20E_Total%7D%7D%7B%5Cpartial%20W_1%7D%3D%5Cfrac%7B%5Cpartial%5Ctext%7B%20E_Total%7D%7D%7B%5Cpartial%20%5Ctext%7Ba_h%7D_1%7D%2A%5Cfrac%7B%7B%5Cpartial%20%5Ctext%7Ba_h%7D_1%7D%7D%7B%5Cpartial%20h_1%7D%2A%5Cfrac%7B%5Cpartial%20h_1%7D%7B%5Cpartial%20W_1%7D)

Equation (12) gives us the value for firs partial derivative, and coming to 2nd derivative as we have already seen earlier a_h1 is just a sigmoid of h1, so the derivative will be similar to equation (2) instead of a_o1 here we will have a_h1.

And coming to the h1 derivative, once we we insert the value of h1 and find out the partial derivative we get i1 as result.

And finally including all these results into above equation we get

![\frac{\partial\text{ E_Total}}{\partial W_1}=((\text{a_o}_1-t_1)*\text{a_o}_1*(1-\text{a_o}_1)*W_5+](https://render.githubusercontent.com/render/math?math=\color{purple}\huge%20%5Cfrac%7B%5Cpartial%5Ctext%7B%20E_Total%7D%7D%7B%5Cpartial%20W_1%7D%3D%28%28%5Ctext%7Ba_o%7D_1-t_1%29%2A%5Ctext%7Ba_o%7D_1%2A%281-%5Ctext%7Ba_o%7D_1%29%2AW_5%2B)
![\\(\text{a_o}_2-t_2)*\text{a_o}_2*(1-\text{a_o}_2)* W_7)*
\text{a_h}_1*(1-\text{a_h}_1)*i_1](https://render.githubusercontent.com/render/math?math=\color{purple}\huge%20%5C%5C%28%5Ctext%7Ba_o%7D_2-t_2%29%2A%5Ctext%7Ba_o%7D_2%2A%281-%5Ctext%7Ba_o%7D_2%29%2A%20W_7%29%2A%0A%5Ctext%7Ba_h%7D_1%2A%281-%5Ctext%7Ba_h%7D_1%29%2Ai_1)

Similarly the Loss gradient with respect to W2 will be

![\frac{\partial\text{ E_Total}}{\partial W_2}=((\text{a_o}_1-t_1)*\text{a_o}_1*(1-\text{a_o}_1)*W_5+](https://render.githubusercontent.com/render/math?math=\color{purple}\huge%20%5Cfrac%7B%5Cpartial%5Ctext%7B%20E_Total%7D%7D%7B%5Cpartial%20W_2%7D%3D%28%28%5Ctext%7Ba_o%7D_1-t_1%29%2A%5Ctext%7Ba_o%7D_1%2A%281-%5Ctext%7Ba_o%7D_1%29%2AW_5%2B)
![\\(\text{a_o}_2-t_2)*\text{a_o}_2*(1-\text{a_o}_2)* W_7)*
\text{a_h}_1*(1-\text{a_h}_1)*i_2](https://render.githubusercontent.com/render/math?math=\color{purple}\huge%20%5C%5C%28%5Ctext%7Ba_o%7D_2-t_2%29%2A%5Ctext%7Ba_o%7D_2%2A%281-%5Ctext%7Ba_o%7D_2%29%2A%20W_7%29%2A%0A%5Ctext%7Ba_h%7D_1%2A%281-%5Ctext%7Ba_h%7D_1%29%2Ai_2)

Now, lets derive for the 2nd nueron in Hidden layer.

![\frac{\partial\text{ E_Total}}{\partial W_3}=\frac{\partial\text{ E_Total}}{\partial \text{a_h}_2}*\frac{{\partial \text{a_h}_2}}{\partial h_2}*\frac{\partial h_2}{\partial W_3}](https://render.githubusercontent.com/render/math?math=\color{red}\Large%20%5Cfrac%7B%5Cpartial%5Ctext%7B%20E_Total%7D%7D%7B%5Cpartial%20W_3%7D%3D%5Cfrac%7B%5Cpartial%5Ctext%7B%20E_Total%7D%7D%7B%5Cpartial%20%5Ctext%7Ba_h%7D_2%7D%2A%5Cfrac%7B%7B%5Cpartial%20%5Ctext%7Ba_h%7D_2%7D%7D%7B%5Cpartial%20h_2%7D%2A%5Cfrac%7B%5Cpartial%20h_2%7D%7B%5Cpartial%20W_3%7D)

As we can see, we can apply samilar logic of W1 gradient on the above equation. And The gradient of E total with respect to a_h2 is already derived at equation (13).

![\frac{\partial\text{ E_Total}}{\partial W_3}=((\text{a_o}_1-t_1)*\text{a_o}_1*(1-\text{a_o}_1)*W_6+](https://render.githubusercontent.com/render/math?math=\color{purple}\huge%20%5Cfrac%7B%5Cpartial%5Ctext%7B%20E_Total%7D%7D%7B%5Cpartial%20W_3%7D%3D%28%28%5Ctext%7Ba_o%7D_1-t_1%29%2A%5Ctext%7Ba_o%7D_1%2A%281-%5Ctext%7Ba_o%7D_1%29%2AW_6%2B)
![\\(\text{a_o}_2-t_2)*\text{a_o}_2*(1-\text{a_o}_2)* W_8)*
\text{a_h}_1*(1-\text{a_h}_1)*i_1](https://render.githubusercontent.com/render/math?math=\color{purple}\huge%20%5C%5C%28%5Ctext%7Ba_o%7D_2-t_2%29%2A%5Ctext%7Ba_o%7D_2%2A%281-%5Ctext%7Ba_o%7D_2%29%2A%20W_8%29%2A%0A%5Ctext%7Ba_h%7D_1%2A%281-%5Ctext%7Ba_h%7D_1%29%2Ai_1)

And on the same note the Gradient with respect to W4 becomes

![\frac{\partial\text{ E_Total}}{\partial W_4}=((\text{a_o}_1-t_1)*\text{a_o}_1*(1-\text{a_o}_1)*W_6+](https://render.githubusercontent.com/render/math?math=\color{purple}\huge%20%5Cfrac%7B%5Cpartial%5Ctext%7B%20E_Total%7D%7D%7B%5Cpartial%20W_4%7D%3D%28%28%5Ctext%7Ba_o%7D_1-t_1%29%2A%5Ctext%7Ba_o%7D_1%2A%281-%5Ctext%7Ba_o%7D_1%29%2AW_6%2B)
![\\(\text{a_o}_2-t_2)*\text{a_o}_2*(1-\text{a_o}_2)* W_8)*
\text{a_h}_1*(1-\text{a_h}_1)*i_2](https://render.githubusercontent.com/render/math?math=\color{purple}\huge%20%5C%5C%28%5Ctext%7Ba_o%7D_2-t_2%29%2A%5Ctext%7Ba_o%7D_2%2A%281-%5Ctext%7Ba_o%7D_2%29%2A%20W_8%29%2A%0A%5Ctext%7Ba_h%7D_1%2A%281-%5Ctext%7Ba_h%7D_1%29%2Ai_2)
