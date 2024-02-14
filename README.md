# Universal Differential Equations for SIR regional transmission: repository

This repository collects the code files for the project "Learning COVID-19 Regional Transmission Using Universal Differential Equations in a SIR Model" published as preprint in ArXiv. Access the papper [here](https://arxiv.org/abs/2310.16804).

### Abstract

Highly-interconnected societies difficult to model the spread of infectious diseases such as COVID-19. Single-region SIR models fail to account for incoming forces of infection and expanding them to a large number of interacting regions involves many assumptions that do not hold in the real world. We propose using Universal Differential Equations (UDEs) to capture the influence of neighboring regions and improve the model's predictions in a combined SIR+UDE model. UDEs are differential equations totally or partially defined by a deep neural network (DNN). We include an additive term to the SIR equations composed by a DNN that learns the incoming force of infection from the other regions. The learning is performed using automatic differentiation and gradient descent to approach the change in the target system caused by the state of the neighboring regions. We compared the proposed model using a simulated COVID-19 outbreak against a single-region SIR and a fully data-driven model composed only of a DNN. The proposed UDE+SIR model generates predictions that capture the outbreak dynamic more accurately, but a decay in performance is observed at the last stages of the outbreak. The single-area SIR and the fully data-driven approach do not capture the proper dynamics accurately. Once the predictions were obtained, we employed the SINDy algorithm to substitute the DNN with a regression, removing the black box element of the model with no considerable increase in the error levels.


# Files

Here we upload the files required to reproduce the results showed on the paper.

## SIR model (SIR.jl)

As baseline, we fit a single SIR model's infection rate $\beta$ amd recovery rate $\gamma$, excluding information from neighboring regions using Gradient Descent. Both parameters were initalizated randomly. A single SIR model is not expected to generate correct predictions for each location, given the omission of the incomming force of infection. 

The SIR.jl file presents the implementation of the SIR model:

 ```math
\begin{equation}
\begin{aligned}
\dot{S}_{\text{target}} &= -\beta S I  \\
\dot{I}_{\text{target}} &= \beta S I - \gamma I\\
\dot{R}_{\text{target}} &= \gamma I \\
\end{aligned}
\end{equation}
```

## Full UDE model (UDE.jl)

Additionally, an universal differential equation model was trained composed exclusively of a DNN. The DNN was trained to predict the changes in the target region, tacking as input information from the target to the neighboring regions. 

The UDE.jl file contains the implementation of the model summarized as follows, with $u$ as the state of the SIR system at a specific time $t$:

 ```math
\begin{equation}
\begin{aligned}
\dot{S}_{\text{target}} &= \text{DNN}_{\mu}(u_{\text{target}};u_{\text{neighbors}})[1] \\
\dot{I}_{\text{target}} &= \text{DNN}_{\mu}(u_{\text{target}};u_{\text{neighbors}})[2] \\
\dot{R}_{\text{target}} &= \text{DNN}_{\mu}(u_{\text{target}};u_{\text{neighbors}})[3] \\
\end{aligned}
\end{equation}
 ```

## SIR+UDE model (SIR+UDE.jl)

The SIR+UDE model combines theSIR model with an extra additive term estimated by a DNN, in charge of capturing the incomming force of infection from the neighboring regions. This influence is added to the infected compartment of the target region and substracted from the susceptibles. 

The SIR+UDE.jl shows the implementation of the model: 

 ```math
\begin{equation}
\begin{aligned}
\dot{S}_{\text{target}} &= -\beta S I - \text{DNN}_{\theta}(u_{\text{neighbors}})\\
\dot{I}_{\text{target}} &= \beta SI + \text{DNN}_{\theta}(u_{\text{neighbors}})- \gamma I  \\
\dot{R}_{\text{target}} &= \gamma I \\
\end{aligned}
\end{equation}
 ```

## SINDY.ipynb

We used the SINDy algorithm to perform a post hoc analysis of the function learned by the DNN, intending to discover an algebraic form dor such a function and substitute the DNN element of the SIR+UDE model. The SINDY.ipynb loads the trained SIR+UDE models with the lowest error and applies the SINDY algorithm to remove the DNN from the model. 

## Results.ipynb

The Results notebook reads and plots the results of the different models. It was used to generate the figures for the publication.
