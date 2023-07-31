## Battery_SOC_Estimation
SOC (State of Charge) estimation for a battery using an ensemble approach with Coulomb counting and pre-trained LSTM prediction

An accurate estimation of battery’s State of Charge (SoC) is a prerequisite prior to 
devising battery management and control systems. Traditional techniques including Coulomb 
counting and open circuit voltage (OCV) methods still need improvements due to battery’s 
innate issues such as non-linearity, temperature dependence, and aging effects. We 
introduce a machine learning-based approach for estimating the SoC of a battery using voltage, 
current, and temperature data. We utilized battery data from four Tesla Model 3 battery packs 
with varying temperature and discharge cycle environments. This dataset encompasses the 
battery's SoC, voltage, current, and temperature measurements over time. 

Our findings demonstrated that our model could estimate the battery's SoC with an RMSE 
of less than 2%. The proposed methodology overcomes challenges inherent in conventional 
estimation techniques and offers the potential for application across diverse battery 
technologies while ensuring the explainability of the model's predictions

## Files 
+ Battery_Data.mat: Battery data for validation purposes.
+ trained_lstm.mat: Pre-trained LSTM network's weights.
+ Model.m: SOC estimator, ensemble approach with Coulomb counting and pre-trained LSTM prediction.

## reference
The data and parts of the example code were based on the following reference material.
```
@inproceedings{kollmeyer2022blind,
  title={A blind modeling tool for standardized evaluation of battery state of charge estimation algorithms},
  author={Kollmeyer, Phillip J and Naguib, Mina and Khanum, Fauzia and Emadi, Ali},
  booktitle={2022 IEEE Transportation Electrification Conference \& Expo (ITEC)},
  pages={243--248},
  year={2022},
  organization={IEEE}
}
```
