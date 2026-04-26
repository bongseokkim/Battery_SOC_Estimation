
# MATLAB AI Challenge 2023 – Battery SoC Estimation

This repository contains the official source code for our project, **“Electric Car Battery State of Charge (SoC) Estimation Using an Ensemble Method,”** which won **1st place** in the **MATLAB AI Challenge 2023**.

The project develops an ensemble machine learning approach to estimate the State of Charge (SoC) of electric vehicle batteries using battery operation data.

## Project Information

- **Competition:** MATLAB AI Challenge 2023
- **Award:** 1st Place Winner
- **Project Title:** Electric Car Battery State of Charge (SoC) Estimation Using an Ensemble Method
- **Presenters / Developers:** Jaemin Park, Bongseok Kim


## Description

Accurate battery State of Charge estimation is important for electric vehicle battery management systems. This project applies ensemble-based machine learning methods to improve SoC estimation accuracy and robustness

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
