# Seminar_DDoSForecast
## Introduction
This repository is a collection of code that I use for the "seminar" course in the Computer Science Department of Gadjah Mada University. The topic of this seminar is "Deep Learning Method for Prediction of DDoS Attacks on Social Media" which is taken from [DOI:10.1142/S2424922X19500025](http://dx.doi.org/10.1142/S2424922X19500025)

## Configuration/Setup
#### Virtual Environment : Conda
How to create virtual environment: `conda env create --name <env name> --file environment.yml`
#### Tweet Module (twint)
Because there's some problem with `twint` module in pip3's default source. We need to install twint manualy from their repository.<br/>
command: `pip3 install --user --upgrade git+https://github.com/twintproject/twint.git@origin/master#egg=twint`

## How to use
#### Train model
command: `python3 train.py`
#### Run prediction
command: `python3 train.py`
input: `<tweet keyword>`

## Screenshot
![SS_pred1](https://user-images.githubusercontent.com/33412865/119638238-aa57d780-be40-11eb-825b-9f315b64258b.png)
![red](https://user-images.githubusercontent.com/33412865/119638260-af1c8b80-be40-11eb-81b5-78127fbb68d4.png)
