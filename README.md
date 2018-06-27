# Variational AutoEncoder
## Setup

```
$ virtualenv -p python3 VAE
$ cd VAE
$ source bin/activate
$ pip3 install -r requirements.txt
```

## training
### normal autoencoder
```
$ python3 autoencoder.py train
```

if you want to change model

```
$ python3 autoencoder.py train --params_path='***.yml'
```

### convolutional autoencoder
```
$ python3 conv_ae.py train
```

if you want to change model

```
$ python3 conv_ae.py train --params_path='***.yml'
```

### variational autoencoder
```
$ python3 vae.py train
```

if you want to change model

```
$ python3 vae.py train --vae_params_path='***.yml'
```

## predict
```
$ python3 autoencoder.py(conv_ae.py/vae.py) predict --model_dir='MODEL_NAME' --fig_name='FIG_NAME'
```
