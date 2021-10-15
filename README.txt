Install Keras v2.0, scikit-learn
sudo pip install keras scikit-learn

# settings in main.py

TEST = Ture
# when TEST = Ture, the code just test the trained DEMVC model

train_ae = False
# when train_ae = Ture, the code will pre-train the autoencoders first, and the fine-turn the model with DEMVC

data = 'MNIST_USPS_COMIC'     
# the tested datasets contain:
# 'MNIST_USPS_COMIC'        (CAE)
# 'BDGP'                                 (FAE)

# run the code：
python main.py