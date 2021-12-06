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
# 'BDGP'                    (FAE)

# run the code：
python main.py

@article{xu2021deep,
  title={Deep embedded multi-view clustering with collaborative training},
  author={Xu, Jie and Ren, Yazhou and Li, Guofeng and Pan, Lili and Zhu, Ce and Xu, Zenglin},
  journal={Information Sciences},
  volume={573},
  pages={279--290},
  year={2021}
}
