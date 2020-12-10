import torch
from spotlight.factorization.explicit import ExplicitFactorizationModel
from spotlight.cross_validation import random_train_test_split
from spotlight.evaluation import rmse_score

def build_model(data, loss, embedding_dim, n_iter, batch_size, l2, learning_rate, **kwargs):
    model = ExplicitFactorizationModel(loss=loss,
                                   embedding_dim=embedding_dim,  # latent dimensionality
                                   n_iter=n_iter,  # number of epochs of training
                                   batch_size=batch_size,  # minibatch size
                                   l2=l2,  # strength of L2 regularization
                                   learning_rate=learning_rate,
                                   use_cuda=torch.cuda.is_available())
    
    train, test = random_train_test_split(data, random_state=np.random.RandomState(42))
    model.fit(train, verbose=True)
    test_rmse = rmse_score(model, test)
    return test_rsme