from ini import*

import numpy as np

def next_batch(num, data):
    '''
    Return a total of `num` random samples and labels. 
    '''
    idx = np.arange(0 , len(data))
    np.random.shuffle(idx)
    idx = idx[:num]
    data_shuffle = [data[ i] for i in idx]
    

    return np.asarray(data_shuffle)

def train(network_architecture, learning_rate=0.001,
          batch_size=100, training_epochs=10, display_step=5):
    vae = VA(network_architecture, 
                                 learning_rate=learning_rate, 
                                 batch_size=batch_size)
    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = (n_samples) / batch_size
        # Loop over all batches
        for i in range(total_batch):
            batch_xs= next_batch(batch_size, x_train)

            # Fit training using batch data
            cost = vae.partial_fit(batch_xs)
            # Compute average loss
            avg_cost += cost / n_samples * batch_size

        # Display logs per epoch step
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1), "cost=", avg_cost)
    return vae


network_architecture = \
    dict(n_hidden_recog_1=500, # 1st layer encoder neurons
         n_hidden_recog_2=500, # 2nd layer encoder neurons
         n_hidden_gener_1=500, # 1st layer decoder neurons
         n_hidden_gener_2=500, # 2nd layer decoder neurons
         n_input=784, # MNIST data input (img shape: 28*28)
         n_z=32)  # dimensionality of latent space

vae_2d = train(network_architecture, training_epochs=50)


x_sample = next_batch(100, x_train)
z_mu = vae_2d.transform(x_sample)
np.shape(z_mu)



z=[]
for q in range (i):
    s=(np.heaviside(z_mu[q],0))
    z.append(s.astype(int))

m= np.array(z)

np.savetxt('binary.txt', m, fmt='%d')