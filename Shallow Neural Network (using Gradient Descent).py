#!/usr/bin/env python
# coding: utf-8

# In[13]:


import tensorflow as tf
import numpy as np


# In[14]:


EPOCHS = 1000


# In[15]:


class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.d1 = tf.keras.layers.Dense(128, input_dim=2, activation='sigmoid')
        self.d2 = tf.keras.layers.Dense(10, activation='softmax')
            
    def call(self, x, training=None, mask=None):
        x = self.d1(x)
        return self.d2(x)


# In[16]:


@tf.function
def train_step(model, inputs, labels, loss_object, optimizer, train_loss, train_metric):
    with tf.GradientTape() as tape:
        predictions = model(inputs)
        loss = loss_object(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables) # df(x)/dx
    
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    train_loss(loss)
    train_metric(labels, predictions)


# In[17]:


np.random.seed(0)

pts = list()
labels = list()
center_pts = np.random.uniform(-8.0, 8.0, (10, 2))
for label, center_pt in enumerate(center_pts):
    for _ in range(100):
        pts.append(center_pt + np.random.randn(*center_pt.shape))
        labels.append(label)

pts = np.stack(pts, axis=0).astype(np.float32)
labels = np.stack(labels, axis=0)

train_ds = tf.data.Dataset.from_tensor_slices((pts, labels)).shuffle(1000).batch(32)


# In[18]:


model = MyModel()


# ### CrossEntropy, Adam Optimizer

# In[19]:


loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam()


# ### Accuracy

# In[20]:


train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')


# In[22]:


for epoch in range(EPOCHS):
    for x, label in train_ds:
        train_step(model, x, label, loss_object, optimizer, train_loss, train_accuracy)
        
    template = 'Epoch {}, Loss: {}, Accuracy: {}'
    print(template.format(epoch + 1,
                          train_loss.result(),
                          train_accuracy.result() * 100))
    train_loss.reset_states()
    train_accuracy.reset_states()


# In[23]:


np.savez_compressed('ch2_dataset.npz', inputs=pts, labels=labels)

W_h, b_h = model.d1.get_weights()
W_o, b_o = model.d2.get_weights()
W_h = np.transpose(W_h)
W_o = np.transpose(W_o)
np.savez_compressed('ch2_parameters.npz',
                    W_h=W_h,
                    b_h=b_h,
                    W_o=W_o,
                    b_o=b_o)


# In[ ]:




