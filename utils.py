import numpy as np

#Divise l'image en patchs pour la convolution
def patches_generator(image, kernel_size):
    image_h = image.shape[0] - kernel_size + 1
    image_w = image.shape[1] - kernel_size + 1

    # Itère chaque pixel excluant les derniers pixels en fonction de la taille du kernel pour éviter débordement
    for h in range(image_h):
        for w in range(image_w):
            # Slice l'image de h à h+kernel_size et de w à w+kernel_size pour obtenir un patch de la même taille que le kernel
            patch = slice(image, h, w, kernel_size)
            yield patch, h, w

def slice(image, h, w, size):
    return image[h:h + size, w:w + size, :]

# First layer
class ConvolutionLayer:
    def __init__(self, kernel_num, kernel_size):
        self.kernel_num = kernel_num
        self.kernel_size = kernel_size
        self.kernels = None 
        
    def initialize_kernels(self, num_input_channels):
        self.kernels = np.random.randn(self.kernel_num, self.kernel_size, self.kernel_size, num_input_channels) / (self.kernel_size * self.kernel_size)

    def forward_prop(self, image):
        # Si 2D, ajouter une nouvelle dimension pour les channels (ex: (H, W) -> (H, W, 1))
        if image.ndim == 2: 
            image = image[:, :, np.newaxis] 
        self.image = image
        image_h, image_w, num_input_channels = image.shape
        
        # Initialise les kernels selon les nombres de channels, 1 pour premier layer
        if self.kernels is None:
            self.initialize_kernels(num_input_channels)
        
        conv_out = np.zeros((image_h - self.kernel_size + 1, image_w - self.kernel_size + 1, self.kernel_num))
        
        #Fait un dot entre le patch de l'image d'entrée et les kernels pour chaque position (h, w) et chaque kernel
        for patch, h, w in patches_generator(image, self.kernel_size):
            conv_out[h, w, :] = np.tensordot(self.kernels, patch, axes=([1,2,3],[0,1,2]))
        
        return conv_out

    def back_prop(self, dE_dY, alpha):
        # dE_dY: (H_out, W_out, kernel_num)
        # Gradient du loss par rapport aux kernels
        dE_dk = np.zeros_like(self.kernels) # (kernel_num, kernel_size, kernel_size, channels)

        # Gradient du loss par rapport à l'image d'entrée
        dE_dX = np.zeros_like(self.image)    # (H, W, channels)

        for patch, h, w in patches_generator(self.image, self.kernel_size):
            # Gradient pour chaque filtre
            dE_dY_filter = dE_dY[h, w, :]

            # Ajuste les poids des kernels avec le gradient du patch de l'image d'entrée 
            dE_dk += np.tensordot(dE_dY_filter, patch, axes=0)

            # Ajuste le gradient de l'image d'entrée à la position (h, w) avec un dot entre le gradient du filtre et le kernel
            dE_dX[h:h+self.kernel_size, w:w+self.kernel_size, :] += np.tensordot(dE_dY_filter, self.kernels, axes=([0],[0]))

        self.kernels -= alpha * dE_dk
        return dE_dX

class MaxPoolingLayer:
    def __init__(self, kernel_size):
        self.kernel_size = kernel_size

    def forward_prop(self, image):
        # Si 2D, ajouter une nouvelle dimension pour les channels (ex: (H, W) -> (H, W, 1))
        if image.ndim == 2:
            image = image[:, :, np.newaxis]
        self.image = image
        image_h, image_w, num_kernels = image.shape
        h_out = image_h // self.kernel_size
        w_out = image_w // self.kernel_size

        # Trim l'image pour que ses dimensions soient divisibles par kernel_size
        image_trimmed = image[:h_out*self.kernel_size, :w_out*self.kernel_size, :]
        # Reshape image en patchs de pooling
        image_reshaped = image_trimmed.reshape(h_out, self.kernel_size, w_out, self.kernel_size, num_kernels)
        # Maximise selon les dimensions du kernel
        max_pooling_output = image_reshaped.max(axis=(1, 3)) 

        return max_pooling_output

    def back_prop(self, dE_dY, alpha=None):
        H, W, C = self.image.shape
        h_out = H // self.kernel_size
        w_out = W // self.kernel_size

        # Comme dans forward_prop
        image_trimmed = self.image[:h_out*self.kernel_size, :w_out*self.kernel_size, :]
        image_reshaped = image_trimmed.reshape(h_out, self.kernel_size, w_out, self.kernel_size, C)
        max_val = image_reshaped.max(axis=(1, 3), keepdims=True)

        # Crée un masque pour les positions des max dans chaque patch
        mask = (image_reshaped == max_val)

        # Reshape dE_dY pour correspondre à la forme du patch de pooling
        dE_dY_expanded = dE_dY[:, np.newaxis, :, np.newaxis, :]

        # Applique le gradient aux max positions
        dE_dX_reshaped = mask * dE_dY_expanded

        # Reshape back to original image shape
        dE_dX = dE_dX_reshaped.reshape(h_out*self.kernel_size, w_out*self.kernel_size, C)

        # Pad pour les dimensions qui ont été trimées
        dE_dX_full = np.zeros_like(self.image)
        dE_dX_full[:h_out*self.kernel_size, :w_out*self.kernel_size, :] = dE_dX

        return dE_dX_full
    
class ReLULayer:
    def forward_prop(self, image):
        self.image = image
        return np.maximum(0, image)

    def back_prop(self, gradient):
        grad_input = gradient * (self.image > 0)
        return grad_input

class SoftmaxLayer:
    def __init__(self, input_units, output_units):
        self.weight = np.random.randn(input_units, output_units) / input_units
        self.bias = np.zeros(output_units)

    def forward_prop(self, image):
        self.original_shape = image.shape

        image_flattened = image.flatten()
        self.flattened_input = image_flattened

        first_output = np.dot(image_flattened, self.weight) + self.bias
        self.output = first_output

        # Empeche une explosion
        shifted = first_output - np.max(first_output)
        exp_vals = np.exp(shifted)

        #Activation
        softmax_output = exp_vals / np.sum(exp_vals)
        return softmax_output

    def back_prop(self, dE_dY, alpha):
        # dE_dY: gradient du loss par rapport au output

        #Empeche une explosion
        shifted = self.output - np.max(self.output)
        transformation_eq = np.exp(shifted) # Numerateur de la softmax
        S_total = np.sum(transformation_eq) # Dénominateur de la softmax

        # Multiplication Jacobian
        exp_i = transformation_eq[:, np.newaxis] # (n, 1)
        exp_j = transformation_eq[np.newaxis, :] # (1, n)
        jacobian = -exp_i @ exp_j / (S_total ** 2)
        # Remplir diagonal
        diag_entries = transformation_eq * (S_total - transformation_eq) / (S_total ** 2)
        np.fill_diagonal(jacobian, diag_entries)

        # Gradient du loss par rapport au input du softmax
        dE_dZ = dE_dY @ jacobian
        # Gradient du loss par rapport aux poids
        dE_dw = self.flattened_input[np.newaxis].T @ dE_dZ[np.newaxis]
        # Gradient du loss par rapport au bias
        dE_db = dE_dZ
        # Gradient du loss par rapport à l'image d'entrée
        dE_dX = self.weight @ dE_dZ

        self.weight -= alpha * dE_dw
        self.bias -= alpha * dE_db

        return dE_dX.reshape(self.original_shape)

def CNN_forward(image, label, layers):
    output = image/255.
    for layer in layers:
        #print("Layer {}: {}".format(layers.index(layer), output.shape))
        output = layer.forward_prop(output)
    # Compute loss (cross-entropy) and accuracy
    loss = -np.log(output[label]+1e-15)
    accuracy = 1 if np.argmax(output) == label else 0
    return output, loss, accuracy

def CNN_backprop(gradient, layers, alpha=0.05):
    grad_back = gradient
    for layer in layers[::-1]:
        if type(layer) in [ConvolutionLayer, SoftmaxLayer]:
            grad_back = layer.back_prop(grad_back, alpha)
        elif type(layer) in [MaxPoolingLayer, ReLULayer]:
            grad_back = layer.back_prop(grad_back)
    return grad_back


def CNN_training(image, label, layers, alpha=0.05):
    # Forward step
    output, loss, accuracy = CNN_forward(image, label, layers)
    # Initial gradient
    gradient = np.zeros(16)
    #print("Output of the network: {}".format(output))
    gradient[label] = -1/(output[label]+1e-15)

    # Backprop step
    gradient_back = CNN_backprop(gradient, layers, alpha)

    return output, loss, accuracy




"""
import pandas as pd
import matplotlib.pyplot as plt
#Test the convolutions with 1 image, to put in the article
# Test
df_train = pd.read_csv('train.csv')
img = df_train.iloc[40,:].values[1:]
img = np.reshape(img,(28,28))
plt.imshow(img, cmap='gray')
plt.show()
print(img.shape)
plt.savefig('images/original_image.png', format='png', dpi=1200)

# Test with a convolution of 16 filters of size 3x3
my_conv = ConvolutionLayer(32,3)
output = my_conv.forward_prop(img)
# See the dimensions of the output volume, they follow the usual formula
print(output.shape)

# Plot 16th volume after the convolution
plt.imshow(output[:,:,15], cmap='gray')
plt.show()
plt.savefig('images/image_convolved.png', format='png', dpi=1200)
"""
