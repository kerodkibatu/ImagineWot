# Imagine Wot

**Imagine Wot** is a toy experimental Generative AI model designed to generate traditional Ethiopian dishes with the power of a Convolutional Autoencoder to imagine new dishes from existing ones with a similar style.

![Project Banner](https://www.mathworks.com/discovery/autoencoder/_jcr_content/mainParsys/image.adapt.1200.medium.svg/1718184511831.svg)

## Short Description

Imagine Wot is an experimental toy model that taps into the creative potential of AI to generate traditional Ethiopian recipes. By leveraging a Convolutional Autoencoder, the model learns patterns and structures within existing Ethiopian recipes, allowing it to generate new, unique, and culturally authentic recipes from the latent space.

## Solution Details

The core of Imagine Wot is a **Convolutional Autoencoder**, a neural network architecture commonly used for unsupervised learning. This model was chosen because of its ability to effectively compress high-dimensional data, such as the intricate details of recipes, into a lower-dimensional latent space. Here’s a breakdown of the solution:

1. **Data Preparation**: The model is trained on a curated dataset of traditional Ethiopian recipes, including ingredients, cooking methods, and cultural context. The data is preprocessed and structured for training the Convolutional Autoencoder.

2. **Model Architecture**: The Convolutional Autoencoder consists of an encoder that compresses the recipe data into a latent space representation and a decoder that reconstructs the recipe from this compressed form. The latent space captures the underlying patterns and variations in the data.

3. **Training Process**: The model is trained by minimizing the reconstruction error between the original recipes and the generated ones. As the training progresses, the model learns to capture the essence of Ethiopian cuisine in its latent space.

4. **Recipe Generation**: Once trained, the model can generate new recipes by sampling from the latent space. These generated recipes are unique, and diverse, yet rooted in the traditional Ethiopian culinary style.

## Core Ideas and Concepts

### Latent Space Exploration

Latent space refers to the compressed, abstract representation of data learned by the encoder part of the Autoencoder. In the context of Imagine Wot, this space holds the key to understanding the variations and structures of Ethiopian recipes. By exploring this space, we can generate entirely new recipes that still retain the cultural and culinary essence of Ethiopian cuisine.

![Latent Space Visualization](https://i.sstatic.net/bAMl5.png)

#### Example Latent Space of an AutoEncoder trained on the MNIST dataset
![Latent Space Visualization](https://miro.medium.com/v2/resize:fit:1400/1*pkbn-_3Nwibt1ufXjhiBkQ.png)

### Convolutional Autoencoder

The Convolutional Autoencoder is central to the functionality of Imagine Wot. It learns to encode the recipe data into a compact form while retaining essential information. Here’s how it works:

- **Encoder**: Compresses the input recipe data into a latent vector.
- **Latent Space**: A compressed, abstract representation of the recipe data.
- **Decoder**: Reconstructs the recipe from the latent vector.

This architecture is particularly effective in capturing the complex structures and patterns found in culinary data.

![Visualization](https://www.mathworks.com/discovery/autoencoder/_jcr_content/mainParsys/image_copy_copy_copy_1667155049.adapt.1200.medium.svg/1718184511884.svg)
![Visualization](https://miro.medium.com/v2/resize:fit:850/1*VYH3i2-2CZ6Fyd7Bv9UHFw.png)