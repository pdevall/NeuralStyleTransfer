# Neural Style Transfer
Neural Style Transfer is a process of applying the Style of paintings to the Content Image.<br>
Suppose there is a content image and a Style Image shown below

![alt text](https://github.com/pdevall/NeuralStyleTransfer/blob/master/NeuralStyleTransfer/images/content1_small.jpg) ![alt text](https://github.com/pdevall/NeuralStyleTransfer/blob/master/NeuralStyleTransfer/images/style3_small.jpg)

Once the Neural Style transfer is applied the output image will be show below <br>
![alt text](https://github.com/pdevall/NeuralStyleTransfer/blob/master/NeuralStyleTransfer/images/inputB2.jpg)

# Steps
* **Visualization**
* **Preprocessing**
* **Model Creation**
* **Compute Loss**
* **Compute Gradients**
* **Run Model**

```python
from keras.preprocessing.image import load_img
from keras.preprocessing.image import save_img
from keras.preprocessing import image
import numpy as np
%matplotlib inline
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf


```


```python
img_w = 400
img_h = 400
```


```python
def load_reshape_img(imagePath):
    img  = load_img(imagePath).resize((img_w, img_h))
    imgArray = image.img_to_array(img)
    imgArray = np.reshape(imgArray, ((1,) + imgArray.shape))
    return imgArray, img
```


```python
def deprocess_image(input_image):
    x = np.reshape(input_image, (img_w, img_h, 3))
    # Remove zero-center by mean pixel
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    # 'BGR'->'RGB'
    x = x[:, :, ::-1]
    x = np.clip(x, 0, 255).astype('uint8')
    return x
```


```python
contentImgTest, contentImg = load_reshape_img("images/content1.jpg")
styleImgTest, styleImg = load_reshape_img("images/style3.jpg")

plt.imshow(contentImg)
plt.title("Content Image")
plt.show()
plt.imshow(styleImg)
plt.title("Style Image")
plt.show()

contentImg.save("images/content1_small.jpg")
styleImg.save("images/style3_small.jpg")

print("Content Image Shape :" + str(contentImgTest.shape) )
print("Style Image Shape :" + str(styleImgTest.shape) )
```

```python
imputImgArray = np.random.rand(img_w,img_h, 3) * 255
inputImage = Image.fromarray(imputImgArray.astype('uint8')).convert('RGB')
inputImage.save("images/input.jpg")
```


```python
inputImageTest, inputImg = load_reshape_img("images/input.jpg")

plt.imshow(inputImg)
plt.title("Input Image")
plt.show()
print("Input Image Shape :" + str(inputImageTest.shape) )

```

```python
content_layers = [('block5_conv2', 0.025)]
style_layers = [('block1_conv1', .2),
                ('block2_conv1', .125),
                ('block3_conv1', 1.),
                ('block4_conv1', .3),
                ('block5_conv1', 1.)
               ]

num_content_layers = len(content_layers)
num_style_layers = len(style_layers)

```


```python
def get_model():
    modelVGG = tf.keras.applications.vgg19.VGG19(weights='imagenet', include_top=False)
    modelVGG.trainable = False
    model_layers = [modelVGG.get_layer(styleLayerName).output for styleLayerName, weight in style_layers]
    model_layers = model_layers + [modelVGG.get_layer(contentLayerName).output for contentLayerName, weight in content_layers]  
    model = tf.keras.Model(modelVGG.input, model_layers)
    return model
```


```python
model = get_model()
```


```python
contentImageArray, _ = load_reshape_img("images/content.jpg")
styleImgArray, _ = load_reshape_img("images/style3.jpg")
contentImageArray = tf.keras.applications.vgg19.preprocess_input(tf.convert_to_tensor(contentImageArray))
styleImgArray = tf.keras.applications.vgg19.preprocess_input(tf.convert_to_tensor(styleImgArray))
contentActivations = model(contentImageArray)
styleActivations = model(styleImgArray)
 

```


```python
def compute_content_cost(contentActivation, targetActivation):
    m, n_w, n_h, n_c = targetActivation.get_shape().as_list()
    contentActivationUnrolled = tf.reshape(contentActivation, [m, n_w * n_h, n_c])
    targetActivationUnrolled =  tf.reshape(targetActivation, ((m, n_w * n_h, n_c)))
    content_cost = tf.math.reduce_sum(tf.square(tf.subtract(contentActivationUnrolled, targetActivationUnrolled)))
    return content_cost
    
```


```python
def gram_matrix(activation):
    gramMatrixOfActivation = tf.matmul(activation, tf.transpose(activation))
    return gramMatrixOfActivation
```


```python
def compute_layer_style_cost(styleActivation, targetActivation):
    m, n_w, n_h, n_c = targetActivation.get_shape().as_list()
    styleActivationUnrolled = tf.transpose(tf.reshape(styleActivation, ((n_w * n_h, n_c))), perm=[1, 0])
    targetActivationUnrolled = tf.transpose(tf.reshape(targetActivation, ((n_w * n_h, n_c))), perm=[1, 0])
    styleGramMatrix = gram_matrix(styleActivationUnrolled)
    targetGramMatrix = gram_matrix(targetActivationUnrolled)
    style_layer_cost =  tf.math.reduce_sum(tf.square(tf.subtract(styleGramMatrix, targetGramMatrix)))/(4 * (n_c * n_c) * (n_w * n_h) * (n_w * n_h))
    return style_layer_cost
```


```python
def compute_total_content_cost(inputActivations):
    contentOnlyActivations = contentActivations[num_style_layers:]
    inputContentOnlyActivations = inputActivations[num_style_layers:]
    content_cost = 0.
    for layer_index in range(len(inputContentOnlyActivations)):
         content_cost = content_cost + content_layers[layer_index][1] * compute_content_cost(contentOnlyActivations[layer_index], inputContentOnlyActivations[layer_index])
    return content_cost
```


```python
def compute_total_style_cost(inputActivations):
    styleOnlyActivations = styleActivations[:num_style_layers]
    inputContentOnlyActivations = inputActivations[:num_style_layers]
    style_cost = 0.
    for layer_index in range(len(inputContentOnlyActivations)):
         style_cost = style_cost + style_layers[layer_index][1] * compute_layer_style_cost(styleOnlyActivations[layer_index], inputContentOnlyActivations[layer_index])
    return style_cost
```


```python
def compute_total_cost(cfg, alpha=10, beta=40):
    inputActivations = model(cfg['init_image'])
    totalContentCost = compute_total_content_cost(inputActivations)
    totalStyleCost = compute_total_style_cost(inputActivations)
    totalCost = alpha * totalContentCost + beta * totalStyleCost
    
    return totalCost, totalContentCost, totalStyleCost
```


```python
def compute_grads(cfg):
    with tf.GradientTape() as tape: 
        all_loss = compute_total_cost(cfg)
    return tape.gradient(all_loss[0], cfg['init_image']), all_loss
```


```python
def model_nn(input_image_path, num_iterations = 1001):
    inputImageArray, _ = load_reshape_img(input_image_path)
    inputImageArray = tf.keras.applications.vgg19.preprocess_input(inputImageArray)
    inputImageArray = tf.Variable(tf.convert_to_tensor(inputImageArray), trainable=True, dtype=tf.float32)
    iter_count = 1
   
    
    cfg = {
      'init_image': inputImageArray,
    }
    optimizer = tf.keras.optimizers.Adam(10.0)
    
    for i in range(num_iterations):
        grads, all_loss = compute_grads(cfg)
        optimizer.apply_gradients([(grads, inputImageArray)])
         # Print every 20 iteration.
        if i%50 == 0:
            print("total Loss at Iteration :" +str(i) + str(all_loss[0]))
            print("total content Loss at Iteration :" +str(i) + str(all_loss[1]))
            print("total style Loss at Iteration :" +str(i) + str(all_loss[2]))
            # Display intermediate images
            # Use the .numpy() method to get the concrete numpy array
            img = inputImageArray.numpy()
            plot_img = deprocess_image(img)
            plot_img = Image.fromarray(plot_img)
            plot_img.save("images/inputB"+str(iter_count)+".jpg")
            iter_count = iter_count + 1
    return inputImageArray
    
```


```python
model_nn("images/content1.jpg")
```
