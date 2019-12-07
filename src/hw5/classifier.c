#include <math.h>
#include <stdlib.h>
#include "image.h"
#include "matrix.h"

// Run an activation function on each element in a matrix,
// modifies the matrix in place
// matrix m: Input to activation function
// ACTIVATION a: function to run
void activate_matrix(matrix m, ACTIVATION a) {
    for (int i = 0; i < m.rows; i++) {
        double sum = 0;
        for (int j = 0; j < m.cols; j++) {
            double x = m.data[i][j];
            if (a == LOGISTIC) {
                // TD
                m.data[i][j] = 1.0 / (1.0 + exp(- x));
            } else if (a == RELU) {
                // TD
                if (x < 0) {
                    x = 0;
                }
                m.data[i][j] = x;
            } else if (a == LRELU) {
                // TD
                if (x < 0) {
                    x = 0.1 * x;
                }
                m.data[i][j] = x;
            } else if (a == SOFTMAX) {
                // TD
                m.data[i][j] = exp(x);
            }
            sum += m.data[i][j];
        }
        if (a == SOFTMAX) {
            // TD: have to normalize by sum if we are using SOFTMAX
            for (int j = 0; j < m.cols; j++) {
                m.data[i][j] = m.data[i][j] / sum;
            }
        }
    }
}

// Calculates the gradient of an activation function and multiplies it into
// the delta for a layer
// matrix m: an activated layer output
// ACTIVATION a: activation function for a layer
// matrix d: delta before activation gradient
void gradient_matrix(matrix m, ACTIVATION a, matrix d) {
    for (int i = 0; i < m.rows; i++) {
        for (int j = 0; j < m.cols; j++) {
            double x = m.data[i][j];
            // TD: multiply the correct element of d by the gradient
            if (a == LOGISTIC) {
                d.data[i][j] = x * (1.0 - x) * d.data[i][j];
            } else if (a == RELU) {
                if (x <= 0.0) { // Why only <= 0.0 works but not < 0.0? Maybe error?
                    d.data[i][j] = 0.0;
                }
            } else if (a == LRELU) {
                if (x <= 0.0) {
                    d.data[i][j] = 0.1 * d.data[i][j];
                }
            } 
            // Linear and Soft Max has gradient of 1 everywhere so nothing to multiply.
        }
    }
}

// Forward propagate information through a layer
// layer *l: pointer to the layer
// matrix in: input to layer
// returns: matrix that is output of the layer
matrix forward_layer(layer *l, matrix in) {
    l->in = in;  // Save the input for backpropagation

    // TD: multiply input by weights and apply activation function.
    matrix out = matrix_mult_matrix(l->in, l->w);
    activate_matrix(out, l->activation);

    free_matrix(l->out);  // free the old output
    l->out = out;       // Save the current output for gradient calculation
    return out;
}

// Backward propagate derivatives through a layer
// layer *l: pointer to the layer
// matrix delta: partial derivative of loss w.r.t. output of layer
// returns: matrix, partial derivative of loss w.r.t. input to layer
matrix backward_layer(layer *l, matrix delta) {
    // 1.4.1
    // delta is dL/dy
    // TD: modify it in place to be dL/d(xw)
    gradient_matrix(l->out, l->activation, delta);

    // 1.4.2
    // TD: then calculate dL/dw and save it in l->dw
    matrix xt = transpose_matrix(l->in);
    free_matrix(l->dw);  // free the previous one
    matrix dw = matrix_mult_matrix(xt, delta);
    free_matrix(xt);
    l->dw = dw;
    
    // 1.4.3
    // TD: finally, calculate dL/dx and return it.
    matrix wt = transpose_matrix(l->w);
    matrix dx = matrix_mult_matrix(delta, wt);
    free_matrix(wt);

    return dx;
}

// Update the weights at layer l
// layer *l: pointer to the layer
// double rate: learning rate
// double momentum: amount of momentum to use
// double decay: value for weight decay
void update_layer(layer *l, double rate, double momentum, double decay) {
    // TD:
    // Calculate Δw_t = dL/dw_t - λw_t + mΔw_{t-1}
    // save it to l->v
    matrix temp = axpy_matrix(-decay, l->w, l->dw);
    matrix v = axpy_matrix(momentum, l->v, temp);
    free_matrix(temp);

    free_matrix(l->v);
    l->v = v;

    // Update l->w
    matrix w = axpy_matrix(rate, l->v, l->w);
    free_matrix(l->w);
    l->w = w;
    // Remember to free any intermediate results to avoid memory leaks
}

// Make a new layer for our model
// int input: number of inputs to the layer
// int output: number of outputs from the layer
// ACTIVATION activation: the activation function to use
layer make_layer(int input, int output, ACTIVATION activation) {
    layer l;
    l.in  = make_matrix(1,1);
    l.out = make_matrix(1,1);
    l.w   = random_matrix(input, output, sqrt(2./input));
    l.v   = make_matrix(input, output);
    l.dw  = make_matrix(input, output);
    l.activation = activation;
    return l;
}

// Run a model on input X
// model m: model to run
// matrix X: input to model
// returns: result matrix
matrix forward_model(model m, matrix X) {
    int i;
    for(i = 0; i < m.n; ++i){
        X = forward_layer(m.layers + i, X);
    }
    return X;
}

// Run a model backward given gradient dL
// model m: model to run
// matrix dL: partial derivative of loss w.r.t. model output dL/dy
void backward_model(model m, matrix dL) {
    matrix d = copy_matrix(dL);
    int i;
    for(i = m.n-1; i >= 0; --i){
        matrix prev = backward_layer(m.layers + i, d);
        free_matrix(d);
        d = prev;
    }
    free_matrix(d);
}

// Update the model weights
// model m: model to update
// double rate: learning rate
// double momentum: amount of momentum to use
// double decay: value for weight decay
void update_model(model m, double rate, double momentum, double decay) {
    int i;
    for(i = 0; i < m.n; ++i){
        update_layer(m.layers + i, rate, momentum, decay);
    }
}

// Find the index of the maximum element in an array
// double *a: array
// int n: size of a, |a|
// returns: index of maximum element
int max_index(double *a, int n) {
    if(n <= 0) return -1;
    int i;
    int max_i = 0;
    double max = a[0];
    for (i = 1; i < n; ++i) {
        if (a[i] > max){
            max = a[i];
            max_i = i;
        }
    }
    return max_i;
}

// Calculate the accuracy of a model on some data d
// model m: model to run
// data d: data to run on
// returns: accuracy, number correct / total
double accuracy_model(model m, data d) {
    matrix p = forward_model(m, d.X);
    int i;
    int correct = 0;
    for(i = 0; i < d.y.rows; ++i){
        if(max_index(d.y.data[i], d.y.cols) == max_index(p.data[i], p.cols)) ++correct;
    }
    return (double)correct / d.y.rows;
}

// Calculate the cross-entropy loss for a set of predictions
// matrix y: the correct values
// matrix p: the predictions
// returns: average cross-entropy loss over data points, 1/n Σ(-ylog(p))
double cross_entropy_loss(matrix y, matrix p) {
    int i, j;
    double sum = 0;
    for(i = 0; i < y.rows; ++i){
        for(j = 0; j < y.cols; ++j){
            sum += -y.data[i][j]*log(p.data[i][j]);
        }
    }
    return sum/y.rows;
}


// Train a model on a dataset using SGD
// model m: model to train
// data d: dataset to train on
// int batch: batch size for SGD
// int iters: number of iterations of SGD to run (i.e. how many batches)
// double rate: learning rate
// double momentum: momentum
// double decay: weight decay
void train_model(model m, data d, int batch, int iters, double rate, double momentum, double decay) {
    int e;
    for(e = 0; e < iters; ++e){
        data b = random_batch(d, batch);
        matrix p = forward_model(m, b.X);
        fprintf(stderr, "%06d: Loss: %f\n", e, cross_entropy_loss(b.y, p));
        matrix dL = axpy_matrix(-1, p, b.y); // partial derivative of loss dL/dy
        backward_model(m, dL);
        update_model(m, rate/batch, momentum, decay);
        free_matrix(dL);
        free_data(b);
    }
}


// Questions 
//
// 5.2.2.1 Why might we be interested in both training accuracy and testing accuracy? 
// What do these two numbers tell us about our current model?
// 
// We are interested in both accuracies since we want our model to be as accurate as 
// possible but not overfitting to the training data at the same time.
// The training accuracy tells us how good our current model is doing on the training 
// data, and the testing accuracy tells us how good our current model is doing on the 
// test data, combining them together we know how much farway we are currently on the
// accuracy of test data and training data, which gives hint of how much our current 
// model is overfitting to the training data.
//
//
// 5.2.2.2 Try varying the model parameter for learning rate to different powers of 10 
// (i.e. 10^1, 10^0, 10^-1, 10^-2, 10^-3) and training the model. What patterns do you 
// see and how does the choice of learning rate affect both the loss during training and 
// the final model accuracy?
// 
// rate = 10:
//     training accuracy: 0.09915;
//     test accuracy: 0.1009;
//     pattern during training: loss start to be nan frrom the 4th round, this indicates 
//         that the gradient decent is overshooting.
// rate = 1:
//     training accuracy: 0.8505833;
//     test accuracy: 0.8463;
//     pattern during training: loss varies a lot and never converge, probably still 
//         overrshooting.
// rate = 0.1:
//     training accuracy: 0.92071666;
//     test accuracy: 0.9171;
//     pattern during training: loss dropped pertty quickly and stablized at about 0.3.
// rate = 0.01:
//     training accuracy: 0.9034333;
//     test accuracy: 0.9091;
//     pattern during training: loss dropped slower than before and stablized at about 0.3.
// rate = 0.001:
//     training accuracy: 0.85903;
//     test accuracy: 0.8669;
//     pattern during training: loss dropped even slower and stopped beforer it stablized.
//         based on the final accuracy, the prograrm stopped beforer reaching a minimum.
// Overall, a higher learning rate helps the training converge to local minima quicker,
//     but too high a rate could make the gradient decent overshoot and never converge.
//
//
// 5.2.2.3 Try varying the parameter for weight decay to different powers of 10: 
// (10^0, 10^-1, 10^-2, 10^-3, 10^-4, 10^-5). How does weight decay affect the final 
// model training and test accuracy?
// 
// decay = 1:
//     training accuracy: 0.895816667;
//     test accuracy: 0.8968;
// decay = 0.1:
//     training accuracy: 0.9151833;
//     test accuracy: 0.9144;
// decay = 0.01:
//     training accuracy: 0.920067;
//     test accuracy: 0.9166;
// decay = 0.001:
//     training accuracy: 0.9206667;
//     test accuracy: 0.917;
// decay = 0.0001:
//     training accuracy: 0.9207;
//     test accuracy: 0.9171;
// decay = 0.00001:
//     training accuracy: 0.9207167;
//     test accuracy: 0.9171;
// decay = 0:
//     training accuracy: 0.9207167;
//     test accuracy: 0.9171;
// Overall, seems like the lower the decay is, the higher the accuracy will be.
// But the trade off is that with a lower decay, the model is moer likely to overfit
// on the training data.
//
//
// 5.2.3.1 Currently the model uses a logistic activation for the first layer. 
// Try using a the different activation functions we programmed. How well do they perform? 
// What's best?
// 
// LOGISTIC:
//     training accuracy: 0.889283;
//     test accuracy: 0.8949;
// RELU:
//     training accuracy: 0.92605;
//     test accuracy: 0.9281;
// LRELU:
//     training accuracy: 0.92433;
//     test accuracy: 0.9263;
// LINEAR:
//     training accuracy: 0.9133167;
//     test accuracy: 0.9162;
// SOFTMAX:
//     training accuracy: 0.0.6131167;
//     test accuracy: 0.6043;
// RELU performs the best.
//     
//
// 5.2.3.2 Using the same activation, find the best (power of 10) learning rate for your 
// model. What is the training accuracy and testing accuracy?
// 
// rate = 10:
//     training accuracy: 0.09915;
//     test accuracy: 0.1009;
// rate = 1:
//     training accuracy: 0.09915;
//     test accuracy: 0.1009;
// rate = 0.1:
//     training accuracy: 0.95093333;
//     test accuracy: 0.9439;
// rate = 0.01:
//     training accuracy: 0.92605;
//     test accuracy: 0.9281;
// rate = 0.001:
//     training accuracy: 0.86703333;
//     test accuracy: 0.8766;
// rate = 0.1 works the best, with 
//     training accuracy: 0.95093333;
//     test accuracy: 0.9439;
//
//
// 5.2.3.3 Right now the regularization parameter `decay` is set to 0. Try adding some decay 
// to your model. What happens, does it help? Why or why not may this be?
//
// decay = 0:
//     training accuracy: 0.95093333;
//     test accuracy: 0.9439; 
// decay = 1:
//     training accuracy: 0.9239;
//     test accuracy: 0.9262;
// decay = 0.1:
//     training accuracy: 0.94935;
//     test accuracy: 0.9443;
// decay = 0.01:
//     training accuracy: 0.94875;
//     test accuracy: 0.9448;
// decay = 0.001:
//     training accuracy: 0.945683;
//     test accuracy: 0.9393;
// decay = 0.0001:
//     training accuracy: 0.94925;
//     test accuracy: 0.9433;
// Decay = 0.1 worked the best. The test accuracy improved to 0.9443, although the training 
// accuracy decrerased. This is because adding some decay prevents the model from overfitting
// to the training data. More decay added means less overfit but the model is also harder
// to reach the minimum.
// 
//
// 5.2.3.4 Modify your model so it has 3 layers instead of two. The layers should be 
// `inputs -> 64`, `64 -> 32`, and `32 -> outputs`. Also modify your model to train for 
// 3000 iterations instead of 1000. Look at the training and testing error for different 
// values of decay (powers of 10, 10^-4 -> 10^0). Which is best? Why?
// 
// decay = 0:
//     training accuracy: 0.9770833;
//     test accuracy: 0.9629; 
// decay = 1:
//     training accuracy: 0.92315;
//     test accuracy: 0.9245;
// decay = 0.1:
//     training accuracy: 0.97535;
//     test accuracy: 0.9666;
// decay = 0.01:
//     training accuracy: 984333;
//     test accuracy: 0.9704;
// decay = 0.001:
//     training accuracy: 0.979816;
//     test accuracy: 0.9662;
// decay = 0.0001:
//     training accuracy: 0.98245;
//     test accuracy: 0.9667;
// Decay = 0.01 worked the best. The same reason as before, more decay means less overfit
// but also means it's harder to reach the minimum, therefore there's a balance point in
// the middle.
// 
//
// 5.3.2.1 How well does your network perform on the CIFAR dataset?
// 
// With pararmeters:
//     batch = 128
//     iters = 3000
//     rate = 0.1
//     momentum = .9
//     decay = 0.001
// and a neural network with 3 layers:
//     make_layer(inputs, 128, LOGISTIC),
//     make_layer(128, 64, LOGISTIC),
//     make_layer(64, outputs, SOFTMAX)
// I got result: 
//     training accuracy: 0.4724;
//     test accuracy: 0.456;



