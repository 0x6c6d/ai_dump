// gcc main.c -lm -o main

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define INPUTS 2
#define HIDDEN_NODES 2
#define OUTPUTS 1
#define TRAINING_SETS 4

// initialize weights with random numbers
double init_weights() { return ((double)rand()) / ((double)RAND_MAX); }

double sigmoid(double x) { return 1 / (1 + exp(-x)); }
double derivative_sigmoid(double x) { return x * (1 - x); }

void shuffle(int *array, size_t n) {
	if (n <= 1) {
		return;
	}

	for (size_t i = 0; i < n - 1; ++i) {
		size_t j = i + rand() % (n - i);
		int t = array[j];
		array[j] = array[i];
		array[i] = t;
	}
}

int main() {
	const double learning_rate = 0.1f;

	double hidden_layer[HIDDEN_NODES];
	double output_layer[OUTPUTS];

	double hidden_layer_bias[HIDDEN_NODES];
	double output_layer_bias[OUTPUTS];

	double hidden_weights[INPUTS][HIDDEN_NODES];
	double output_weights[HIDDEN_NODES][OUTPUTS];

	double training_inputs[TRAINING_SETS][INPUTS] = {
		{0.0f, 0.0f}, {0.0f, 1.0f}, {1.0f, 0.0f}, {1.0f, 1.0f}};
	double training_outputs[TRAINING_SETS][INPUTS] = {
		{0.0f}, {1.0f}, {1.0f}, {0.0f}};

	// initializing weights for the 3 layers
	for (int i = 0; i < INPUTS; i++) {
		for (int j = 0; j < HIDDEN_NODES; j++) {
			hidden_weights[i][j] = init_weights();
		}
	}

	for (int i = 0; i < HIDDEN_NODES; i++) {
		for (int j = 0; j < OUTPUTS; j++) {
			output_weights[i][j] = init_weights();
		}
	}

	for (int i = 0; i < OUTPUTS; i++) {
		output_layer_bias[i] = init_weights();
	}

	int trainingset_order[] = {0, 1, 2, 3};
	int number_of_epochs = 10001;

	// training loop for a number of epochs
	for(int epochs = 0; epochs < number_of_epochs; epochs++) {
		shuffle(trainingset_order, TRAINING_SETS);

		if(epochs % 1000 == 0) {
			printf("\nEpoch: %d\n", epochs);
		}

		for(int x = 0; x < TRAINING_SETS; x++) {
			int i = trainingset_order[x];

			// 1. FORWARD PASS

			// compute hidden layer activation
			for(int j = 0; j < HIDDEN_NODES; j++) {
				double activation = hidden_layer_bias[j];

				for(int k = 0; k < INPUTS; k++) {
					activation += training_inputs[i][k] * hidden_weights[k][j];
				}

				hidden_layer[j] = sigmoid(activation);
			}

			// compute output layer activation
			for(int j = 0; j < OUTPUTS; j++) {
				double activation = output_layer_bias[j];

				for(int k = 0; k < HIDDEN_NODES; k++) {
					activation += hidden_layer[k] * output_weights[k][j];
				}

				output_layer[j] = sigmoid(activation);
			}

			if(epochs % 1000 == 0) {
				printf("Input: %g %g | Expected Output: %g | Output: %g\n", 
					training_inputs[i][0], training_inputs[i][1], training_outputs[i][0], output_layer[0]);
			}


			// 2. BACKPROPAGATION

			// compute chage in output weigths
			double delta_output[OUTPUTS];
			for(int j = 0; j < OUTPUTS; j++){
				double error = (training_outputs[i][j] - output_layer[j]);
				delta_output[j] = error * derivative_sigmoid(output_layer[j]);
			}

			// compute change in hidden weights
			double delta_hidden[HIDDEN_NODES];
			for(int j = 0; j < HIDDEN_NODES; j++) {
				double error = 0.0f;
				for(int k = 0; k < OUTPUTS; k++) {
					error += delta_output[k] * output_weights[j][k];
				}

				delta_hidden[j] = error * derivative_sigmoid(hidden_layer[j]);
			}

			// apply changes in output weights
			for(int j = 0; j < OUTPUTS; j++) {
				output_layer_bias[j] += delta_output[j] * learning_rate;
				for(int k = 0; k < HIDDEN_NODES; k++) {
					output_weights[k][j] += hidden_layer[k] * delta_output[j] * learning_rate;
				}
			}

			// apply changes in hidden weights
			for(int j = 0; j < HIDDEN_NODES; j++) {
				hidden_layer_bias[j] += delta_hidden[j] * learning_rate;
				for(int k = 0; k < INPUTS; k++) {
					hidden_weights[k][j] += training_inputs[i][k] * delta_hidden[j] * learning_rate;
				}
			}
		}

	}

	fputs("\n\nFinal Hidden Weights\n[\n", stdout);
	for(int j = 0; j < HIDDEN_NODES; j++){
		fputs("  [ ", stdout);
		for(int k = 0; k < INPUTS; k++) {
			printf("%f ", hidden_weights[k][j]);
		}
		fputs("]\n", stdout);
	}

	fputs("]\n\nFinal Hidden Biases\n[ ", stdout);
	for(int j = 0; j < HIDDEN_NODES; j++){
		printf("%f ", hidden_layer_bias[j]);
	}
	fputs("]", stdout);

	fputs("\n\nFinal Output Weights\n[ ", stdout);
	for(int j = 0; j < OUTPUTS; j++){
		for(int k = 0; k < HIDDEN_NODES; k++) {
			printf("%f ", output_weights[k][j]);
		}
		fputs("]", stdout);
	}

	fputs("\n\nFinal Output Biases\n[ ", stdout);
	for(int j = 0; j < OUTPUTS; j++){
		printf("%f ", output_layer_bias[j]);
	}

	fputs("] \n", stdout);

	return 0;	
}
