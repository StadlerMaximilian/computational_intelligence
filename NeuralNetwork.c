/*
 * Neural Network
 * 
 * mainly used for nonlinear regression 
 * 
 * implements a simple multi-layer Neural Network 
 * using Backpropagation-Learning with Gradient Descent (optimized version using momentum techniques)
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <unistd.h>
#include <time.h>
#include <float.h>

#define TRUE 1
#define FALSE 0

#define SIGN 1
#define NOSIGN 0

#define FLOATING 1
#define NOFLOATING 0

#define CLASSIFICATION 0
#define REGRESSION 1

#define MIN(x,y) ((x)<(y) ? (x) : (y))
#define MAX(x,y) ((x)>(y) ? (x) : (y))
#define ABS(x) ( (x) > 0 ? (x) : -(x))

#define sqr(x) ( (x) * (x) )


/////////////////////////////////////////////////
/////  DEFINE NETWORK PARAMETERS HERE   /////////
/////////////////////////////////////////////////
#define LAYERS 5
#define NEURONPERLAYER 1,4,8,4,1
#define INPUTCHANNELS 1
#define OUTPUTCHANNELS 1
#define MAXINPUTLENGTH 1000

#define PROBLEMTYPE REGRESSION

#define PRINTBOOL TRUE
#define SIMULATEANNEALING FALSE
#define MOMENTUM TRUE

#define TIMEOUT 298 //timeout for training

#define LEARNINGRATE 0.001
#define LEARNINGRATEFACTOR 10
#define LEARNINGRATEDECAYCONSTANT 0.000001
#define BIAS 1
#define WEIGHTSTARTRANGE 0.2

#define AMOUNTMAXINCREASESTEPS 2000
#define EPOCHESPERERRORCHECK 1
#define MINIMUMERRORDECREASESTEP DBL_EPSILON
#define INITIALTEMPERATURE 100
#define TEMPERATURESTEP 0.05
#define TEMPERATURERANDOMRANGE 0.01
#define OLDUPDATEFACTOR 1
////////////////////////////////////////////////




/*
 * Utility functions
 *      random
 */
double randomDouble(double from, double to){
    return ( (double)rand() / (double)RAND_MAX ) * (to - from) + from;
}

double meanValue(double *array, int size){
    if(array == NULL || size == 0){
        printf("NULL-pointer-exception!!\n");
        return 0.0;
    }
    
    double sum = 0.0;
    for(int i = 0; i < size; i++){
        sum = sum + array[i];
    }
    return sum/((double) size);
}


/*
 * Network struct and functions
 *      createNetwork
 *      freeNetwork
 *      printNetworkWeights
 *      printNetwork
 */

typedef struct Network_struct{
    double minimalError;
    double learningRate;
    double aimLearningRate;
    int epoche;
    int layers;
    int inputChannels;
    int outputChannels;
    int trained;
    double error;
    int *neuronsPerLayer;
    double *input;
    double *output;
    double **neuronNets;
    double **neuronOutputs;
    double **neuronErrorsignals;
    double ***weights;
    double ***oldUpdates; 
    
    //needed for simulated Annealing
    double ***backupWeights;
    double bestError;
    int initSA;
    double temperature;
} Network;


Network* createNetwork(int layers, int *neuronsPerLayer, int inputChannels, int outputChannels){
    if(inputChannels != neuronsPerLayer[0] || outputChannels != neuronsPerLayer[layers - 1]){
        printf("ERROR, invalid setup! (input/output channels do not fit with neuronsPerLayer.\n");
        return NULL;
    }
    //first layer is always just the input --> do not use for further calculations!!
    //last layer is always the linear output layer
    
    Network* newNetwork = calloc(1, sizeof(Network));
    
    //fill in fixed values
    newNetwork->learningRate = LEARNINGRATE;
    newNetwork->layers = layers;
    newNetwork->inputChannels = inputChannels;
    newNetwork->outputChannels = outputChannels;
    newNetwork->neuronsPerLayer = neuronsPerLayer;
    newNetwork->trained = FALSE; //setup
    newNetwork->initSA = FALSE;
    newNetwork->temperature = INITIALTEMPERATURE;
    newNetwork->aimLearningRate = LEARNINGRATE;
    
    double *output = calloc(outputChannels, sizeof(double));
    newNetwork->output = output;
    
    
    //create dynamically weight-matrix and matrix of neurons
    double ***weights = calloc(layers, sizeof(double **));
    double ***oldUpdates = calloc(layers, sizeof(double **));
    double ***backupWeights = calloc(layers, sizeof(double **));
    double **neuronOutputs = calloc(layers, sizeof(double *));
    double **neuronNets = calloc(layers, sizeof(double *));
    double **neuronErrorsignals = calloc(layers, sizeof(double *));
    
    for(int i = 0; i < layers; i++){
        weights[i] = calloc(neuronsPerLayer[i] + 1, sizeof(double *)); 
        oldUpdates[i] = calloc(neuronsPerLayer[i] + 1, sizeof(double *)); 
        backupWeights[i] = calloc(neuronsPerLayer[i] + 1, sizeof(double *)); 
        neuronOutputs[i] = calloc(neuronsPerLayer[i] + 1, sizeof(double)); 
        neuronNets[i] = calloc(neuronsPerLayer[i] + 1, sizeof(double)); 
        neuronErrorsignals[i] = calloc(neuronsPerLayer[i] + 1, sizeof(double)); 
        for(int j = 0; j < neuronsPerLayer[i] + 1; j++){
            int weightsPerNeuron = (i > 0) ? neuronsPerLayer[i-1] + 1 : inputChannels + 1;
            weights[i][j] = calloc( weightsPerNeuron , sizeof(double )  );
            oldUpdates[i][j] = calloc( weightsPerNeuron , sizeof(double )  );
            backupWeights[i][j] = calloc( weightsPerNeuron , sizeof(double )  );
            if(j == neuronsPerLayer[i]){
                neuronOutputs[i][j] = BIAS;
                neuronNets[i][j] = BIAS;
            }
        }
        for(int j = 0; j < neuronsPerLayer[i]; j++){
            int weightsPerNeuron = (i > 0) ? neuronsPerLayer[i-1] + 1 : inputChannels + 1;
            for(int k = 0; k < weightsPerNeuron; k++){
                weights[i][j][k] = randomDouble(-WEIGHTSTARTRANGE, WEIGHTSTARTRANGE);
            }
        }
    }
    
    newNetwork->weights = weights;
    newNetwork->backupWeights = backupWeights;
    newNetwork->neuronErrorsignals = neuronErrorsignals;
    newNetwork->neuronNets = neuronNets;
    newNetwork->neuronOutputs = neuronOutputs;
    newNetwork->oldUpdates = oldUpdates;
   
    return newNetwork;
}

void freeNetwork(Network *network){    
    //free 2nd layer of weights and inputs
    for(int i = 0; i < network->layers; i++){
        for(int j = 0; j < network->neuronsPerLayer[i] + 1; j++){
            free(network->weights[i][j]);
            free(network->backupWeights[i][j]);
            free(network->oldUpdates[i][j]);
        }
    }
    //free 1st layer of inputs/weights/inputs/...
    for(int i = 0; i < network->layers; i++){
        free(network->neuronErrorsignals[i]);
        free(network->neuronNets[i]);
        free(network->neuronOutputs[i]);
        free(network->weights[i]);
        free(network->oldUpdates[i]);
        free(network->backupWeights[i]);
    }
    
    //free pointers themselves
    free(network->weights);
    free(network->backupWeights);
    free(network->oldUpdates);
    free(network->input);
    free(network->output);
    free(network->neuronErrorsignals);
    free(network->neuronNets);
    free(network->neuronOutputs);
    
    //finally free network
    free(network); 
}


void printNetworkWeights(Network *network){ 
    if(network == NULL){
        printf("NULL-pointer-exception!\n");
        return;
    }
    
    printf("...Weights: \n");
    for(int i = 0; i < network->layers; i++){
        printf("layer %d:\t", i);
        for(int j = 0; j < network->neuronsPerLayer[i]; j++){
            int weightsPerNeuron = (i > 0) ? network->neuronsPerLayer[i-1] + 1 : network->inputChannels + 1;
            for(int k = 0; k < weightsPerNeuron; k++){
                printf("%f\t", network->weights[i][j][k]);
            }
            printf("\n\t\t");
        }
        printf("\n");
    }
    printf("\n");
}


void printNeuronErrorsignal(Network *network){
    if(network == NULL){
        printf("NULL-pointer-exception!\n");
        return;
    }
        
    printf("...Errorsignal: \n");
    for(int i = 0; i < network->layers; i++){
        printf("layer %d:\t", i);
        for(int j = 0; j < network->neuronsPerLayer[i]; j++){
            printf("%f\t", network->neuronErrorsignals[i][j]);
            printf("\n\t\t");
        }
        printf("\n");
    }
    printf("\n");
}

void printNeuronNet(Network *network){
    if(network == NULL){
        printf("NULL-pointer-exception!\n");
        return;
    }    
    
    printf("...Net: \n");
    for(int i = 0; i < network->layers; i++){
        printf("layer %d:\t", i);
        for(int j = 0; j < network->neuronsPerLayer[i]; j++){
            printf("%f\t", network->neuronNets[i][j]);
            printf("\n\t\t");
        }
        printf("\n");
    }
    printf("\n");
}

void printNeuronOutput(Network *network){
    if(network == NULL){
        printf("NULL-pointer-exception!\n");
        return;
    }    
    
    printf("...Output: \n");
    for(int i = 0; i < network->layers; i++){
        printf("layer %d:\t", i);
        for(int j = 0; j < network->neuronsPerLayer[i] + 1; j++){
            printf("%f\t", network->neuronOutputs[i][j]);
            printf("\n\t\t");
        }
        printf("\n");
    }
    printf("\n");
}


void printNetwork(Network *network){
    if(network == NULL){
        printf("NULL-pointer-exception!\n");
        return;
    }    
    
    printf("\n---------------------------------------------------------------------------\n");
    printf("NETWORK:\n");
    printf("\t layers: %d\n", network->layers);
    printf("\t inChan: %d\n", network->inputChannels);
    printf("\t learningRate: %f\n", network->learningRate);
    printf("\t minimalError: %f\n", network->minimalError);
    printf("\t current Error: %f\n", network->error);
    printf("\t current out: %f\n", network->output[0]);
    printf("\n\n");
    
    for(int i = 0; i < network->layers; i++){
        printf("LAYER %d:\n", i);
        printf("\tWeight:\n\t\t\t\t");
        for(int j = 0; j < network->neuronsPerLayer[i]; j++){   
            int weightsPerNeuron = (i > 0) ? network->neuronsPerLayer[i-1] + 1 : network->inputChannels + 1;
            for(int k = 0; k < weightsPerNeuron; k++){
                printf("%f ", network->weights[i][j][k]);
            }
            if(j == network->neuronsPerLayer[i]){
                printf("\n");
                break;
            }
            printf("\n\t\t\t\t");
        }      
        printf("\n");
        for(int j = 0; j < network->neuronsPerLayer[i]+1; j++){
            printf("\tnet, out, error:\t%f, %f, %f", network->neuronNets[i][j], network->neuronOutputs[i][j], network->neuronErrorsignals[i][j]);
            printf("\n");
        }      
    printf("\n");  
    }
    printf("\n");
    
    
    printf("\n---------------------------------------------------------------------------\n");
}


double ***createNetworkWeights(Network *network){
    if(network == NULL){
        printf("NULL-pointer-exception!\n");
        return NULL;
    }    
    
    //create dynamically weight-matrix
    double ***weights = calloc(network->layers, sizeof(double **));
    
    for(int i = 0; i < network->layers; i++){
        weights[i] = calloc(network->neuronsPerLayer[i]  + 1, sizeof(double *));
        for(int j = 0; j < network->neuronsPerLayer[i] + 1; j++){
            int weightsPerNeuron = (i > 0) ? network->neuronsPerLayer[i-1] + 1 : network->inputChannels + 1;
            weights[i][j] = calloc( weightsPerNeuron , sizeof(double)  );
            for(int k = 0; k < weightsPerNeuron; k++){
                weights[i][j][k] = 0.0;
            }
        }
    }
    
    return weights;
}

void copyNetworkWeights(double ***destination, Network *network){
    if(network == NULL || destination == NULL){
        printf("NULL-pointer-exception!\n");
        return;
    }    
    
    for(int i = 0; i < network->layers; i++){
        for(int j = 0; j < network->neuronsPerLayer[i] + 1; j++){
            int weightsPerNeuron = (i > 0) ? network->neuronsPerLayer[i-1] + 1 : network->inputChannels + 1;
            for(int k = 0; k < weightsPerNeuron; k++){
                destination[i][j][k] = network->weights[i][j][k];
            }
        }
    }
}

void freeNetworkWeights(double ***weights, Network *network){
    if(network == NULL || weights == NULL){
        printf("NULL-pointer-exception!\n");
        return;
    }    
    
    for(int i = 0; i < network->layers; i++){
        for(int j = 0; j < network->neuronsPerLayer[i] + 1; j++){
            free(weights[i][j]);
        }
    }
    for(int i = 0; i < network->layers; i++){
        free(weights[i]);
    }
    free(weights);
}


/*
 * Neural Network Functions
 *      propagateInput
 *      backpropagateError
 *      updateWeights
 * 
 */
void propagateInput(Network *network, double *input){  
    if(network == NULL || input == NULL){
        printf("NULL-pointer-exception!\n");
        return;
    }    
    
    if(network == NULL || input == NULL){
        printf("NULL-Pointer-Exception\n");
        return;
    }
    //set input in the input layer
    for(int i = 0; i < network->neuronsPerLayer[0]; i++){
        for(int j = 0; j < network->inputChannels; j++){//only up to neuronsPerLayer[0] --> do not set bias again
            network->neuronNets[0][j] = input[j];
            network->neuronOutputs[0][j] = input[j];
        }

    }
    
    //propagate input into hidden-layer --> layer 1 until last (index < network->layers)
    for(int i = 1; i < network->layers; i++){
        for(int j = 0; j < network->neuronsPerLayer[i]; j++){//do not propagate input into bias neurons
            double net = 0; 
            // calculate net (including bias!!)
            for(int k = 0; k < network->neuronsPerLayer[i-1]+1; k++){
                net = net + network->neuronOutputs[i-1][k] * network->weights[i][j][k];
            }
            network->neuronNets[i][j] = net;
            if( i < network->layers - 1){
                network->neuronOutputs[i][j] = tanh(network->neuronNets[i][j]);
            }
            else{
                network->neuronOutputs[i][j] = network->neuronNets[i][j];
                network->output[j] = network->neuronOutputs[i][j];
            }
        }
    }
}    


double calculateNetworkError(Network *network, double *trueOutput){
    if(network == NULL || trueOutput == NULL){
        printf("NULL-pointer-exception!\n");
        return 0.0;
    }    
    
    double error = 0;
    for(int i = 0; i < network->outputChannels; i++){
        error  = error + sqr( (trueOutput[i] - network->output[i]) ) ;
    }
    network->error =error;
    return error;
}


double calculateNetworkErrorSet(Network *network, double **inputSet, double** trueOutputSet, int setSize, int startIndex){
    double error = 0;
    for(int i = startIndex; i < setSize; i++){
        propagateInput(network, inputSet[i]);
        error = error + calculateNetworkError(network, trueOutputSet[i] );
    }
    return error;
    
}
void backpropagateError(Network *network, double *trueOutput){
    if(network == NULL || trueOutput == NULL){
        printf("NULL-pointer-exception!\n");
        return;
    }    
    
    //set error function in the output layer
    int layers = network->layers;   
    for(int i = 0; i < network->outputChannels; i++){
        network->neuronErrorsignals[ layers - 1 ][i] = 2*( trueOutput[i] - network->neuronOutputs[ layers - 1][i] ); 
    }

    //back-propagate error into network
    double delta = 0;
    for(int i = layers - 2; i > 0; i--){//loop through network layers backwards (except input layer)
        for(int j = 0; j < network->neuronsPerLayer[i]; j++){//loop through neurons in one layer (exclude bias neuron)
            delta = 0;
            for(int k = 0; k < network->neuronsPerLayer[i+1]; k++ ){//sum over all neurons in previous and sum delta_k * w_kj to delta_j (exclude bias neuron)
                delta = delta + network->neuronErrorsignals[i+1][k] * network->weights[i+1][k][j];
            }
            delta = delta *(1 - sqr( (network->neuronOutputs[i][j]) )  );
            network->neuronErrorsignals[i][j] = delta;        
        }
    }
}


void updateWeights(Network *network, double *trueOutput){
    if(network == NULL || trueOutput == NULL){
        printf("NULL-pointer-exception!\n");
        return;
    }
    
    backpropagateError(network, trueOutput);
    //skip first layer for weight update  
    for(int i = 1; i < network->layers; i++){
        for(int j = 0; j < network->neuronsPerLayer[i]; j++){
            //loop also over bias neuron!!
            for(int k = 0; k < network->neuronsPerLayer[i-1] + 1 ; k++){
                network->weights[i][j][k] += network->learningRate * network->neuronErrorsignals[i][j] * network->neuronOutputs[i-1][k];
            }
        }
    }
}

void updateWeightsMomentum(Network *network, double *trueOutput){
    if(network == NULL || trueOutput == NULL){
        printf("NULL-pointer-exception!\n");
        return;
    }
    double update = 0;
    backpropagateError(network, trueOutput);
    //skip first layer for weight update  
    for(int i = 1; i < network->layers; i++){
        for(int j = 0; j < network->neuronsPerLayer[i]; j++){
            //loop also over bias neuron!!
            for(int k = 0; k < network->neuronsPerLayer[i-1] + 1 ; k++){
                update = network->learningRate * network->neuronErrorsignals[i][j] * network->neuronOutputs[i-1][k];
                network->weights[i][j][k] += update + OLDUPDATEFACTOR * network->oldUpdates[i][j][k];
                network->oldUpdates[i][j][k] = update;
            }
        }
    }
}

void simulateAnnealing(Network *network, double **input, double **output, int setSize){
    if(network->temperature == 0){
        double ***oldWeights = network->weights;
        network->weights = network->backupWeights; //use current best weight set
        network->backupWeights = createNetworkWeights(network);
        freeNetworkWeights(oldWeights, network);
        network->temperature -= 1;
        return;
    }
    else if(network->temperature < 0){
        return;
    }
    //network->backupWeights equals x_approx
    //network->bestError equals f(x_approx)
    
    double ***weightsBackup = createNetworkWeights(network);
    double ***weightsBackupRestore = createNetworkWeights(network);
    double initialError = 0;
    double newError = 0;
    
    if(network->initSA == FALSE){
        copyNetworkWeights(network->backupWeights, network); //copy current network settings
        copyNetworkWeights(weightsBackup, network); //equals x
        copyNetworkWeights(weightsBackupRestore, network);
        
        initialError = calculateNetworkErrorSet(network, input, output, setSize, 0);
        network->bestError = initialError;
        network->initSA = TRUE;
        
    }
    else{
        copyNetworkWeights(weightsBackup, network); //equals x
        copyNetworkWeights(weightsBackupRestore, network); 

        initialError = calculateNetworkErrorSet(network, input, output, setSize, 0);   
    }
    
    
    //shake weights randomly
    for(int i = 1; i < network->layers; i++){
        for(int j = 0; j < network->neuronsPerLayer[i]; j++){
            //loop also over bias neuron!!
            for(int k = 0; k < network->neuronsPerLayer[i-1] + 1 ; k++){
                network->weights[i][j][k] += randomDouble(-TEMPERATURERANDOMRANGE, TEMPERATURERANDOMRANGE);
            }
        }
    }
    
    newError = calculateNetworkErrorSet(network, input, output, setSize, 0);//equals f(y), network->weight equals y
    
    if(newError <= initialError){
        copyNetworkWeights(weightsBackup, network); //equals x = y
        initialError = newError;
    }
    else{
        if( randomDouble(0,1) <= exp( -( (newError-initialError)/network->temperature ))){
            copyNetworkWeights(weightsBackup, network); //equals x = y
            initialError = newError;
        }
        else{//restore initial weights
            double ***oldWeights = network->weights;
            network->weights = weightsBackupRestore;
            //network backupWeights mot not be modified, not changed
            weightsBackupRestore = oldWeights;
        }
    }
    
    if(initialError < network->bestError){//if error is small, update bestError WeightSet
        double ***oldBackupWeights = network->backupWeights;
        network->bestError = initialError;
        network->backupWeights = weightsBackup;
        //network weights must not be modified, already done
        weightsBackup = oldBackupWeights;
    }

    freeNetworkWeights(weightsBackup, network);
    freeNetworkWeights(weightsBackupRestore, network);
    
    network->temperature -= TEMPERATURESTEP;
    network->epoche += 1;
}



void gradientDescent(Network *network, double **input, double **output, int setSize){
          //decay learningRate exponentially to LEARNINGRATE
        network->learningRate = LEARNINGRATEFACTOR* network->aimLearningRate * exp( - LEARNINGRATEDECAYCONSTANT*((double)network->epoche) ) + network->aimLearningRate;
        //train on traingSet
        for(int k = 0; k < EPOCHESPERERRORCHECK; k++){
            for(int i = 0; i < setSize; i++){
                propagateInput(network, input[i]);
                updateWeights(network, output[i] );
            }
            network->epoche += 1;
        }
}

void gradientDescentMomentum(Network *network, double **input, double **output, int setSize){
          //decay learningRate exponentially to LEARNINGRATE
        network->learningRate = LEARNINGRATEFACTOR* network->aimLearningRate * exp( - LEARNINGRATEDECAYCONSTANT*((double)network->epoche) ) + network->aimLearningRate;
        //train on traingSet
        for(int k = 0; k < EPOCHESPERERRORCHECK; k++){
            for(int i = 0; i < setSize; i++){
                propagateInput(network, input[i]);
                updateWeightsMomentum(network, output[i] );
            }
            network->epoche += 1;
        }
}

/*
 * Neural Network Learning
 *      trainNetwork
 *          routine of propagating input, backtracking error until error is small enough
 *      testNetwork
 *          routine of propagating a given input Set for actual testing values without error propagation
 */
void trainNetwork(Network *network,
                  double **inputSet, double **trueOutputSet, int setSize, double *outMax, int printBOOL){
    if(network == NULL || inputSet == NULL || trueOutputSet == NULL){
        printf("NULL-pointer-exception!\n");
        return;
    }    
    
    
    //start timer in train-function
    time_t startTime = time(NULL);
    
    //implements early stopping
    double error = setSize * 100.0 - 1; //high 
    double lastError = setSize * 100.0;
    int trainingSetSize = 0.7 * setSize; 
    double ***weightsBackup = createNetworkWeights(network);
    
    if(printBOOL){
        printf("%d vs %d\n", trainingSetSize, setSize);
    }
 
    int errorMinimumCounter = 0;
    
    
    while( (errorMinimumCounter < AMOUNTMAXINCREASESTEPS) && (time(NULL) < startTime + TIMEOUT) ){//train until the lasterror has been smaller than the error for 10 times      
        
        if(MOMENTUM == TRUE){
            gradientDescentMomentum(network, inputSet, trueOutputSet, trainingSetSize);
        }
        else{
            gradientDescent(network, inputSet, trueOutputSet, trainingSetSize);
        }
        
        if(SIMULATEANNEALING == TRUE){
            simulateAnnealing(network, inputSet, trueOutputSet, setSize);
        }
        
        //evaluate on evaluationSet
        lastError = error; 
        error = calculateNetworkErrorSet(network, inputSet, trueOutputSet, setSize, trainingSetSize);

        //count, how often lastError has been higher than current error(with some small deviation) --> break while, when too often
        //copy weights, when this first occurs, later restore network weights
        //thus: overfitting should be avoided
        if(lastError - error <= MINIMUMERRORDECREASESTEP){
            if(errorMinimumCounter == 0){
                copyNetworkWeights(weightsBackup, network);
 
            }
            errorMinimumCounter++;
               
        }
        else{
            errorMinimumCounter = 0;
        }
        
        if(printBOOL){
            printf("\t tested loop %d with error %f and lasterror %f,\t %f, errorIndex %d\n", network->epoche, error, lastError, lastError - error, errorMinimumCounter);
        }
        
    } //end of while training loop
    
    network->trained = TRUE;
    
    if(errorMinimumCounter >= AMOUNTMAXINCREASESTEPS){
        double ***oldWeights = network->weights;
        network->weights = weightsBackup;
        freeNetworkWeights(oldWeights, network);
    }
    
}

void testNetwork(Network *network, 
                   double **inputSet, double **resultOutputSet, int setSize){
    if(network == NULL || inputSet == NULL){
        printf("NULL-pointer-exception!\n");
        return;
    }    
    
    
    if(network->trained == FALSE){
        printf("ERROR, Network not trained!");
        return;
    }
    for(int i = 0; i < setSize; i++){
        propagateInput(network, inputSet[i]);
        for(int j = 0; j < network->outputChannels; j++){
            resultOutputSet[i][j] = network->output[j]; //just implemented 1 output neuron !!!
        }
    }
}


/*
 *  Input and Output routines
 *      normalizeInput
 *      readInput
 *      printWithSign
 *
 */
void normalizeInput(double **input, int inputLength, int inputDepth, double *retMax){
    if(input == NULL || retMax == NULL){
        printf("NULL-pointer-exception!\n");
        return;
    }    
    
    
    double *max = calloc(inputDepth, sizeof(double));
    
    for(int i = 0; i < inputDepth; i++){
        max[i] = input[0][i];
        //find maximal input per input channel
        for(int j = 0; j < inputLength;j++){
            if( ABS(input[j][i]) > max[i]){
                max[i] = ABS(input[j][i]);
            }
        }
        //divide all inputs per channel with that maximum
        for(int j = 0; j < inputLength; j++){
            input[j][i] = input[j][i] / max[i] ;
        }
    }   
    for(int i = 0; i < inputDepth; i++){
        retMax[i] = max[i]; 
    }
    free(max);
}

void normalizeOutput(double **output, int inputLength, int inputDepth, double *retMax){
    if(output == NULL || retMax == NULL){
        printf("NULL-pointer-exception!\n");
        return;
    }    
    
    
    double *max = calloc(inputDepth, sizeof(double));
    
    //find maximal input per input channel
    for(int i = 0; i < inputDepth; i++){
        max[i] = output[0][i];
        for(int j = 0; j < inputLength;j++){
            if( ABS(output[j][i]) > max[i]){
                max[i] = ABS(output[j][i]);
            }
        }
        //divide all outputs per channel with that maximum
         for(int j = 0; j < inputLength; j++){
            output[j][i] = output[j][i] / max[i] ;
        }
    }
    
    for(int i = 0; i < inputDepth; i++){
        retMax[i] = max[i]; 
    }
    free(max);
}

void scaleInput(double **input, int inputLength, int inputDepth, double max){
    if(input == NULL){
        printf("NULL-pointer-exception!\n");
        return;
    }    
    
    for(int i = 0; i < inputLength; i++){
        input[i][0] = input[i][0] / max; //only for 1 input: regression !!
    }
}


void readInput(double **trainingInput, double **trainingOutput, int *trainingSetSize,
               double **testInput, int *testSize, double *max, double *outMax){
    if(trainingInput == NULL || trainingOutput == NULL || trainingSetSize == NULL ||
            testInput == NULL || testSize == NULL || max == NULL || outMax == NULL){
        printf("NULL-pointer-exception!\n");
        return;
    }    
    
    char *str = calloc(MAXINPUTLENGTH*10, sizeof(char)); //*10 to make space for signs/digits/dots/commas in string
    int error = FALSE;
    int lengthIndex = 0;
    int startOfTestingData = FALSE;
    int lenTraining = 0;
    int lenTesting = 0;
    
    while( (scanf("%s\n", str) != EOF) && (error != TRUE) &&(lengthIndex < MAXINPUTLENGTH )){
        char *err, *p = str;
        double val = 0;;
        int depthIndex  = 0;
        
        //check for end of training data/start of testing data
        int sum = 0;
        for(int i = 0; i < strlen(str); i++, i++){
            if(str[i]-48 != 0){
                sum = 1;
                break;
            }
        }
        if(sum == 0){
            startOfTestingData = TRUE;
            lengthIndex = 0;
            continue;
        }
        
        if(startOfTestingData == FALSE){
            while (*p) {
                val = strtod(p, &err);
                if (p == err){
                    p++;
                }
                else if ((err == NULL) || (*err == 0)) {
                    //val has now last input 
                    if(depthIndex - INPUTCHANNELS < OUTPUTCHANNELS){
                        trainingOutput[lengthIndex][depthIndex-INPUTCHANNELS] = val;
                    }
                    else{
                        printf("INVALID INPUT\n");
                        error = TRUE;
                        break;
                    }
                    depthIndex++;
                    break; 
                }
                else { 
                    if(depthIndex < INPUTCHANNELS){
                        trainingInput[lengthIndex][depthIndex] = val;
                    }
                    else if(depthIndex - INPUTCHANNELS < OUTPUTCHANNELS){
                        trainingOutput[lengthIndex][depthIndex-INPUTCHANNELS] = val;
                    }
                    else{
                        printf("INVALID INPUT\n");
                        error = TRUE;
                        break;
                    }
                    depthIndex++;
                    p = err + 1; 
                }
            }
            lengthIndex++;
            lenTraining = lengthIndex;
        }
        else{
            while (*p) {
                val = strtod(p, &err);
                if (p == err){
                    p++;
                }
                else if ((err == NULL) || (*err == 0)) {
                    //val has now last input 
                    if(depthIndex < INPUTCHANNELS){
                        testInput[lengthIndex][depthIndex] = val;
                    }
                    else{
                        printf("INVALID INPUT\n");
                        error = TRUE;
                        break;
                    }
                    depthIndex++;
                    break; 
                }
                else { 
                    if(depthIndex < INPUTCHANNELS){
                        testInput[lengthIndex][depthIndex] = val;
                    }
                    else{
                        printf("INVALID INPUT\n");
                        error = TRUE;
                        break;
                    }
                    depthIndex++;
                    p = err + 1; 
                }
            }
            lengthIndex++;
            lenTesting = lengthIndex;   
        }
    }
    
    //set setSize == index after having read the input
    *trainingSetSize = lenTraining;
    normalizeInput(trainingInput, lenTraining, INPUTCHANNELS, max);
    normalizeOutput(trainingOutput, lenTraining, OUTPUTCHANNELS, outMax);
  
    *testSize = lenTesting;
    scaleInput(testInput, lenTesting, 1, *max);
    
    free(str);
}



void printOneWithSign(double value){
    value >= 0 ? printf("+1") : printf("-1");
}


/*
 * Main program
 */

int main(int argc, char** argv) { 
    
    srand(time(NULL));
    
    int printBOOL = PRINTBOOL;
    
    int layers = LAYERS;
    int neuronsPerLayer[LAYERS] = {NEURONPERLAYER};
    int inputChannels = INPUTCHANNELS;
    int outputChannels = OUTPUTCHANNELS;
   
    Network *network = createNetwork(layers, neuronsPerLayer, inputChannels, outputChannels);
    
    if(network == NULL){ 
        printf("Failed to create network!!\n");
        return(-1);
    }
    
    
    //initialize memory
    double **inputSet = calloc(MAXINPUTLENGTH, sizeof(double *));
    double **testInputSet = calloc(MAXINPUTLENGTH, sizeof(double *));
    double **outputSet = calloc(MAXINPUTLENGTH, sizeof(double));
    double **testOutputSet = calloc(MAXINPUTLENGTH, sizeof(double));
    
    if(inputSet == NULL || testInputSet == NULL || outputSet == NULL || testOutputSet == NULL){
        printf("Failed to allocate memory!!\n");
        return(-1);
    }
    
    
    for(int i = 0; i < MAXINPUTLENGTH; i++){
        inputSet[i] = calloc(inputChannels, sizeof(double));
        testInputSet[i] = calloc(inputChannels, sizeof(double));
        outputSet[i] = calloc(outputChannels, sizeof(double));
        testOutputSet[i] = calloc(outputChannels, sizeof(double));
    }
    
    int actualTrainingSetSize = 0;
    int actualTestSetSize = 0;
    double *max = calloc(inputChannels, sizeof(double));
    double *outMax = calloc(outputChannels, sizeof(double));
    
    readInput(inputSet, outputSet, &actualTrainingSetSize, testInputSet, &actualTestSetSize, max, outMax);
    
    
    if(actualTrainingSetSize == 0 || actualTestSetSize == 0){
        printf("\n");
        return(EXIT_SUCCESS);
    }
    

    
    if(printBOOL){
        printf("Training:\n");
    }
    trainNetwork(network, inputSet, outputSet, actualTrainingSetSize, outMax, printBOOL);
    
    if(printBOOL){
        printNetwork(network);
        printf("Testing:\n");
    }

    testNetwork(network, testInputSet, testOutputSet, actualTestSetSize);
    
    if(OUTPUTCHANNELS == 1){
        for(int i = 0; i < actualTestSetSize; i++){
            if(PROBLEMTYPE == REGRESSION || PROBLEMTYPE != CLASSIFICATION){
                printf("%.4f", testOutputSet[i][0]*outMax[0]);
            }
            else{
                printOneWithSign(testOutputSet[i][0]);
            }
            printf("\n");
        }
    }
    else{
        for(int i = 0; i < actualTestSetSize; i++){
            if(PROBLEMTYPE == REGRESSION || PROBLEMTYPE != CLASSIFICATION){
                printf("%.4f", testOutputSet[i][0]*outMax[0]);
            }
            for(int j = 1; j < OUTPUTCHANNELS; j++){
                if(PROBLEMTYPE == REGRESSION || PROBLEMTYPE != CLASSIFICATION){
                    printf(", %.4f", testOutputSet[i][j]*outMax[j]);
                }
            }
            printf("\n");
        }    
    }
    
    //free all used memory
    for(int i = 0; i < MAXINPUTLENGTH; i++){
            free(inputSet[i]);
            free(testInputSet[i]);
            free(outputSet[i]);
            free(testOutputSet[i]);
    }
       
    free(max);
    free(outMax);
    free(inputSet);
    free(testInputSet);
    free(outputSet);
    free(testOutputSet);    
    
    freeNetwork(network);
    
    return (EXIT_SUCCESS);
}

