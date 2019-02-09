/*
 * Vector Quantization
 * 
 * implements a VQ-Algorithm that is able find centers of clusters (given the 
 * amount of clusters) by using the kmeans++ algorithm for initialization and standard
 * kmeans-algorithm
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

#define MAXINPUTLENGTH 1000
#define DIMENSION 2

#define PROBLEMTYPE REGRESSION

#define PRINTBOOL FALSE


#define TIMEOUT 10 //timeout for training

#define LEARNINGRATEINITIAL 0.5
#define NUMITER 1000
#define DUPLICATETHRESHOLD 0.1
#define LLOYDSTOPPINGTHRESHOLD 0.0000001

/////////////////////////////////////////////////




/*
 * Utility functions
 *      random
 */
void myMemsetDouble(double *ptr, double value, int size){
    for(int i = 0; i < size; i++){
        ptr[i] = value;
    }
}

double randomDouble(double from, double to){
    return ( (double)rand() / (double)RAND_MAX ) * (to - from) + from;
}

int randomInt(int from, int to){
    return ( rand() % (to - from)) + from;  
}


int randomIndexWeighted(double *minDistancesSquare, int inputSize){
    double *partialSumDistances = calloc(inputSize, sizeof(double));
    int randomIndex = -1;
    double randomDistance = -1;
    
    //create array with partial sum of distances squared
    for(int i = 0; i < inputSize; i++){
        for(int j = 0; j <= i; j++){
            partialSumDistances[i] = partialSumDistances[i] + minDistancesSquare[j];    
        }
    }
    
    
    randomDistance = randomDouble(0, partialSumDistances[inputSize - 1]);

    for(int i = 0; i < inputSize; i++){
        if(i == (inputSize - 1)){
            randomIndex = i;
            break;
        }
        else if((partialSumDistances[i] <= randomDistance) && (randomDistance < partialSumDistances[i+1]) ){
            randomIndex = i+1;
            break;
        }
    }
    

    
    free(partialSumDistances);
    return randomIndex;
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
 * Vector Functions
 */
typedef struct vectorStruct{
    int dimension;
    int cluster;
    double *coordinates;
} Vector;

Vector *createVector(int dimension){
    Vector *vector = calloc(1, sizeof(Vector)); 
    double *coordinates = calloc(dimension, sizeof(double));
    vector->coordinates = coordinates;
    vector->dimension = dimension;
    
    return vector;
}

void freeVector(Vector *vector){
    if(vector == NULL){
        printf("ERROR: null-pointer-exception\n");
        return;
    }
    
    free(vector->coordinates);
    free(vector);
}

Vector* addVectors(Vector *a, Vector *b){
    if(a == NULL || b == NULL){
        printf("ERROR: null-pointer-exception\n");
        return NULL;
    }
    
    if(a->dimension != b->dimension){
        printf("ERROR: Dimension do not agree.!\n");
        return NULL;
    }
    
    int dimension = a->dimension;
    Vector *result = createVector(dimension);
    for(int i = 0; i < dimension; i++){
        result->coordinates[i] = a->coordinates[i] + b->coordinates[i];
    }
    
    return result;
}

void addToVector(Vector* target, Vector* added){
        if(target == NULL || added == NULL){
        printf("ERROR: null-pointer-exception\n");
        return;
    }
    
    if(target->dimension != added->dimension){
        printf("ERROR: Dimension do not agree.!\n");
        return;
    }
    
    for(int i = 0; i < target->dimension; i++){
        target->coordinates[i] += added->coordinates[i];
    }
    
}



Vector* subVectors(Vector *a, Vector *b){
    if(a == NULL || b == NULL){
        printf("ERROR: null-pointer-exception\n");
        return NULL;
    }
    
    if(a->dimension != b->dimension){
        printf("ERROR: Dimension do not agree.!\n");
        return NULL;
    }
    
    int dimension = a->dimension;
    Vector *result = createVector(dimension);
    for(int i = 0; i < dimension; i++){
        result->coordinates[i] = a->coordinates[i] - b->coordinates[i];
    }
    
    return result;
}

Vector *multVectors(Vector *v, double scalar){    
    if(v == NULL){
        printf("ERROR: null-pointer-exception\n");
        return NULL;
    }
    
    int dimension = v->dimension;
    Vector *result = createVector(dimension);
    for(int i = 0; i < dimension; i++){
        result->coordinates[i] = v->coordinates[i] * scalar;
    }
    
    return result;
}

void multToVector(Vector *target, double scalar){
    if(target == NULL){
        printf("ERROR: null-pointer-exception\n");
        return;
    }
    
    for(int i = 0; i < target->dimension; i++){
        target->coordinates[i] *= scalar;
    }
}

double vectorLength(Vector *v){
    if(v == NULL){
        printf("ERROR: null-pointer-exception\n");
        return 0.0;
    }
    
    double len = 0;
    
    for(int i = 0; i < v->dimension; i++){
        len = len + sqr( v->coordinates[i] );
    }
    
    len = sqrt(len);
    
    return len;
    
}

double vectorDistance(Vector *p1, Vector *p2){
    if(p1 == NULL || p2 == NULL){
        printf("ERROR: null-pointer-exception\n");
        return 0.0;
    }
    
    if(p1->dimension != p2->dimension){
        printf("ERROR: Dimension do not agree.!\n");
        return 0.0;
    }
       
    
    Vector *dif = subVectors(p1, p2);
    double len = vectorLength(dif);
    free(dif);
    return len;    
}

void printVector(Vector *v){
    int dim = v->dimension;
    printf("\nVector: \n");
    printf("\t dimension: %d\n", dim);
    printf("\t coordinates: ");
    for(int i = 0; i < dim; i++){
        if(i == 0)
            printf("%lf", v->coordinates[i]);
        else
            printf(" ,%lf", v->coordinates[i]);
    }
    printf("\n");
}

/*
 * Double Matrix Functions
 */
double **createDoubleMatrix(int dim1, int dim2){
    double **mat = calloc(dim1, sizeof(double *));
    for(int i = 0; i < dim1; i++){
        mat[i] = calloc(dim2, sizeof(double));
    }
    
    return mat;
}

void freeDoubleMatrix(double **mat, int dim1){
    for(int i = 0; i < dim1; i++ ){
        free(mat[i]);
    }
    free(mat);
}

/*
 *  Input and Output routines
 *      read 1st the number of clusters, then unknown amount of x,y coordinates
 * 
 *      normalizeInput
 *      readInput
 *      printWithSign
 *
 */
void normalizeInput(double **input, int inputLength, double *retMax, int dimension){
    if(input == NULL || retMax == NULL){
        printf("NULL-pointer-exception!\n");
        return;
    }    
    
    
    double *max = calloc(dimension, sizeof(double));
    
    for(int i = 0; i < dimension; i++){
        max[i] = input[0][i];
        //find maximal input per input channel
        for(int j = 0; j < inputLength;j++){
            if( ABS(input[j][i]) > max[i]){
                max[i] = ABS(input[j][i]);
            }
        }
        //divide all inputs per channel with that maximum
        for(int j = 0; j < inputLength; j++){
            input[j][i] = input[j][i] / max[i];
        }
    }   
    for(int i = 0; i < dimension; i++){
        retMax[i] = max[i]; 
    }
    free(max);
}


int readAmountClusters(void){
    int clu = 0;
    if( scanf("%d\n", &clu) != EOF){
        return clu;
    }
    else{
        printf("ERROR: Failed to determine amount of clusters\n");
        return -1;
    }
}


void readInput(int *inputSize, double **input, double *scalingFactor){
    int lengthIndex = 0;
    double x = 0;
    double y = 0;
    
    while( scanf("%lf, %lf\n", &x, &y) != EOF){
        input[lengthIndex][0] = x;
        input[lengthIndex][1] = y;
        lengthIndex++;
    }
    
    *inputSize = lengthIndex;
    
    normalizeInput(input, *inputSize, scalingFactor, DIMENSION);

}


/*
 * Functions for Vector Quantization
 */

typedef struct vqStruct{
    int amountClusters;
    Vector **prototypes;
    
    double learningRate;
    
} VQ;

VQ *createVQ(int clusters){
    VQ *newVQ = calloc(1, sizeof(VQ));
    
    Vector **prototypes = calloc(clusters, sizeof(Vector *));
    for(int i = 0; i < clusters; i++){
        prototypes[i] = createVector(DIMENSION);
        for(int j = 0; j < DIMENSION; j++){
            prototypes[i]->coordinates[j] = randomDouble(-1,1); //since normalized input to -10 - 10
        }
    }
    
    newVQ->prototypes = prototypes;
    newVQ->amountClusters = clusters; 
    newVQ->learningRate = LEARNINGRATEINITIAL;
    
    return newVQ;
}

void freeVQ(VQ *vq){
    for(int i = 0; i < vq->amountClusters; i++){
        freeVector(vq->prototypes[i]);
    }
    free(vq->prototypes);
}

void printVQ(VQ *vq, double *scalingFactors){
    if(PRINTBOOL){
        printf("amountClusters: %d\n", vq->amountClusters);
        printf("learningRate: %f\n", vq->learningRate);
    }
    for(int i = 0; i < vq->amountClusters; i++){
        printf("%.4f", vq->prototypes[i]->coordinates[0]*scalingFactors[0]);
        for(int j = 1; j < DIMENSION; j++){
            printf(", %.4f", vq->prototypes[i]->coordinates[j]*scalingFactors[j]);
        }
        printf("\n");
    }
    
    
}


double updateWinningPrototype(VQ *vq, Vector *dataPoint){
    int amountClusters = vq->amountClusters;
    double *distances = calloc(amountClusters, sizeof(double));
    
    //calculate distances between dataPoint and cluster centers
    for(int i = 0; i < amountClusters; i++){
        distances[i] = vectorDistance(vq->prototypes[i], dataPoint);
    }
    
    //select winning prototype
    int minIndex = 0;
    double minDist = distances[0];
    for(int i = 0; i < amountClusters; i++){
        if(distances[i] < minDist){
            minDist = distances[i];
            minIndex = i;
        }
    }
    
    //update winning prototype
    Vector *update_dif = subVectors(dataPoint, vq->prototypes[minIndex]);
    Vector *update = multVectors(update_dif, vq->learningRate);
    double updateSize = vectorLength(update); //determine size of update
    Vector *oldProto = vq->prototypes[minIndex];
    vq->prototypes[minIndex] = addVectors(oldProto, update);
    
    //free intermediate results
    freeVector(update_dif);
    freeVector(oldProto);
    free(distances);
    
    return updateSize;
}


void deleteDuplicatePrototypes(VQ *vq){
    if(vq->amountClusters < 1){
        return;
    }
    
    double distance = -1;
    Vector **oldCenters = vq->prototypes;
    Vector **newCenters = calloc(vq->amountClusters - 1, sizeof(Vector*));
    
    
    
    for(int i = 0; i < vq->amountClusters; i++){
        for(int j = 0; j < vq->amountClusters; j++){
            if(i == j){
                continue; 
            }
            
            distance = vectorDistance(vq->prototypes[i], vq->prototypes[j]);
            if( (distance <= (double)DUPLICATETHRESHOLD) && (distance != -1) && (vq->amountClusters > 1) ){
                //delete that prototype
                //first: calculate the median of the pair of close prototypes
                Vector *median_1 = addVectors(vq->prototypes[i], vq->prototypes[j]);
                Vector *median = multVectors(median_1, 0.5);
                freeVector(median_1);
                
                

                for(int k = 0; k < vq->amountClusters - 1; k++){
                    newCenters[k] = calloc(1, sizeof(Vector));
                }

                for(int k = 0; k < vq->amountClusters; k++){
                    if(k == i){
                        memcpy(newCenters[k], median, sizeof(Vector));
                    }
                    else if(k == j){
                        continue;
                    }
                    else if(k < j){
                        memcpy(newCenters[k], oldCenters[k], sizeof(Vector));
                    }
                    else{
                        memcpy(newCenters[k-1], oldCenters[k], sizeof(Vector));
                    }
                }
                vq->prototypes = newCenters;
                vq->amountClusters = vq->amountClusters - 1;
                freeVector(oldCenters[i]);
                return; 
            }
        }
    }
}


void deleteEmptyPrototypes(VQ *vq, Vector **dataPoints, int inputSize){
    double currentDistance = -1;
    double otherDistance = -1;
    double otherDistanceMin = 100;
    
    if(vq->amountClusters < 1){
        return;
    }
    
    for(int i = 0; i < vq->amountClusters; i++){
        int emptyBool = TRUE;
        
        for(int j = 0; j < inputSize; j++){
            currentDistance = vectorDistance(vq->prototypes[i], dataPoints[j]);
            otherDistanceMin = 100;
            
            for(int k = 0; k < vq->amountClusters; k++){
                if(k == i){ 
                    continue;
                }
                
                otherDistance = vectorDistance(vq->prototypes[k], dataPoints[j]);
                if(otherDistance < otherDistanceMin){
                    otherDistanceMin = otherDistance;
                }
                otherDistance = -1;
            }
            if(currentDistance < otherDistanceMin && currentDistance != -1 && (otherDistanceMin != -1 || otherDistanceMin != 100  )  ){
                    emptyBool = FALSE;
                    break;
            }
            currentDistance = -1;
        }
        
        if(emptyBool == TRUE && (vq->amountClusters > 1)){
            //if emptyBool still true, then prototype i is empty
            Vector **oldCenters = vq->prototypes;
            Vector **newCenters = calloc(vq->amountClusters - 1, sizeof(Vector*));
            for(int j = 0; j < vq->amountClusters - 1; j++){
                newCenters[j] = calloc(1, sizeof(Vector));
            }
            
            for(int j = 0; j < vq->amountClusters; j++){
                if(j < i){
                    memcpy(newCenters[j], oldCenters[j], sizeof(Vector));
                }
                else if(j == i){
                    continue;
                }
                else{
                    memcpy(newCenters[j-1], oldCenters[j], sizeof(Vector));
                }
                
                
            }
            vq->prototypes = newCenters;
            vq->amountClusters = vq->amountClusters - 1;
            freeVector(oldCenters[i]);
            return; 
        }
        
    }
    
}


VQ *kMeansPlusPlus(Vector **dataPoints, int inputSize, int clusters){
    //implement k-means++ algorithm to determine the initial location of clusters
    
    
    //set up empty VQ
    VQ *newVQ = calloc(1, sizeof(VQ));
    Vector **prototypes = calloc(clusters, sizeof(Vector *));
    for(int i = 0; i < clusters; i++){
        prototypes[i] = createVector(DIMENSION);
    }

    
    
    //choose 1st center randomly from data points and copy 
    //  its coordinates into the 0th prototype
    int *randomIndices = calloc(clusters, sizeof(int));
    memset(randomIndices, -1, clusters*sizeof(int));
    randomIndices[0] = randomInt(0, inputSize);
    Vector *randomDataPoint = dataPoints[randomIndices[0]];
    memcpy(prototypes[0]->coordinates, randomDataPoint->coordinates, DIMENSION*sizeof(double));

    //now find for each data point the distance between that point and the nearest 
    //  chosen center
    double *minDistancesSquare = calloc(inputSize, sizeof(double));
    myMemsetDouble(minDistancesSquare, 10, inputSize);
    double distance = 1000;
    
    for(int i = 1; i < clusters; i++){
        //loop through the already assigned clusters (increases every loop)
        for(int j = 0; j < i; j++){
            //now loop through all the dataPoints and find minDistances
            for(int k = 0; k < inputSize; k++){
                distance = vectorDistance(prototypes[j], dataPoints[k]);
                if(distance < minDistancesSquare[k]){
                    minDistancesSquare[k] = distance;
                    distance = 1000;
                }
            }         
        }

        for(int j = 0; j < inputSize; j++){
            minDistancesSquare[j] = sqr(minDistancesSquare[j]);
        }
        //now choose one new data point at random as new center using a weighted 
            //  probability distribution where a point x is chosen with a probability 
            //  proportional to D(x)^2
        randomIndices[i] = randomIndexWeighted(minDistancesSquare, inputSize);
        randomDataPoint = dataPoints[randomIndices[i]];
        memcpy(prototypes[i]->coordinates, randomDataPoint->coordinates, DIMENSION*sizeof(double));
        myMemsetDouble(minDistancesSquare, 10, inputSize);
    
    }
    
    //free used variables
    free(minDistancesSquare);
    free(randomIndices);
    
    //assign values to struct to be returned
    newVQ->prototypes = prototypes;
    newVQ->amountClusters = clusters; 
    newVQ->learningRate = LEARNINGRATEINITIAL;
    return newVQ;
    
}



void trainVQllodys(VQ *vq, Vector **dataPoints, int inputSize){
    ////////////////////////////////////////////////    
    if(PRINTBOOL){
    printf("\n");
    printf("Start training: lloyds algorithm\n");
    }
    ///////////////////////////////////////////////
    
    time_t startTime = time(NULL);
        
    double distance = 100;
    double totalMovedDistance = 100; 
    double minDistance = 100;
    int *clusterCounter = calloc(vq->amountClusters, sizeof(int));
    int epoche = 0; 
    
    Vector **newCenters = calloc(vq->amountClusters, sizeof(Vector*));
    for(int i = 0; i < vq->amountClusters; i++){
        newCenters[i] = createVector(DIMENSION);
    }
    
    
    
    while(time(NULL) < startTime + TIMEOUT  && totalMovedDistance >= LLOYDSTOPPINGTHRESHOLD){
        totalMovedDistance = 0; 
        epoche++;
        //for all dataPoints find the closest center
        //  then update the center by just taking the average of points assigned to that cluster
        for(int i = 0; i < inputSize; i++){
            for(int j = 0; j < vq->amountClusters; j++){
                distance = vectorDistance(dataPoints[i], vq->prototypes[j]);
                if(distance < minDistance){
                    minDistance = distance;
                    dataPoints[i]->cluster = j;
                }
                distance = 100;
            }
            minDistance = 100;
        }
 
        for(int i = 0; i < inputSize; i++){
            clusterCounter[dataPoints[i]->cluster] += 1; //count how many points assigned to clusters
        }
        
        for(int i = 0; i < inputSize; i++){            
            addToVector(newCenters[dataPoints[i]->cluster], dataPoints[i]); //add points to newCenters
        }
        
        
        for(int i = 0; i < vq->amountClusters; i++){
            if(clusterCounter[i] <= 0) continue;
            multToVector(newCenters[i], (double)1/( (double)clusterCounter[i] ));
            distance = vectorDistance(vq->prototypes[i], newCenters[i]);
            totalMovedDistance += distance; 
            
            //////////////////////////////////////////////////////////////////
            if(PRINTBOOL){
                printf("epoche %d, moved center i: %d, %f\n", epoche, i, distance);        
            }
            //////////////////////////////////////////////////////////////////
        }
        
        Vector **tmp = vq->prototypes;
        vq->prototypes = newCenters;
        newCenters = tmp;
        
        for(int i = 0; i < vq->amountClusters; i++){
            myMemsetDouble(newCenters[i]->coordinates, 0, DIMENSION);
        }
        
        memset(clusterCounter, 0, vq->amountClusters*sizeof(int));
    }
    
    ////////////////////////////////////////////////////////
    if(PRINTBOOL){
        printf("\n----------------------------------\n");
        printf("Finished training in epoche %d\n", epoche);
        printf("----------------------------------\n\n");
    }
    ////////////////////////////////////////////////////////
    
    free(clusterCounter);
    for(int i = 0; i < vq->amountClusters; i++){
        freeVector(newCenters[i]);
    }
    free(newCenters);
    
    
}



void trainVQ(VQ *vq, Vector **dataPoints, int inputSize){
    //start timer in train-function
    time_t startTime = time(NULL);
    
    int randomIndex = 0;
    double updateSize = 0;
    int epoche = 0;
    
    while(time(NULL) < startTime + TIMEOUT && epoche < NUMITER){  
        randomIndex = randomInt(0, inputSize);
        //update winning prototype and learning Rate
        updateSize = updateWinningPrototype(vq, dataPoints[randomIndex]);
        vq->learningRate = LEARNINGRATEINITIAL* exp(-((double)epoche/(double)NUMITER*5));
        
        ///////////////////////////////////////////////////////////////////////
        if(PRINTBOOL){
            printf("epoche %d: updateSize is %lf, learning rate is: %lf \n", epoche, updateSize, vq->learningRate);
        }
        //////////////////////////////////////////////////////////////////////////
        
        epoche++;
        if(epoche > NUMITER/10 && epoche % 50 == 0){//check after some steps if an prototype is empty or is a duplicate
            //deleteEmptyPrototypes(vq, dataPoints, inputSize);
            deleteDuplicatePrototypes(vq);
        }
        
        
    }
}

/*
 * Main program
 */

int main(int argc, char** argv) { 
    
    srand((unsigned) time(NULL));
    
    
    int clusters = readAmountClusters();
    int inputSize = 0;
    double **input = createDoubleMatrix(MAXINPUTLENGTH, DIMENSION);
    double *scalingFactors = calloc(DIMENSION, sizeof(double));
    
    readInput(&inputSize, input, scalingFactors);
    
    
    Vector **dataPoints = calloc(inputSize, sizeof(Vector *));
    for(int i = 0; i < inputSize; i++){
        dataPoints[i] = createVector(DIMENSION);
        for(int j = 0; j < DIMENSION; j++){
            dataPoints[i]->coordinates[j] = input[i][j];
        }
    }
    freeDoubleMatrix(input, MAXINPUTLENGTH); //free input after having assigned values to vectors
    
    
    //VQ *vq = createVQ(4 *clusters);
    
    
    VQ* vq = kMeansPlusPlus(dataPoints, inputSize, clusters);
    //trainVQ(vq, dataPoints, inputSize);
    trainVQllodys(vq, dataPoints, inputSize);
    printVQ(vq, scalingFactors);
    
    if(PRINTBOOL){
        sleep(1);
    }
    
    free(scalingFactors);
    
    //free vectors
    for(int i = 0; i < inputSize; i++){
        freeVector(dataPoints[i]);
    }
    free(dataPoints);
    
    return (EXIT_SUCCESS);
}

