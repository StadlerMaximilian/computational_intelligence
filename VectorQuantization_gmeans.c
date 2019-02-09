/*
 * Vector Quantization
 * 
 * implements a VQ-Algorithm that is able to detect the amount of clusters and
 * find according centers by using the gmeans algorithm. 
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

#define MIN(x,y) ((x)<(y) ? (x) : (y))
#define MAX(x,y) ((x)>(y) ? (x) : (y))
#define ABS(x) ( (x) > 0 ? (x) : -(x))

#define sqr(x) ( (x) * (x) )

#define PI 3.14159265358979323846

/////////////////////////////////////////////////
/////  DEFINE NETWORK PARAMETERS HERE   /////////
/////////////////////////////////////////////////

#define MAXINPUTLENGTH 1000
#define DIMENSION 2


#define PRINTBOOL FALSE


#define TIMEOUT 1000//timeout for training


#define SIGNIFICANCELEVEL 0.0001
#define NONCRITICALVALUE 1.8692


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
    return ( rand() % (to - from + 1)) + from;  
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


void swap(double *a, double *b){
    double tmp = *a;
    *a = *b;
    *b = tmp;
}

int bubbleSort(double* list, int list_len){
    char swapped = TRUE;
    int i = 0; //index in for loop
    int len = list_len;
    int count = 0;
    
    if(len == 1){
        return(EXIT_SUCCESS); //already sorted
    }
    else if(len < 1){
        return(-1); //error
    }
    else{
        do{
            swapped = FALSE;
            for(i = 0; i < len-1; i++){
                if(list[i] > list[i+1]){
                    swap(list + i, list + i + 1);
                    swapped = TRUE;
                    count++;
                }
            }
            len--; //reduce length, biggest element already at the end
            if(swapped){
            }
        } while(swapped);
        return(EXIT_SUCCESS);
    }
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

Vector *createVectorCopy(Vector *toCopy){
    if(toCopy == NULL){
        printf("ERROR: null-pointer-exception in in createVectorCopy\n");
        return NULL;
    }
    
    Vector *vector = calloc(1, sizeof(Vector));
    vector->cluster = toCopy->cluster;
    vector->dimension = toCopy->dimension;
    vector->coordinates = calloc(vector->dimension, sizeof(double));
    memcpy(vector->coordinates, toCopy->coordinates, sizeof(double)*vector->dimension);
    
    return vector;
}


void freeVector(Vector *vector){
    if(vector == NULL){
        printf("ERROR: null-pointer-exception in freeVector\n");
        return;
    }
    
    free(vector->coordinates);
    free(vector);
}

Vector *addVectors(Vector *result, Vector *a, Vector *b){
    if(a == NULL || b == NULL){
        printf("ERROR: null-pointer-exception in addVectors\n");
        return NULL;
    }
    
    if(a->dimension != b->dimension){
        printf("ERROR: Dimension do not agree!\n");
        return NULL;
    }
    
    int dimension = a->dimension;
    for(int i = 0; i < dimension; i++){
        result->coordinates[i] = a->coordinates[i] + b->coordinates[i];
    }
    
    return result;
}

Vector *addToVector(Vector* target, Vector* added){
    if(target == NULL || added == NULL){
        printf("ERROR: null-pointer-exception in addToVector\n");
        return NULL;
    }
    
    if(target->dimension != added->dimension){
        printf("ERROR: Dimension do not agree.!\n");
        return NULL;
    }
    
    for(int i = 0; i < target->dimension; i++){
        target->coordinates[i] += added->coordinates[i];
    }
    
    return target;
}



Vector *subVectors(Vector *result, Vector *a, Vector *b){
    if(a == NULL || b == NULL){
        printf("ERROR: null-pointer-exception in subVectors\n");
        return NULL;
    }
    
    if(a->dimension != b->dimension){
        printf("ERROR: Dimension do not agree.!\n");
        return NULL;
    }
    
    int dimension = a->dimension;
    for(int i = 0; i < dimension; i++){
        result->coordinates[i] = a->coordinates[i] - b->coordinates[i];
    }
    
    return result;
}

Vector *multVectors(Vector *result, Vector *v, double scalar){    
    if(v == NULL){
        printf("ERROR: null-pointer-exception in multVectors\n");
        return NULL;
    }
    
    int dimension = v->dimension;
    for(int i = 0; i < dimension; i++){
        result->coordinates[i] = v->coordinates[i] * scalar;
    }
    
    return result;
    
}

Vector *multToVector(Vector *target, double scalar){
    if(target == NULL){
        printf("ERROR: null-pointer-exception in multToVector\n");
        return NULL;
    }
    
    for(int i = 0; i < target->dimension; i++){
        target->coordinates[i] *= scalar;
    }
    
    return target;
}

double vectorLength(Vector *v){
    if(v == NULL){
        printf("ERROR: null-pointer-exception in vectorLength\n");
        return 0.0;
    }
    
    double len = 0;
    
    for(int i = 0; i < v->dimension; i++){
        len = len + sqr( v->coordinates[i] );
    }
    
    len = sqrt(len);
    
    return len;
    
}

double vectorPointDistance(Vector *p1, Vector *p2){
    if(p1 == NULL || p2 == NULL){
        printf("ERROR: null-pointer-exception in vectorPointDistance\n");
        return 0.0;
    }
    
    if(p1->dimension != p2->dimension){
        printf("ERROR: Dimension do not agree.!\n");
        return 0.0;
    }
       
    
    Vector *dif = createVector(DIMENSION);
    subVectors(dif, p1, p2);
    double len = vectorLength(dif);
    freeVector(dif);
    return len;    
}


double vectorScalarProduct(Vector *u, Vector *v){
    if(v->dimension != u->dimension){
        printf("ERROR: dimensions do not agree!\n");
        return 0.0;
    }
    
    double result = 0.0;
    
    for(int i = 0; i < v->dimension; i++){
        result += u->coordinates[i] * v->coordinates[i];
    }
    
    return result;
}


double projectVectorUontoV(Vector *u, Vector *v){
    return vectorScalarProduct(u, v) / vectorScalarProduct(v, v);
}


void printVector(Vector *v){
    int dim = v->dimension;
    printf("\nVector: \n");
    printf("\t dimension: %d\n", dim);
    printf("\t cluster: %d\n", v->cluster);
    printf("\t coordinates: ");
    for(int i = 0; i < dim; i++){
        if(i == 0)
            printf("%lf", v->coordinates[i]);
        else
            printf(" ,%lf", v->coordinates[i]);
    }
    printf("\n");
}

double normalizeVector(Vector *v){
    double norm = vectorLength(v);
    for(int i = 0; i < v->dimension; i++){
        v->coordinates[i] = v->coordinates[i] / norm;
    }
    return norm;
}

/*
 * Vector List
 *  simple list implementation
 *  contains pointers to vectors
 *  Methods
 *      addVector
 *      clearList
 *      createList
 *      freeList
 */

typedef struct vectorListstruct{
    int size;
    int vectorDimension;
    Vector **vectors;
} VectorList;


VectorList *createVectorList(int size, int vectorDim, int allocateVectorBOOL){
    VectorList *vl = calloc(1, sizeof(VectorList));
    vl->size = size; //empty
    vl->vectorDimension = vectorDim;
    vl->vectors = calloc(MAXINPUTLENGTH, sizeof(Vector *)); //make space for maxInput vectors
    
    if(allocateVectorBOOL){
        for(int i = 0; i < size; i++){
            vl->vectors[i] = createVector(vectorDim);
        }
    }
    
    
    return vl;
}

void clearList(VectorList *list, int freeVectorsBOOL){  
    if(freeVectorsBOOL){
        for(int i = 0; i < list->size; i++){
            freeVector(list->vectors[i]);
        }
    }
    else{
        for(int i = 0; i < list->size; i++){
            list->vectors[i] = NULL;
        }
    }
    list->size = 0;
}

void freeList(VectorList *list, int freeVectorBOOL){
    if(freeVectorBOOL){
        for(int i = 0; i < list->size; i++){
            freeVector(list->vectors[i]);
        }
    }
    free(list->vectors);
    free(list);
}

void addVectorToList(VectorList *list, Vector *vector, int copyBOOL){
    if(list->vectorDimension != vector->dimension){
        printf("ERROR: dimensions do not agree!\n");
        return;
    }
    if(copyBOOL){
        list->vectors[list->size] = createVectorCopy(vector);
    }
    else{
        list->vectors[list->size] = vector;
    }
    list->size += 1;
}


void replaceVectorinList(VectorList *list, int replaceIndex, Vector *replacementVector, int freeVectorBOOL){
    if(freeVectorBOOL){    
        Vector *tmp = list->vectors[replaceIndex];
        list->vectors[replaceIndex] = replacementVector;
        free(tmp);    
    }
    else{
        list->vectors[replaceIndex] = replacementVector; 
    }
}



void deleteVectorInList(VectorList *list, int deleteIndex, int freeVectorBOOL){
    if(freeVectorBOOL){
        freeVector(list->vectors[deleteIndex]);
    }
    
    for(int i = deleteIndex; i < list->size - 1; i++){
        list->vectors[i] = list->vectors[i+1];
    }
    
    list->size -= 1;
    

    
}


/*
 *  Matrix Functions
 *
 */

typedef struct matrixStruct{
    int rows;
    int columns;
    double **matrixEntry;
}Matrix;

Matrix *createMatrix(int rows, int columns){
    Matrix *matrix = calloc(1, sizeof(Matrix));
    matrix->rows = rows;
    matrix->columns = columns;
    
    
    double **mat = calloc(rows, sizeof(double *));
    for(int i = 0; i < rows; i++){
        mat[i] = calloc(columns, sizeof(double));
    }
    
    matrix->matrixEntry = mat;
    
    return matrix;
}

void freeMatrix(Matrix *matrix){
    for(int i = 0; i < matrix->rows; i++ ){
        free(matrix->matrixEntry[i]);
    }
    free(matrix->matrixEntry);
    free(matrix);
}

void printMatrix(Matrix *matrix, int descriptionBool){
    if(descriptionBool){
        printf("print Matrix: \n");
        printf("%d x %d\n", matrix->rows, matrix->columns);
        printf("Entries: \n");
    }
    
    for(int row = 0; row < matrix->rows; row++){
        printf("\t");
        for(int col = 0; col < matrix->columns; col++){
            printf("%lf\t", matrix->matrixEntry[row][col]);
        }
        printf("\n");
    }
}



Matrix *createDataMatrixFromList(VectorList *list){
    /* 
    * DATA-MATRIX X
    *      in rows: data points
    *      in columns: coordinates such that mean of columns is 0
    *      input is VectorList of dataPoints a matrix should be created of
    */
    Matrix *dataMatrix = createMatrix(list->size, list->vectorDimension);
    
    //fill in matrix
    for(int i = 0; i < dataMatrix->rows; i++){
        for(int j = 0; j < dataMatrix->columns; j++){
            dataMatrix->matrixEntry[i][j] = list->vectors[i]->coordinates[j];
        }
    }
    
    double mean = 0.0;
    //shift columns such that mean is zero 0
    for(int col = 0; col < dataMatrix->columns; col++){
        //for each columns calculate mean and shift by that value
        for(int row = 0; row < dataMatrix->rows; row++){
            mean += dataMatrix->matrixEntry[row][col];
        }
        mean = mean / (double)dataMatrix->rows;
        for(int row = 0; row < dataMatrix->rows; row++){
            dataMatrix->matrixEntry[row][col] -= mean;
        }
        
    }
    
    return dataMatrix;
}


Matrix *createDataMatrixCovariance(Matrix *dataMatrix){
    //returns dataMatrix^T * dataMatrix
    int vectorDimension = dataMatrix->columns;
    Matrix *covariance = createMatrix(vectorDimension, vectorDimension);
    
    for(int i = 0; i < vectorDimension; i++){
        for(int j = 0; j < vectorDimension; j++){
            for(int k = 0; k < dataMatrix->rows; k++){
                covariance->matrixEntry[i][j] += dataMatrix->matrixEntry[k][i] * dataMatrix->matrixEntry[k][j];
            }
        }
    }
    
    for(int i = 0; i < vectorDimension; i++){
        for(int j = 0; j < vectorDimension; j++){
            covariance->matrixEntry[i][j] = covariance->matrixEntry[i][j] / (double)(dataMatrix->rows - 1);
        }
    }
    
    
    return covariance;
}

Vector *multVectorMatrix(Matrix *matrix, Vector *vector){
    if(matrix->columns != vector->dimension){
        printf("ERROR, dimension do not agree!\n");
        return NULL;
    }
    
    Vector *result = createVector(matrix->rows);
    
    for(int row = 0; row < matrix->rows; row++){
        for(int col = 0; col < matrix->columns; col++){
            result->coordinates[row] += matrix->matrixEntry[row][col] * vector->coordinates[col];
        }
    }
    
    return result;
}

Vector *powerMethod(Matrix *matrix){
    
    //only works for quadratic matrices
    if(matrix->columns != matrix->rows){
        printf("ERROR, matrix not quadratic\n");
        return NULL;
    }
    
    int dim = matrix->columns;
    
    //initialize vector randomly
    Vector *b_k = createVector(dim);    
    Vector *dif = NULL;
    Vector *b_k1 = NULL;
    Vector *tmp = NULL;
    
    for(int i = 0; i < dim; i++){
        b_k->coordinates[i] = randomDouble(-1,1);
    }
    normalizeVector(b_k);
    
    
    double norm = 1000;
    
    while(norm > 1e-7){
        dif = createVector(dim);
        
        b_k1 = multVectorMatrix(matrix, b_k);
        normalizeVector(b_k1);
        
        subVectors(dif, b_k, b_k1);
        norm = vectorLength(dif);
        freeVector(dif);
        
        tmp = b_k;
        b_k = b_k1;
        freeVector(tmp);
    }
    
    return b_k;
}


double matrixQuadraticForm(Matrix *m, Vector *v){
    if(m->columns != m->rows){
        printf("ERROR, matrix not quadratic!\n");
        return 0.0;
    }
    if(m->columns != v->dimension){
        printf("ERROR, dimensions do not agree!\n");
        return 0.0;
    }
   
    
    Vector *inter = multVectorMatrix(m, v);
    
    double result = vectorScalarProduct(v, inter);
    freeVector(inter);
    
    return result;   
}

double rayleighQuotient(Matrix *matrix, Vector *vector){
    if(matrix->columns != matrix->rows){
        printf("ERROR, matrix not quadratic!\n");
        return 0.0;
    }
    if(matrix->columns != vector->dimension){
        printf("ERROR, dimensions do not agree!\n");
        return 0.0;
    }
    
    return matrixQuadraticForm(matrix, vector) / vectorScalarProduct(vector, vector);
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
void normalizeInput(Matrix *input, int inputLength, double *retMax, int dimension){
    if(input == NULL || retMax == NULL){
        printf("NULL-pointer-exception!\n");
        return;
    }    
    
    
    double *max = calloc(dimension, sizeof(double));
    
    for(int i = 0; i < dimension; i++){
        max[i] = input->matrixEntry[0][i];
        //find maximal input per input channel
        for(int j = 0; j < inputLength;j++){
            if( ABS(input->matrixEntry[j][i]) > max[i]){
                max[i] = ABS(input->matrixEntry[j][i]);
            }
        }
        //divide all inputs per channel with that maximum
        for(int j = 0; j < inputLength; j++){
            input->matrixEntry[j][i] = input->matrixEntry[j][i] / max[i];
        }
    }   
    for(int i = 0; i < dimension; i++){
        retMax[i] = max[i]; 
    }
    free(max);
}


void readInput(int *inputSize, Matrix *input, double *scalingFactor){
    int lengthIndex = 0;
    double x = 0;
    double y = 0;
    
    while( scanf("%lf, %lf\n", &x, &y) != EOF){
        input->matrixEntry[lengthIndex][0] = x;
        input->matrixEntry[lengthIndex][1] = y;
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
    double alpha;
    double nonCriticalValue;
    VectorList *centers;    
} VQ;

VQ *createVQ(int initialClusters, double alpha, double nonCriticalValue){
    VQ *newVQ = calloc(1, sizeof(VQ));
    
    VectorList *newList = createVectorList(0, DIMENSION, FALSE);
    
    for(int i = 0; i < initialClusters; i++){
        addVectorToList(newList, createVector(DIMENSION), FALSE);
        for(int j = 0; j < DIMENSION; j++){
            newList->vectors[i]->coordinates[j] = randomDouble(-1,1); //since normalized input to -10 - 10
        }
    }
    
    newVQ->centers = newList;
    newVQ->amountClusters = initialClusters; 
    newVQ->alpha = alpha;
    newVQ->nonCriticalValue = nonCriticalValue;
    
    return newVQ;
}

void freeVQ(VQ *vq){
    freeList(vq->centers, TRUE);
    free(vq);
}

void addCentertoVQ(VQ *vq, Vector *v){
    vq->amountClusters += 1;
    addVectorToList(vq->centers, v, FALSE);
}

void printVQ(VQ *vq, double *scalingFactors){
    if(PRINTBOOL){
        printf("amountClusters: %d\n", vq->amountClusters);
        printf("alpha: %f\n", vq->alpha);
    }
    for(int i = 0; i < vq->amountClusters; i++){
        printf("%.4f", vq->centers->vectors[i]->coordinates[0]*scalingFactors[0]);
        for(int j = 1; j < DIMENSION; j++){
            printf(", %.4f", vq->centers->vectors[i]->coordinates[j]*scalingFactors[j]);
        }
        printf("\n");
    }
    
    
}



VQ *kMeansPlusPlus(Vector **dataPoints, int inputSize, int clusters, double alpha, double nonCriticalValue){
    //implement k-means++ algorithm to determine the initial location of clusters
    
    
    //set up empty VQ
    VQ *newVQ = createVQ(clusters, alpha, nonCriticalValue);
    Vector **prototypes = calloc(clusters, sizeof(Vector *));
    for(int i = 0; i < clusters; i++){
        prototypes[i] = createVector(DIMENSION);
    }

    
    
    //choose 1st center randomly from data points and copy 
    //  its coordinates into the 0th prototype
    int *randomIndices = calloc(clusters, sizeof(int));
    memset(randomIndices, -1, clusters*sizeof(int));
    randomIndices[0] = randomInt(0, inputSize - 1);
    Vector *randomDataPoint = dataPoints[randomIndices[0]];
    memcpy(prototypes[0]->coordinates, randomDataPoint->coordinates, DIMENSION*sizeof(double));

    //now find for each data point the distance between that point and the nearest 
    //  chosen center
    double *minDistancesSquare = calloc(inputSize, sizeof(double));
    myMemsetDouble(minDistancesSquare, 10, inputSize);
    double distance = 1000;
    
    for(int i = 0; i < clusters; i++){
        //loop through the already assigned clusters (increases every loop)
        for(int j = 0; j < i; j++){
            //now loop through all the dataPoints and find minDistances
            for(int k = 0; k < inputSize; k++){
                distance = vectorPointDistance(prototypes[j], dataPoints[k]);
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
    
    
    clearList(newVQ->centers, TRUE);
    
    //assign values to struct to be returned
    for(int i = 0; i < clusters; i++){
        addVectorToList(newVQ->centers, prototypes[i], FALSE);
    }
    newVQ->amountClusters = clusters; 
    return newVQ;
    
}


int clusterIndexMappingFunction(int *clusterIndexMapping, int size, int cluster){
    for(int i = 0; i < size; i++){
        if(clusterIndexMapping[i] == cluster){
            return i;
        }
    }
    
    return 0;
}

void kmeans_lloyds(Vector **dataPoints,  int inputSize, Vector **centers, int amountCenters, time_t endTime){
    ////////////////////////////////////////////////    
    if(PRINTBOOL){
    printf("\n");
    printf("Start training: lloyds algorithm\n");
    }
    ///////////////////////////////////////////////
            
    double distance = 100;
    double totalMovedDistance = 100; 
    double minDistance = 100;
    int *clusterCounter = calloc(amountCenters, sizeof(int));
    int *clusterIndexMapping = calloc(amountCenters, sizeof(int));
    int epoche = 0; 
    Vector *tmp = NULL;
    
    Vector **newCenters = calloc(amountCenters, sizeof(Vector*));
    for(int i = 0; i < amountCenters; i++){
        newCenters[i] = createVector(DIMENSION);
        newCenters[i]->cluster = centers[i]->cluster;
        clusterIndexMapping[i] = centers[i]->cluster;
    }
    
    
    
    while(time(NULL) < endTime  && totalMovedDistance >= LLOYDSTOPPINGTHRESHOLD){
        totalMovedDistance = 0; 
        epoche++;
        //for all dataPoints find the closest center
        //  then update the center by just taking the average of points assigned to that cluster
        for(int i = 0; i < inputSize; i++){
            for(int j = 0; j < amountCenters; j++){
                distance = vectorPointDistance(dataPoints[i], centers[j]);
                if(distance < minDistance){
                    minDistance = distance;
                    dataPoints[i]->cluster = centers[j]->cluster;
                }
                distance = 100;
            }
            minDistance = 100;
        }
 
        for(int i = 0; i < inputSize; i++){
            clusterCounter[ clusterIndexMappingFunction(clusterIndexMapping, amountCenters, dataPoints[i]->cluster) ] += 1; //count how many points assigned to clusters
        }
        
        for(int i = 0; i < inputSize; i++){            
            addToVector(newCenters[ clusterIndexMappingFunction(clusterIndexMapping, amountCenters, dataPoints[i]->cluster) ], dataPoints[i]); //add points to newCenters
        }
        
        
        for(int i = 0; i < amountCenters; i++){
            if(clusterCounter[i] <= 0) continue;
            multToVector(newCenters[i], (double)1/( (double)clusterCounter[i] ));
            distance = vectorPointDistance(centers[i], newCenters[i]);
            totalMovedDistance += distance; 
            
            //////////////////////////////////////////////////////////////////
            if(PRINTBOOL){
                printf("epoche %d, moved center: %d, %f\n", epoche, i, distance);        
            }
            //////////////////////////////////////////////////////////////////
        }
        
                
        
        for(int i = 0; i < amountCenters; i++){
            tmp = centers[i];
            centers[i] = createVectorCopy(newCenters[i]);
            free(tmp);
            myMemsetDouble(newCenters[i]->coordinates, 0, DIMENSION);
        }
        
        memset(clusterCounter, 0, amountCenters*sizeof(int));
    }
    
    ////////////////////////////////////////////////////////
    if(PRINTBOOL){
        printf("\n----------------------------------\n");
        printf("Finished training in epoche %d\n", epoche);
        printf("----------------------------------\n\n");
    }
    ////////////////////////////////////////////////////////
    
    free(clusterCounter);
    for(int i = 0; i < amountCenters; i++){
        freeVector(newCenters[i]);
    }
    free(newCenters);
}
    
   


void kmeans_lloyds_vq(VQ *vq, Vector **dataPoints, int inputSize, time_t endTime){
    ////////////////////////////////////////////////    
    if(PRINTBOOL){
    printf("\n");
    printf("Start training: lloyds algorithm on vq\n");
    }

    kmeans_lloyds(dataPoints, inputSize, vq->centers->vectors, vq->amountClusters, endTime);
    
}


double normalCDF(double value){
    //calculates cumulative distribution function to N(0,1)
    double erf_val = erf( value/sqrt(2) );
    double ret = 0.5 * (1 + erf_val);
    return ret;
}


double a_starSquared(double *Z, int n){
    if(n == 0){
        return 0;
    }
    
    double result = 0.0;
    double nd = (double)n;
    
    for(int i = 0; i < n; i++){
        result += ( 2*i + 1 ) * ( log(Z[i]) + log(1 - Z[n - 1 - i])  );
    }
    result = result/nd;
    result = -result - nd;
    result = result * ( 1 + 4.0/nd - 25.0/( sqr(nd)  )  );
    
    return result;
    
}

    Vector *gMeansClustersplittingVector(VectorList *list){
    //caluclate covariance matrix
    Matrix *dataMatrix = createDataMatrixFromList(list);
    Matrix *covariance = createDataMatrixCovariance(dataMatrix);    
    
    //determine first principal component
    Vector *principalComponent = powerMethod(covariance);    

    
    //determine eigenvalue
    double lambda = rayleighQuotient(covariance, principalComponent);    
    
    //multiply component with sqrt( 2*lambda / PI )
    multToVector(principalComponent, sqrt( 2*lambda / PI));    
    
    freeMatrix(dataMatrix);
    freeMatrix(covariance);
    
    return principalComponent;
}


Vector **gMeansHypothesisTest(VectorList *list, Vector *center, double alpha, double nonCriticalValue, time_t endTime){
    // X = list: subset of dataPoints in d dimensions that belong to center
    //1. choose a significance level (alpha = 0.0001)  --> passed to function    
    //2. initialize 2 centers (children of center)
    Vector **ret = NULL;
    Vector *m = gMeansClustersplittingVector(list);
    Vector *c1 = createVector(DIMENSION);
    Vector *c2 = createVector(DIMENSION);
    Vector **children = calloc(2, sizeof(Vector *));    
    
    addVectors(c1, center, m); //children 1 is center + m
    subVectors(c2, center, m); //children 2 is center - m    
    
    c1->cluster = -1; //for gMeans children 1
    c2->cluster = -2; //for gMeans children 2
    
    children[0] = c1;
    children[1] = c2;
    
    if(PRINTBOOL){
        printf("---------------------\n");
        printf("Hypothesis-Test:\n\n");
        printf("children\n");
        printVector(c1);
        printVector(c2);
    }

    
    //3. run k-means on these 2 centers in X 
    //   --> c1, c2 are children chosen by k-means
    kmeans_lloyds(list->vectors, list->size, children, 2, endTime);
    
    c1 = children[0];
    c2 = children[1];
    //original children freed already in kmeans_lloyds!!

    
    
    //4. v = c1 - c2 (vector connecting children)
    //project X onto v and transform it such that mean = 0, variance = 1
    // --> special 1d representation of data
    Vector *v = createVector(DIMENSION);
    subVectors(v, c1, c2);
    
    double *projectedData = calloc(list->size, sizeof(double));
    double mean = 0.0; 
    double std = 1;
    
    for(int i = 0; i < list->size; i++ ){
        projectedData[i] = projectVectorUontoV(list->vectors[i], v);
        mean += projectedData[i];
    }
    
    mean = mean/list->size;    
    
    for(int i = 0; i < list->size; i++ ){
        projectedData[i] -= mean;
        std += sqr( projectedData[i] );
    }
    
    std = sqrt( std/list->size );
    
    for(int i = 0; i < list->size; i++){
        projectedData[i] = projectedData[i] / std;
    }
    
    
    //put projectedData in order!!!
    bubbleSort(projectedData, list->size);    
    

    
    //5. z_i = F(x'_(i)) where F is the N(0,1) cumulative distribution function
    double *Z = calloc(list->size, sizeof(double));
    
    for(int i = 0; i < list->size; i++){
        Z[i] = normalCDF(projectedData[i]);
    }
    
    
    //if A_*^2(z) is in the range of non-critical values of confindence level alpha
    // --> accept to keep original centers, discard (c1, c2) (return NULL)
    // --> otherwise keep c1, c2 instead of original center (return c1, c2)
    double Astar = a_starSquared(Z, list->size);
    if(PRINTBOOL){
        printf("Astar: %f\n", Astar);   
        printf("---------------------\n");
    }

    if( Astar < nonCriticalValue){
        freeVector(c1);
        freeVector(c2);
        free(children);        
        ret = NULL;
    }
    else{
        ret =  children; //contains c1 and c2          
    }
    
    //free used memory (m, v, Z, projectedData)
    freeVector(m);
    freeVector(v);
    free(Z);
    free(projectedData);    
    
    
    return ret;
}


void gMeansAlgorithm(VQ *vq, Vector **dataPoints, int inputSize, time_t endTime, double *scaling){
    VectorList *dataList_j = createVectorList(0, DIMENSION, FALSE);
    Vector **childrenHypothesisTest = NULL; 
    int increaseAmountCenterCounter = 1;
    
    
    //vq initialized to one random point from the data (done before)
    
    //repeat until no more centers are added   
    while(time(NULL) < endTime && increaseAmountCenterCounter > 0){
        increaseAmountCenterCounter = 0;
        
        //run kmeans(centers, dataPoints)
        kmeans_lloyds_vq(vq, dataPoints, inputSize, endTime);
        
        for(int j = 0; j < vq->amountClusters; j++){
            // X_j = {x_i | class(x_i) = j} is set of datapoints assigned to center j
            for(int d = 0; d < inputSize; d++){
                if(dataPoints[d]->cluster == j){
                    addVectorToList(dataList_j, dataPoints[d], FALSE);
                }
            }
            //use a statistical test to detect if each X_j follow an Gaussian distribution
            childrenHypothesisTest = gMeansHypothesisTest(dataList_j, vq->centers->vectors[j], vq->alpha, vq->nonCriticalValue, endTime);
            //if the data look gaussian keep center c_j, else replace it with 2 children centers
            if(childrenHypothesisTest != NULL){
                //data do not look gaussian --> replace center j with 1st children and insert 2nd in the end of list
                replaceVectorinList(vq->centers, j, childrenHypothesisTest[0], TRUE);
                vq->centers->vectors[j]->cluster = j;
                addVectorToList(vq->centers, childrenHypothesisTest[1], FALSE);
                vq->centers->vectors[vq->centers->size - 1]->cluster = vq->centers->size - 1;
                free(childrenHypothesisTest);
                increaseAmountCenterCounter++;
            }
            
            clearList(dataList_j, FALSE);

        }
              
        
        vq->amountClusters += increaseAmountCenterCounter;
    }
    
    //run kmeans(centers, dataPoints)
    kmeans_lloyds_vq(vq, dataPoints, inputSize, endTime);
}

/*
 * Main program
 */

int main(int argc, char** argv) { 
    
    time_t endTime = time(NULL) + TIMEOUT;
    
    srand((unsigned) time(NULL));
    
    
    int inputSize = 0;
    Matrix *input = createMatrix(MAXINPUTLENGTH, DIMENSION);
    double *scalingFactors = calloc(DIMENSION, sizeof(double));
    
    readInput(&inputSize, input, scalingFactors);
    
    
    Vector **dataPoints = calloc(inputSize, sizeof(Vector *));
    for(int i = 0; i < inputSize; i++){
        dataPoints[i] = createVector(DIMENSION);
        for(int j = 0; j < DIMENSION; j++){
            dataPoints[i]->coordinates[j] = input->matrixEntry[i][j];
        }
    }
    freeMatrix(input); //free input after having assigned values to vectors
    
    VQ *vq = kMeansPlusPlus(dataPoints, inputSize, 1, SIGNIFICANCELEVEL, NONCRITICALVALUE);
   
    
    gMeansAlgorithm(vq, dataPoints, inputSize, endTime, scalingFactors);
    //kmeans_lloyds_vq(vq, dataPoints, inputSize, endTime);
    
    printVQ(vq, scalingFactors);
    
    
    free(scalingFactors);
    
    //free vectors
    for(int i = 0; i < inputSize; i++){
        freeVector(dataPoints[i]);
    }
    free(dataPoints);
    
    
    return (EXIT_SUCCESS);
}

