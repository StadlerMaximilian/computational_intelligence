/*
 * SOM for Traveling-Salesman
 * 
 * implements Self-organizing-Map for finding a short route between given cities
 * 
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <unistd.h>
#include <float.h>

#define TRUE 1
#define FALSE 0

#define MIN(x,y) ((x)<(y) ? (x) : (y))
#define MAX(x,y) ((x)>(y) ? (x) : (y))
#define ABS(x) ( (x) > 0 ? (x) : -(x))

#define sqr(x) ( (x) * (x) )

#define PI 3.14159265358979323846

#define LINEAR 0
#define EXPONENTIAL 1
#define INVERSETIME 2

#define ONLINE 1
#define OFFLINE 2


/////////////////////////////////////////////////
/////  DEFINE NETWORK PARAMETERS HERE   /////////
/////////////////////////////////////////////////

#define DIMENSION 2

#define PRINTBOOL FALSE

#define TIMEOUT 298 //timeout for training

#define RATEADADAPTION EXPONENTIAL

#define BREAKCOUNTER 200

#define LEARNINGRATEINITIAL 0.5

#define NUMITER 1e5
#define MAXITER 1e8

#define TRAININGMODE ONLINE

#define NORMALIZE_DIRECTIONS_TOTALMAX TRUE

#define RATIO_CITIES_PROTOTYPES 2
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
        printf("NULL-pointer-exception in meanValue!\n");
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
    int cityID; //dataPoints only get cityID assigned
    int neighborhoodID; //SOM prototypes get cityID for representingCity (i.e. closest) and 
                        //get a fixed neighboorhoodID representing the neighborhood relations
    double *coordinates;
} Vector;

Vector *createVector(int dimension){
    Vector *vector = calloc(1, sizeof(Vector)); 
    double *coordinates = calloc(dimension, sizeof(double));
    vector->coordinates = coordinates;
    vector->dimension = dimension;
    vector->cityID = -1; //not assigned
    vector->neighborhoodID = -1; //not assigned 
    
    return vector;
}

Vector *createVectorCopy(Vector *toCopy){
    if(toCopy == NULL){
        printf("ERROR: null-pointer-exception in in createVectorCopy\n");
        return NULL;
    }
    
    Vector *vector = calloc(1, sizeof(Vector));
    vector->dimension = toCopy->dimension;
    vector->cityID = toCopy->cityID;
    vector->neighborhoodID = toCopy->neighborhoodID;
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

void addVectors(Vector *result, Vector *a, Vector *b){
    if(a == NULL || b == NULL){
        printf("ERROR: null-pointer-exception in addVectors\n");
        return;
    }
    
    if(a->dimension != b->dimension){
        printf("ERROR: Dimension do not agree!\n");
        return;
    }
    
    int dimension = a->dimension;
    for(int i = 0; i < dimension; i++){
        result->coordinates[i] = a->coordinates[i] + b->coordinates[i];
    }
}

void addToVector(Vector* target, Vector* added){
        if(target == NULL || added == NULL){
        printf("ERROR: null-pointer-exception in addToVector\n");
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



void subVectors(Vector *result, Vector *a, Vector *b){
    if(a == NULL || b == NULL){
        printf("ERROR: null-pointer-exception in subVectors\n");
        return;
    }
    
    if(a->dimension != b->dimension){
        printf("ERROR: Dimension do not agree.!\n");
        return;
    }
    
    int dimension = a->dimension;
    for(int i = 0; i < dimension; i++){
        result->coordinates[i] = a->coordinates[i] - b->coordinates[i];
    }
    
}

void multVectors(Vector *result, Vector *v, double scalar){    
    if(v == NULL){
        printf("ERROR: null-pointer-exception in multVectors\n");
        return;
    }
    
    int dimension = v->dimension;
    for(int i = 0; i < dimension; i++){
        result->coordinates[i] = v->coordinates[i] * scalar;
    }
    
}

void multToVector(Vector *target, double scalar){
    if(target == NULL){
        printf("ERROR: null-pointer-exception in multToVector\n");
        return;
    }
    
    for(int i = 0; i < target->dimension; i++){
        target->coordinates[i] *= scalar;
    }
    
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

void printVector(Vector *v){
    int dim = v->dimension;
    printf("\nVector: \n");
    printf("\t dimension: %d\n", dim);
    printf("\t cityID: %d\n", v->cityID);
    printf("\t NR: %d\n", v->neighborhoodID);
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
 * Cities Structs
 */

typedef struct citiesStruct{
    Vector **cities;
    int amountOfCities;
    double *scalingFactor;
} Cities;


Cities *createCities(void){
    Cities *ret = calloc(1, sizeof(Cities));
    ret->amountOfCities = -1; //not assigned
    
    return ret;
}


void freeCities(Cities *c){
    for(int i = 0; i < c->amountOfCities; i++){
        freeVector(c->cities[i]);
    }
    free(c->cities);
    free(c->scalingFactor);
}


void printCities(Cities *c){
    if(c == NULL){
        printf("ERROR\n");
        return;
    }
    else if(c->amountOfCities < 1){
        printf("No cities to display\n");
        return;
    }
    
    Vector *city = NULL;
    printf("-----------------------\n");
    printf("Display %d Cities:\n", c->amountOfCities);
    for(int i = 0; i < c->amountOfCities; i++){
        city = c->cities[i];
        printf("\t %d, %lf, %lf\n", city->cityID, city->coordinates[0]*c->scalingFactor[0], city->coordinates[1]*c->scalingFactor[1]);
    }
    printf("-----------------------\n");
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
void normalizeInput(Cities *cities){
    if(cities == NULL){
        printf("ERROR: NULL-pointer-exception in normalizeInput!\n");
        return;
    }    
    
    
    double *max = calloc(DIMENSION, sizeof(double));
    
    for(int i = 0; i < DIMENSION; i++){
        max[i] = cities->cities[0]->coordinates[i];
        //find maximal input per input channel
        for(int j = 0; j < cities->amountOfCities ;j++){
            if( ABS(cities->cities[j]->coordinates[i]) > max[i]){
                max[i] = ABS(cities->cities[j]->coordinates[i]);
            }
        }
    }
    
    /////////////////////////////////////// //code such that normalized with total maximum of input
    double totalMax = 0.0;
    if(NORMALIZE_DIRECTIONS_TOTALMAX){
        for(int i = 0; i < DIMENSION; i++){
            if(max[i] > totalMax){
                totalMax = max[i];
            }
        }
    }
    ///////////////////////////////////////  
    
    for(int i = 0; i < DIMENSION; i++){   
        //divide all inputs per channel totalMaximum
        
        //////////////////////
        if(NORMALIZE_DIRECTIONS_TOTALMAX){
            max[i] = totalMax; 
        }
        //////////////////////
        
        for(int j = 0; j < cities->amountOfCities; j++){
            cities->cities[j]->coordinates[i] = cities->cities[j]->coordinates[i] / max[i];
        }
    }   

    cities->scalingFactor = max;
}



void readInput(Cities *cities){
    int id = 0;
    int x = 0;
    int y = 0;
    
    int len = 0;     
    
    while( scanf("%d, %d, %d\n", &id, &x, &y) != EOF){

        if(cities->amountOfCities <= 0){
            cities->cities = calloc(1000, sizeof(Vector *)); //maximal input length
            cities->cities[0] = createVector(DIMENSION);
            cities->cities[0]->cityID = id;
            cities->cities[0]->coordinates[0] = (double)x;
            cities->cities[0]->coordinates[1] = (double)y;
            cities->amountOfCities = 1;
        }
        else{
            len = cities->amountOfCities; 
            cities->cities[len] = createVector(DIMENSION);
            cities->cities[len]->cityID = id;
            cities->cities[len]->coordinates[0] = (double)x;
            cities->cities[len]->coordinates[1] = (double)y;
            cities->amountOfCities = len + 1;
        }
    }
    
    Vector **tmp = calloc(cities->amountOfCities, sizeof(Vector *));
    memcpy(tmp, cities->cities, cities->amountOfCities*sizeof(Vector *));
    Vector **oldCities = cities->cities;
    cities->cities = tmp;
    free(oldCities);
    
    
    normalizeInput(cities);

}


/*
 * Functions for Vector Quantization
 */

typedef struct somStruct{
    int amountPrototypes;
    Vector **prototypes;
    double learningRate;
    double neighborRadius;
    double initialLearningRate;
    double tau0; //radius constant
    double lambda; //time constant exponential decay
    double cInverse; //time constant inverse time decay
    double numIter;
    int rateAdaptionType;
    double *scalingFactor;
} SOM;

SOM *createSOM(int amountCities, double *scalingFactor, Vector *middle, double radius, int numIter, int rateAdaptionType){
    
    int amountPrototypes = amountCities * RATIO_CITIES_PROTOTYPES;
    
    SOM *newSOM = calloc(1, sizeof(SOM));
    
    
    Vector **prototypes = calloc(amountPrototypes, sizeof(Vector *));
    for(int i = 0; i < amountPrototypes; i++){
        prototypes[i] = createVector(DIMENSION);
        prototypes[i]->neighborhoodID = i;
        prototypes[i]->coordinates[0] = middle->coordinates[0] + 2 * sin( 2*PI/(double)amountPrototypes * i  ); //sin and cos since input is normalized
        prototypes[i]->coordinates[1] = middle->coordinates[1] + 2 * cos( 2*PI/(double)amountPrototypes * i  ); //to range from -1 to 1
    }
    
    newSOM->prototypes = prototypes;
    newSOM->amountPrototypes = amountPrototypes; 
    newSOM->learningRate = LEARNINGRATEINITIAL;
    newSOM->initialLearningRate = LEARNINGRATEINITIAL;
    newSOM->numIter = numIter;
    newSOM->tau0 = amountCities/2;
    newSOM->lambda = numIter/log(10*newSOM->tau0);
    newSOM->cInverse = numIter/100;
    newSOM->rateAdaptionType = rateAdaptionType;
    newSOM->scalingFactor = calloc(DIMENSION, sizeof(double));
    memcpy(newSOM->scalingFactor, scalingFactor, DIMENSION*sizeof(double));
    return newSOM;
}

void freeSOM(SOM *som){
    for(int i = 0; i < som->amountPrototypes; i++){
        freeVector(som->prototypes[i]);
    }
    free(som->prototypes);
    free(som->scalingFactor);
}

void printSOM(SOM *som){
    if(PRINTBOOL){
        printf("-----------------------------------\n");
        printf("Print SOM:\n");
        printf("amountPrototypes: %d\n", som->amountPrototypes);
        printf("learningRate: %f\n", som->learningRate);
    }
    for(int i = 0; i < som->amountPrototypes; i++){
        printf("%d, %d", som->prototypes[i]->neighborhoodID, (int)(som->prototypes[i]->coordinates[0] * som->scalingFactor[0]));
        for(int j = 1; j < DIMENSION; j++){
            printf(", %d", (int)(som->prototypes[i]->coordinates[j]*som->scalingFactor[j]));
        }
        printf("\n");
     
    }
    
    if(PRINTBOOL){
        printf("-----------------------------------\n");
    }
    
}


void printSOMroute(SOM *som, Cities *cities){
    
    double minDistance = 100;
    int minIndex = 0;
    double ProtoCityDistance = 0.0;
    //assign cities to prototypes
    for(int i = 0; i < cities->amountOfCities; i++){
        minDistance = 100;
        minIndex = 0;
        for(int j = 0; j < som->amountPrototypes; j++){
            ProtoCityDistance = vectorPointDistance(som->prototypes[j], cities->cities[i]);
            if(ProtoCityDistance < minDistance){
                minDistance = ProtoCityDistance;
                minIndex = som->prototypes[j]->neighborhoodID;
            }
        }
        
        cities->cities[i]->neighborhoodID = minIndex;
        
    }    
    
    
    for(int i = 0; i < som->amountPrototypes; i++){
        for(int j = 0; j < cities->amountOfCities; j++){
            if(cities->cities[j]->neighborhoodID == som->prototypes[i]->neighborhoodID){
                printf("%d\n", cities->cities[j]->cityID);
            }
        }
        
    }
    
}


double learningRate(SOM *som, int epoche){
        
    switch(som->rateAdaptionType){
        case LINEAR:
            som->learningRate = som->initialLearningRate * (1 - epoche/som->numIter);
            break;
        case EXPONENTIAL:
            som->learningRate = som->initialLearningRate * exp(-epoche/som->lambda);
            break;
        case INVERSETIME:
            som->learningRate = som->initialLearningRate * (som->cInverse/(som->cInverse + epoche));
            break;
        default:
            som->learningRate = 0;
            break;
    }
    return som->learningRate;
}

double neighborhoodInfluence(SOM *som, int epoche, int bmuIndex, int neighborIndex){
    som->neighborRadius = som->tau0 * exp(-epoche/som->lambda);
    
    //determine distance of neurons
    double distance = abs( bmuIndex - neighborIndex );
    if(distance > som->tau0){
        distance = som->amountPrototypes - distance;
    }
    
    double influence = 0.0;
    if(som->neighborRadius < 0.00000000001){
        if(bmuIndex == neighborIndex){
            influence = 1;
        }
        else{
            influence = 0;
        }
    }
    else if(distance < som->neighborRadius){
        influence = exp(- sqr( distance ) / (2* sqr( som->neighborRadius )) );
    }
    return influence;
}


double *updatePrototypes_offline(SOM *som, Vector **dataPoints, int dataSize, int epoche){
    
    Vector **oldPoints = calloc(som->amountPrototypes, sizeof(Vector *));
    for(int i = 0; i < som->amountPrototypes; i++){
        oldPoints[i] = createVectorCopy(som->prototypes[i]);
    }
    
    //determine the best matching unit u_c(i) for each data item x_i
    double minDistance = 100;
    double distance = 0.0;
    int minIndex = -1;
    for(int i = 0; i < dataSize; i++){
        minDistance = vectorPointDistance(dataPoints[i], som->prototypes[0]);
        minIndex = 0;
        
        for(int j = 1; j < som->amountPrototypes; j++){
            distance = vectorPointDistance(dataPoints[i], som->prototypes[j]);
            if(distance < minDistance){
                minDistance = distance;
                minIndex = j;
            }
        }

        dataPoints[i]->neighborhoodID = minIndex; //for cities use neighborhoodID to represent the best matching unit
    }
    
    
    
    //update each model vector m_i to better fit the data items assigned to it
    //and the data in its neighborhood
    //m_i(t+1) = [ sum u_ic(k) * x_k ] / [sum u_ic(k], where u is neighborhood influence
    
    double influenceSum = 0;
    double singleInfluence = 0;
    double *weightedVectorSum = calloc(DIMENSION, sizeof(double));
    
    for(int i = 0; i < som->amountPrototypes; i++){
        influenceSum = 0;
        
        for(int k = 0; k < dataSize; k++){
            singleInfluence = neighborhoodInfluence(som, epoche, dataPoints[k]->neighborhoodID, i);
            influenceSum += singleInfluence;
            
            if(singleInfluence > 0){
                for(int dim = 0; dim < DIMENSION; dim++){
                    weightedVectorSum[dim] += singleInfluence*dataPoints[k]->coordinates[dim];
                }
            }
            
        }
        
        if(influenceSum > 0){
            for(int dim = 0; dim < DIMENSION; dim++){
                weightedVectorSum[dim] = weightedVectorSum[dim] / influenceSum;

                som->prototypes[i]->coordinates[dim] = weightedVectorSum[dim];
                weightedVectorSum[dim] = 0.0;
            }
        }
      
    }
    
    free(weightedVectorSum);
    
    
    double *updateDistance = calloc(som->amountPrototypes, sizeof(double));
    
    for(int i = 0; i < som->amountPrototypes; i++){
        updateDistance[i] = vectorPointDistance(som->prototypes[i], oldPoints[i]);
        freeVector(oldPoints[i]);
    }
    
    free(oldPoints);
    
    return updateDistance; 
    
}



double updatePrototypes_online(SOM *som, Vector *dataPoint, int epoche){
    double distances = 0.0;
    int minIndex = 0;
    double minDist = 1000;
    
    //calculate distances between dataPoint and cluster centers and calculate BMU
    for(int i = 0; i < som->amountPrototypes; i++){
        distances = vectorPointDistance(som->prototypes[i], dataPoint);
        if(distances < minDist){
            minDist = distances;
            minIndex = i; //minIndex contains the BMU
        }
    }
        
    Vector *update = createVector(DIMENSION);
    distances = 0.0;
    
    //update all prototypes according to neighborhood-influence 
    for(int i = 0; i < som->amountPrototypes; i++){    
        subVectors(update, dataPoint, som->prototypes[i]);
        multToVector(update, ( learningRate(som, epoche)*neighborhoodInfluence(som, epoche, minIndex,i ) ) );
        distances += vectorLength(update); //determine size of update
        addToVector(som->prototypes[i], update);
        
    }
    
    freeVector(update);
    return distances/(som->amountPrototypes);
}
   

void trainSOM_online(SOM *som, Cities *cities){
    //start timer in train-function
    time_t startTime = time(NULL);
    
    int randomIndex = 0;
    double meanUpdateSize = 0.0;
    
    int counter = 0;
    int epoche = 0;
    
    while(time(NULL) < startTime + TIMEOUT && epoche < MAXITER){  
        randomIndex = randomInt(0, cities->amountOfCities-1);
        //update winning prototype and learning Rate
        meanUpdateSize = updatePrototypes_online(som, cities->cities[randomIndex], epoche);
               
        if( meanUpdateSize < 1e-8){
            counter++;
        }
        else{
            counter = 0;
        }
        
        if( counter == BREAKCOUNTER){
            break;
        }
        
        
 
        ///////////////////////////////////////////////////////////////////////
        if(PRINTBOOL){
            printf("epoche %d: updateSize is %lf, learning rate is: %lf, neighborRadius is: %lf, counter is: %d\n", epoche, meanUpdateSize , som->learningRate, som->neighborRadius, counter);
        }
        //////////////////////////////////////////////////////////////////////////
        epoche++;
        
    }
    
    
    
}

/*
 * Main program
 */

int main(int argc, char** argv) { 
    
    srand((unsigned) time(NULL));
    
    
    Cities *cities = calloc(1, sizeof(Cities));
    readInput(cities); 
    
    
    //determine center of all cities (average in coordinates)
    Vector *coordMeans = createVector(DIMENSION);
    double radius = 0;
    double distance = 0; 
    
    for(int i = 0; i < cities->amountOfCities; i++){
        for(int j = 0; j < DIMENSION; j++){
            coordMeans->coordinates[j] += cities->cities[i]->coordinates[j];
        }

    }
    for(int j = 0; j < DIMENSION; j++){
            coordMeans->coordinates[j] = coordMeans->coordinates[j]/cities->amountOfCities;               
    }

    //determine radius to center of all cities (i.e. find maximum distance of cities)
    for(int i = 0; i < cities->amountOfCities; i++){
        distance = vectorPointDistance(coordMeans, cities->cities[i]);
        if(distance > radius){
            radius = distance;
        }
    }
    
    
    
    //initialize SOM 
    SOM *som = createSOM(cities->amountOfCities, cities->scalingFactor, coordMeans, radius, NUMITER, RATEADADAPTION);
    freeVector(coordMeans);
    
        
    trainSOM_online(som, cities);
           
    printSOMroute(som, cities);
    
    

    //free all used memory
    freeCities(cities);
    freeSOM(som);
    
    return (EXIT_SUCCESS);
}

