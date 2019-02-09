/*
 * Hoppfields Network
 * 
 * implements Hoppfields Network, that is able to read small binary images (e.g. black
 * and white, in this task images consisting of * and .) learn them 
 * and handle small distortions of images. 
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

#define MIN(x,y) ((x)<(y) ? (x) : (y))
#define MAX(x,y) ((x)>(y) ? (x) : (y))
#define ABS(x) ( (x) > 0 ? (x) : -(x))

#define sqr(x) ( (x) * (x) )


/////////////////////////////////////////////////
/////  DEFINE NETWORK PARAMETERS HERE   /////////
/////////////////////////////////////////////////

#define NUMPIXELX 20
#define NUMPIXELY 10


#define PRINTBOOL TRUE


#define TIMEOUT 100 //timeout for training
#define RANDOMFLIPS 10
#define CONVERGENCE 5
#define FLIPBOOL FALSE
#define SELFLINK TRUE

#define HIGHVALUE '*' //42
#define LOWVALUE '.' //46
#define TRAINING 1
#define TESTING 2
#define SWITCHMODE -2
#define IMAEOF -3
#define IMASWITCHMODE -4
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


int THRESHOLD(int i){
    if(i >= 0){
        return 1;
    }
    else if(i < 0){
        return -1;
    }
    return 1;
}

/*
 * Vector Functions
 */
typedef struct imageVectorStruct{
    int numPixelsX;
    int numPixelsY;
    int size; 
    int *colorValues;
} ImageVector;

ImageVector *createImageVector(int x, int y){
    int size = x*y;
    if(size <= 0){
        printf("ERORR: Image has invalid size!\n");
    }
    
    ImageVector *vector = calloc(1, sizeof(ImageVector)); 
    int *colorValues = calloc(size, sizeof(int));
    vector->colorValues = colorValues;
    vector->numPixelsX = x;
    vector->numPixelsY = y;
    vector->size = size;
    
    return vector;
}

void freeVector(ImageVector *vector){
    if(vector == NULL){
        printf("ERROR: null-pointer-exception\n");
        return;
    }
    
    free(vector->colorValues);
    free(vector);
}

ImageVector* addVectors(ImageVector *a, ImageVector *b){
    if(a == NULL || b == NULL){
        printf("ERROR: null-pointer-exception\n");
        return NULL;
    }
    
    if(a->size != b->size){
        printf("ERROR: Sizes do not agree.!\n");
        return NULL;
    }
    
    int size = a->size;
    ImageVector *result = createImageVector(a->numPixelsX, a->numPixelsY);
    for(int i = 0; i < size; i++){
        result->colorValues[i] = a->colorValues[i] + b->colorValues[i];
    }
    
    return result;
}

void addToVector(ImageVector* target, ImageVector* added){
        if(target == NULL || added == NULL){
        printf("ERROR: null-pointer-exception\n");
        return;
    }
    
    if(target->size != added->size){
        printf("ERROR: Dimension do not agree.!\n");
        return;
    }
    
    for(int i = 0; i < target->size; i++){
        target->colorValues[i] += added->colorValues[i];
    }
    
}

char convertBinaryToSymbol(int s){
    if(s == 1){
        return HIGHVALUE;
    }
    else if(s == -1){
        return LOWVALUE;
    }
    else{
        return 'E';
    }
}

void printVector(ImageVector *v, char descriptionBool){
    int size = v->size;
    int x = v->numPixelsX;
    int y = v->numPixelsY;
    
    if(descriptionBool == TRUE){
        printf("\nImage: \n");
        printf("size: %d pixels (%d x %d)\n", size, x, y);
    }
    
    for(int i = 0; i < size; i++){
        if(i != 0 && i % x == 0){
            printf("\n");
        }
        printf("%c", convertBinaryToSymbol(v->colorValues[i]));
    }
    printf("\n");
}

void printVectorsNextTo(ImageVector *left, ImageVector *right){
    if(left == NULL || right == NULL){
        return;
    }
    if(left->numPixelsX != right->numPixelsX || left->numPixelsY != right->numPixelsY){
        return; 
    }
    
    int x = left->numPixelsX;
    int y = left->numPixelsY;
    
    for(int yp = 0; yp < y; yp++){
        for(int xp = 0; xp < x; xp++ ){
            printf("%c", convertBinaryToSymbol(left->colorValues[xp + x*yp]));
        }
        if(yp == y/2){
            printf("\t-->\t");
        }
        else{
            printf("\t\t");
        }
        for(int xp = 0; xp < x; xp++ ){
            printf("%c", convertBinaryToSymbol(right->colorValues[xp + x*yp]));
        }
        printf("\n");
    }
}

void applyThreshold(ImageVector *v){
    for(int i = 0; i < v->size; i++){
        v->colorValues[i] = THRESHOLD(v->colorValues[i]);
    }
}

void copyVectors(ImageVector *target, ImageVector *source){
    target->numPixelsX = source->numPixelsX;
    target->numPixelsY = source->numPixelsY;
    target->size = source->size;
            
    for(int i = 0; i < target->size; i++){
        target->colorValues[i] = source->colorValues[i];
    }
    
}

/*
 * Image container
 */
typedef struct imagesStruct{
    int numPixelX;
    int numPixelY;
    ImageVector **trainingImages;
    int trainingSize;
    ImageVector **testingImages;
    ImageVector **testingImagesBackup;
    int testingSize;
    ImageVector **output;
    int outputSize;
} Images;

Images *createImages(int x, int y){
    Images *ret = calloc(1, sizeof(Images));
    ret->numPixelX = x;
    ret->numPixelY = y;
    
    return ret;
}

void freeImages(Images *images){
    for(int i = 0; i < images->trainingSize; i++){
        freeVector(images->trainingImages[i]);
    }
    for(int i = 0; i < images->testingSize; i++){
        freeVector(images->testingImages[i]);
    }
    for(int i = 0; i < images->outputSize; i++){
        freeVector(images->output[i]);
    }
    
    free(images->trainingImages);
    free(images->testingImages);
    free(images->output);
    free(images);
}

void createImagesOutput(Images *images){
    if(images->testingSize == 0){
        return;
    }
    images->outputSize = images->testingSize;
    
    images->output = calloc(images->outputSize, sizeof(ImageVector *));

    for(int i = 0; i < images->outputSize; i++){
        images->output[i] = createImageVector(images->numPixelX, images->numPixelY);

    }
    
    if(PRINTBOOL){
        images->testingImagesBackup = calloc(images->outputSize, sizeof(ImageVector *));
        for(int i = 0; i < images->testingSize; i++){
            images->testingImagesBackup[i] = createImageVector(images->numPixelX, images->numPixelY);
            copyVectors(images->testingImagesBackup[i], images->testingImages[i]);
        }
    }
    
    
}

int compareImageVector(ImageVector *a, ImageVector *b){
    if(a == NULL || b == NULL){
        return -1;
    }
    else if(a->numPixelsX != b->numPixelsX || a->numPixelsY != b->numPixelsY){
        return -1;
    }
    int counter = 0;
    int size = a->size;
    for(int i = 0; i < size; i++){
        if(a->colorValues[i] == b->colorValues[i]){
            counter++;
        }
    }
    
    return counter;
    
}

/*
 * Hopfield Functions
 */
typedef struct hopfieldMapStruct{
    int imageVectorSize;
    int **map;
} HopfieldMap;


HopfieldMap *createHopfieldMap(int imageVectorSize){
    
    HopfieldMap *map = calloc(1, sizeof(HopfieldMap));    
    
    int size = imageVectorSize;
    int **hmap = calloc(size, sizeof(int *));
    for(int i = 0; i < size; i++){
        hmap[i] = calloc(size, sizeof(int));
    }
    
    map->imageVectorSize = size;
    map->map = hmap;
    
    return map;
}

void freeHopfieldMap(HopfieldMap *map){
    for(int i = 0; i < map->imageVectorSize; i++ ){
        free(map->map[i]);
    }
    free(map);
}

void applyThresholdMap(HopfieldMap *map){
    for(int i = 0; i < map->imageVectorSize; i++){
        for(int j = 0; j < map->imageVectorSize; j++){
            map->map[i][j] = THRESHOLD(map->map[i][j]);
        }
    }
}

void trainHopfieldMap(HopfieldMap *map, Images *images){
    int size = map->imageVectorSize;
    
    for(int pt = 0; pt < images->trainingSize; pt++){
        for(int i = 0; i < size; i++){
            for(int j = 0; j < size; j++){
                if(i == j && SELFLINK != TRUE){
                    map->map[i][j] = 0;
                    continue;
                }
                map->map[i][j] += images->trainingImages[pt]->colorValues[i]*images->trainingImages[pt]->colorValues[j];
            }
        }
    }
    
    //applyThresholdMap(map);
}


void recoverHopfieldMap(HopfieldMap *map, Images *images, time_t startTime){

    int size = map->imageVectorSize;
    int equalityPixels = -1;
    int oldEqualityPixels = 0; 
    int randomIndex = -1; 
    int counter = 0; 
    int contractorFound = -1;
    
    ImageVector *backup = createImageVector(images->output[0]->numPixelsX, images->output[0]->numPixelsY );
    
    for(int pat = 0; pat < images->testingSize; pat++){
        while(time(NULL) < (startTime + TIMEOUT/images->testingSize)){ 
            
            copyVectors(backup, images->testingImages[pat]);
            
            for(int n1 = 0; n1 < size; n1++){
                images->output[pat]->colorValues[n1] = 0;
                for(int n2 = 0; n2 < size; n2++){
                    images->output[pat]->colorValues[n1] += map->map[n1][n2]*images->testingImages[pat]->colorValues[n2];
                }
            }
                       
            applyThreshold(images->output[pat]);
            oldEqualityPixels = equalityPixels;
            equalityPixels = compareImageVector(images->output[pat], backup);
            if(counter == CONVERGENCE){
                contractorFound = -1;
                for(int k = 0; k < images->trainingSize; k++){
                    contractorFound = compareImageVector(images->output[pat], images->trainingImages[k]);
                    if(contractorFound == images->output[pat]->size){
                        break;
                    }
                    else if(contractorFound == 0){
                        break;
                    }
                }
                
                if(contractorFound == images->output[pat]->size || contractorFound == 0){
                    break;
                }
                else{
                    //introduce some random noise
                    for(int k = 0; k < RANDOMFLIPS; k++){
                            randomIndex = randomInt(0, size-1);
                            images->output[pat]->colorValues[randomIndex] *= -1;
                    }
                    counter = 0; 
                }
            }
            else if(equalityPixels == oldEqualityPixels){
                counter++;
            }
            else{
                counter = 0; 
            }
            //Feed back input
            memcpy(images->testingImages[pat]->colorValues, images->output[pat]->colorValues, images->output[pat]->size * sizeof(int));
           
        }
        startTime = time(NULL);
    }
    
    free(backup);
}

/*
 *  Input and Output routines
 */

void stringToBinArray(char *string, int length, ImageVector *vector){
    for(int i = 0; i < length; i++){
        if(string[i] == HIGHVALUE){
            vector->colorValues[i] = 1;
        }
        else if(string[i] =='\0'){
            return;
        }        
        else{
            vector->colorValues[i] = -1;
        }
    }
}


int readImage(int numPixelX, int numPixelY, ImageVector *image){
    int lineCounter = 0;
    int charsRead = 0;
    int size = numPixelX * numPixelY;
    char *inputString = calloc( numPixelX +1 , sizeof(char));
    char *longString = calloc( size + 1, sizeof(char));
    int returnValue = 0; 
    
    while(lineCounter <= numPixelY){
        charsRead = scanf("%s\n", inputString);
        if(charsRead == EOF && lineCounter == numPixelY){
            returnValue = IMAEOF;
            break;
        }
        else if(charsRead == EOF && lineCounter < numPixelY){
            returnValue = EOF;
            break;
        }
        else if(inputString[0] == '-' ){
            if(inputString[1] == '-' && lineCounter == numPixelY){
                returnValue = IMASWITCHMODE;
                break;
            }
            else if(inputString[1] == '-' && lineCounter < numPixelY){
                returnValue = SWITCHMODE;
                break;
            }
            else{
                returnValue = 1;
                break;
            }
        }
        else{
            memcpy(longString + numPixelX * lineCounter, inputString, numPixelX);
        }
            
        memset(inputString, '\0', numPixelX + 1);
        lineCounter++;
    }
    
    stringToBinArray(longString, size, image);
    free(inputString);
    free(longString);
    return returnValue;
}

void readInput(Images *images){
    int x = images->numPixelX;
    int y = images->numPixelY;
    
    int trainingLength = -1;
    int testingLength = -1;
    
    ImageVector **training = calloc(1, sizeof(ImageVector *)); //assume at least 1 trainingInput
    training[0] = createImageVector(x,y);
    ImageVector **testing = calloc(1, sizeof(ImageVector *)); //assume at least 1 testingInput
    testing[0] = createImageVector(x,y);
    
    char mode = TRAINING;
    
    int readImageCheck = 0;
    int index = 0; 
    //scan strings until end of file
    while( readImageCheck != EOF){
        
        if(mode == TRAINING){
            if(trainingLength == -1){
                index = 0;
            }
            else{
                index = trainingLength;
            }
            readImageCheck = readImage(x, y, training[index]);
            if(readImageCheck == 1 && trainingLength == -1){
                trainingLength = 0; 
            }
            else if(readImageCheck == IMAEOF && trainingLength == -1){
                trainingLength = 0;
                readImageCheck = EOF;
            }
            else if(readImageCheck == IMAEOF){
                readImageCheck = EOF;
            }
            else if(readImageCheck == IMASWITCHMODE && trainingLength == -1){
                trainingLength = 0; 
                readImageCheck = SWITCHMODE;
            }
            else if(readImageCheck == IMASWITCHMODE){
                readImageCheck = SWITCHMODE;
            }
        }
        else if(mode == TESTING){
            if(testingLength == -1){
                index = 0;
            }
            else{
                index = testingLength;
            }
            readImageCheck = readImage(x, y, testing[index]);
            if(readImageCheck == 1 && testingLength == -1){
                testingLength = 0; 
            }
            else if(readImageCheck == IMAEOF && testingLength == -1){
                testingLength = 0;
                readImageCheck = EOF;
            }
            else if(readImageCheck == IMAEOF){
                readImageCheck = EOF;
            }
        }
            
        
        switch(readImageCheck){
            case EOF: ; break;
            case SWITCHMODE:
                mode = TESTING;
                images->trainingSize = trainingLength + 1;
                break;
            default:
                if(mode == TRAINING){
                    trainingLength++;
                    training = realloc(training, (trainingLength+1)*sizeof(ImageVector *));
                    training[trainingLength] = createImageVector(x,y);
                }
                else if(mode == TESTING){
                    testingLength++;
                    testing = realloc(testing, (testingLength + 1)*sizeof(ImageVector *));
                    testing[testingLength] = createImageVector(x,y);
                }
                
        }       
    }
    
    if(testingLength > -1){
        images->testingSize = testingLength + 1;
        images->testingImages = testing;
    }
    if(trainingLength > -1){
        images->trainingImages = training;
    }
}




/*
 * Main program
 */

int main(int argc, char** argv) { 
    
    srand((unsigned) time(NULL));
    time_t startTime = time(NULL);
    
    int numPixelX = NUMPIXELX;
    int numPixelY = NUMPIXELY;
    Images *images = createImages(numPixelX, numPixelY);
    HopfieldMap *map = createHopfieldMap(numPixelX*numPixelY);
    readInput(images);
    
    
    if(PRINTBOOL){
        printf("Training Images:\n");
        for(int i = 0; i < images->trainingSize; i++){
            printVector(images->trainingImages[i], TRUE);
            printf("\n");
        }
        printf("\n-------------------------\n");
        printf("Testing Images:\n");
        for(int i = 0; i < images->testingSize; i++){
            printVector(images->testingImages[i], TRUE);
            printf("\n");
        }
    }
    
    
    trainHopfieldMap(map, images);
    createImagesOutput(images);
    recoverHopfieldMap(map, images, startTime);
    
    if(PRINTBOOL){
        printf("\n-------------------------\n");
        printf("Reconstructed Images:\n\n");
        for(int i = 0; i < images->outputSize; i++){
            printVectorsNextTo(images->testingImagesBackup[i], images->output[i]);
            if(i < images->outputSize - 1){
                printf("-\n");
            }
        }
    }
    else{
        for(int i = 0; i < images->outputSize; i++){
            printVector(images->output[i], FALSE);
            if(i < images->outputSize - 1){
                printf("-\n");
            }
        }
    }
    
    
    if(PRINTBOOL){
        sleep(0.1);
    }
    freeImages(images);
    freeHopfieldMap(map);
    
    return (EXIT_SUCCESS);
}

