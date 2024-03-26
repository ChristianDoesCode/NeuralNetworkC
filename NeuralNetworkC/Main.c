#include <stdio.h>
#include <stdlib.h>
#include <math.h>

//Variable to store the length of the mallocKillList because you can get the length of a heap allocated array - Christian Phillips
unsigned int mallocKillListLength = 2;
//void** to store all the pointers that need to be freed at the end of the program - Christian Phillips
void** mallocKillList = NULL;

//Beggining of RandomNumberGen.h

//this is used as the seed for the random number generator - Christian Phillips
unsigned long long seed;

//Function that returns the max value that can be stored in a uLong - Christian Phillips
unsigned long long getMaxULongLongValue()
{
    unsigned long long uLongLongLength = 0LLU;
    uLongLongLength = ~uLongLongLength;
    return uLongLongLength;
}

//Function that returns the max value that can be stored in a uInt - Christian Phillips
unsigned int getMaxUIntValue()
{
    unsigned int uIntLength = 0;
    uIntLength = ~uIntLength;
    return uIntLength;
}

//Sets the seed for the random number gen - Christian Phillips
void setRandSeed(unsigned long inputSeed)
{
    seed = inputSeed;
}

//Function to generate pseudo-random ints - Christian Phillips
unsigned int genRandInt()
{
    seed = (561 * seed + 33) % getMaxULongLongValue();
    return seed % getMaxUIntValue();
}

//Ending of RandomNumberGen.h

struct baseNode
{
    float value;
    float* pDerivatives;
    int pDerivativesLength;
    struct baseNode* parentNodes;
};

struct nodeBias
{
    float value;
    float gradient;
};

//Make Structures for the nodes, rawNodes and weights of the network - Christian Phillips

//Structure for input nodes probaly could have just used an array of floats - Christian Phillips
struct inputNode
{
    float value;
};

//This is the structure for the raw hidden nodes stores the value the bias and the partial derivative of its parent node with respect to itself - Christian Phillips
struct rawHiddenNode
{
    float value;
    struct nodeBias bias;
    float* pDerivatives;
    int pDerivativesLength;
    struct baseNode* parentNodes;
};

struct rawOutputNode
{
    float value;
    struct nodeBias bias;
    float* pDerivatives;
    int pDerivativesLength;
    struct baseNode* parentNodes;
};

struct hiddenNode
{
    float value;
    float* pDerivatives;
    int pDerivativesLength;
    struct baseNode* parentNodes;
};

struct outputNode
{
    float value;
    float* pDerivatives;
    int pDerivativesLength;
    struct baseNode* parentNodes;
};

struct weight
{
    float value;
    float pDerivative;
    int pDerivativesLength;
    struct baseNode* parentNodes;
    char parentNodeType;
    void* childNode;
    float gradient;
};

float relu(float num)
{
    if (num >= 0)
    {
        return num;
    }
    else
    {
        return 0.0f;
    }
}

//Make a struct for the Neural Network - Christian Phillips
struct NeuralNetwork
{
    //Variables to set the number of nodes and layers - Christian Phillips
    unsigned int numInputNodes;
    unsigned int numHiddenLayers;
    unsigned int numOutputNodes;
    unsigned int* hiddenLayers;

    //Variables to store the values of the nodes - Christian Phillips
    struct inputNode* inputNodes;
    struct rawHiddenNode** rawHiddenNodes;
    struct hiddenNode** hiddenNodes;
    struct rawOutputNode* rawOutputNodes;
    struct outputNode* outputNodes;

    //Variables for the weights between the layers - Christian Phillips
    struct weight** weightsInput_Hidden;
    struct weight*** weightsHidden_Hidden;
    struct weight** weightsHidden_Output;

    //float array to store the losses for the neural network - Christian Phillips
    float loss;

    int correctOutputNode;

    float learningRate;
};

//This is the method to perform the soft max activation function on the raw Output nodes which calculates the output nodes it takes the pointer to the neural network and the index for which output node is being calculated - Christian Phillips
float softMax(struct NeuralNetwork* pNetwork, unsigned int outputNodeIndex)
{
    float sumRawOuputs = 0.0f;

    for (int i = 0; i < pNetwork->numOutputNodes; i++)
    {
        sumRawOuputs += (float)exp(pNetwork->rawOutputNodes[i].value);
    }

    return (float)exp(pNetwork->rawOutputNodes[outputNodeIndex].value) / sumRawOuputs;
}

//This is an importaint function that adds 
int addMallocKill(void* ptr)
{
    if (mallocKillList == NULL)
    {
        mallocKillList = (void**)malloc(sizeof(void*) * mallocKillListLength);

        if (mallocKillList == NULL)
        {
            return EXIT_FAILURE;
        }

        mallocKillList[0] = ptr;

        for (int i = 1; i < mallocKillListLength; i++)
        {
            mallocKillList[i] = NULL;
        }

        return EXIT_SUCCESS;
    }
    else if (mallocKillList[mallocKillListLength - 1] != NULL)
    {
        void** newMallocKillList = (void**)malloc(sizeof(void*) * mallocKillListLength * 2);

        if (newMallocKillList == NULL)
        {
            return EXIT_FAILURE;
        }

        for (int i = 0; i < mallocKillListLength; i++)
        {
            newMallocKillList[i] = mallocKillList[i];
        }

        newMallocKillList[mallocKillListLength] = ptr;

        for (int i = mallocKillListLength + 1; i < mallocKillListLength * 2; i++)
        {
            newMallocKillList[i] = NULL;
        }

        mallocKillListLength = mallocKillListLength * 2;

        printf("mallocKillListLength: %d\n", mallocKillListLength);

        free(mallocKillList);

        mallocKillList = newMallocKillList;

        return EXIT_SUCCESS;
    }
    else
    {
        for (int i = 0; i < mallocKillListLength; i++)
        {
            if (mallocKillList[i] == NULL)
            {
                mallocKillList[i] = ptr;
                break;
            }
        }
        return EXIT_SUCCESS;
    }
}

//function to act like a constructor for the Neural Network
struct NeuralNetwork* constructNeuralNetwork(unsigned int numIN, unsigned int* HL, unsigned int numON, unsigned int numHL)
{
    struct NeuralNetwork* pNetwork = (struct NeuralNetwork*)malloc(sizeof(struct NeuralNetwork));
    addMallocKill(pNetwork);

    if (pNetwork == NULL)
    {
        return NULL;
    }

    pNetwork->learningRate = 0.01f;

    pNetwork->numInputNodes = numIN;
    pNetwork->hiddenLayers = HL;
    pNetwork->numOutputNodes = numON;
    pNetwork->numHiddenLayers = numHL;

    pNetwork->inputNodes = (struct inputNode*)malloc(sizeof(struct inputNode) * numIN);
    addMallocKill(pNetwork->inputNodes);

    pNetwork->rawHiddenNodes = (struct rawHiddenNode**)malloc(sizeof(struct rawHiddenNode*) * numHL);
    addMallocKill(pNetwork->rawHiddenNodes);

    for (int i = 0; i < numHL; i++)
    {
        pNetwork->rawHiddenNodes[i] = (struct rawHiddenNode*)malloc(sizeof(struct rawHiddenNode) * HL[i]);
        addMallocKill(pNetwork->rawHiddenNodes[i]);
    }

    pNetwork->hiddenNodes = (struct hiddenNode**)malloc(sizeof(struct hiddenNode*) * numHL);
    addMallocKill(pNetwork->hiddenNodes);

    for (int i = 0; i < numHL; i++)
    {
        pNetwork->hiddenNodes[i] = (struct hiddenNode*)malloc(sizeof(struct hiddenNode) * HL[i]);
        addMallocKill(pNetwork->hiddenNodes[i]);
    }

    pNetwork->rawOutputNodes = (struct rawOutputNode*)malloc(sizeof(struct rawOutputNode) * numON);
    addMallocKill(pNetwork->rawOutputNodes);
    pNetwork->outputNodes = (struct outputNode*)malloc(sizeof(struct outputNode) * numON);
    addMallocKill(pNetwork->outputNodes);

    //give the loss parameter of the neural network a default value - Christian Phillips
    pNetwork->loss = 0.0f;

    return pNetwork;
}

int setWeights(struct NeuralNetwork* pNetwork)
{
    //float used to store the result of the formula used to calculate the weight
    float randFloat = 0.0;

    pNetwork->weightsInput_Hidden = (struct weight**)malloc(sizeof(struct weight*) * pNetwork->numInputNodes);
    addMallocKill(pNetwork->weightsInput_Hidden);

    for (int i = 0; i < pNetwork->numInputNodes; i++)
    {
        pNetwork->weightsInput_Hidden[i] = (struct weight*)malloc(sizeof(struct weight) * pNetwork->hiddenLayers[0]);
        addMallocKill(pNetwork->weightsInput_Hidden[i]);
    }

    pNetwork->weightsHidden_Hidden = (struct weight***)malloc(sizeof(struct weight**) * (pNetwork->numHiddenLayers - 1));
    addMallocKill(pNetwork->weightsHidden_Hidden);

    for (int i = 0; i < pNetwork->numHiddenLayers - 1; i++)
    {
        pNetwork->weightsHidden_Hidden[i] = (struct weight**)malloc(sizeof(struct weight*) * pNetwork->hiddenLayers[i]);
        addMallocKill(pNetwork->weightsHidden_Hidden[i]);

        for (int j = 0; j < pNetwork->hiddenLayers[i]; j++)
        {
            pNetwork->weightsHidden_Hidden[i][j] = (struct weight*)malloc(sizeof(struct weight) * pNetwork->hiddenLayers[i + 1]);
            addMallocKill(pNetwork->weightsHidden_Hidden[i][j]);
        }
    }

    pNetwork->weightsHidden_Output = (struct weight**)malloc(sizeof(struct weight*) * pNetwork->hiddenLayers[pNetwork->numHiddenLayers - 1]);
    addMallocKill(pNetwork->weightsHidden_Output);

    for (int i = 0; i < pNetwork->hiddenLayers[pNetwork->numHiddenLayers - 1]; i++)
    {
        pNetwork->weightsHidden_Output[i] = (struct weight*)malloc(sizeof(struct weight) * pNetwork->numOutputNodes);
        addMallocKill(pNetwork->weightsHidden_Output[i]);
    }

    for (int i = 0; i < pNetwork->numInputNodes; i++)
    {
        for (int j = 0; j < pNetwork->hiddenLayers[0]; j++)
        {
            randFloat = (genRandInt() % 100) / 50.0f - 1.0f;
            pNetwork->weightsInput_Hidden[i][j].value = randFloat;
            pNetwork->weightsInput_Hidden[i][j].parentNodes = (struct baseNode*)(&(pNetwork->rawHiddenNodes[0][j]));
            pNetwork->weightsInput_Hidden[i][j].childNode = &(pNetwork->inputNodes[i]);
        }
    }

    for (int i = 0; i < pNetwork->numHiddenLayers - 1; i++)
    {
        for (int j = 0; j < pNetwork->hiddenLayers[i]; j++)
        {
            for (int k = 0; k < pNetwork->hiddenLayers[i + 1]; k++)
            {
                randFloat = (genRandInt() % 100) / 50.0f - 1.0f;
                pNetwork->weightsHidden_Hidden[i][j][k].value = randFloat;
                pNetwork->weightsHidden_Hidden[i][j][k].parentNodes = (struct baseNode*)(&(pNetwork->rawHiddenNodes[i + 1][k]));
                pNetwork->weightsHidden_Hidden[i][j][k].childNode = &(pNetwork->hiddenNodes[i][j]);
            }
        }
    }

    for (int i = 0; i < pNetwork->hiddenLayers[pNetwork->numHiddenLayers - 1]; i++)
    {
        for (int j = 0; j < pNetwork->numOutputNodes; j++)
        {
            randFloat = (genRandInt() % 100) / 50.0f - 1.0f;
            pNetwork->weightsHidden_Output[i][j].value = randFloat;
            pNetwork->weightsHidden_Output[i][j].parentNodes = (struct baseNode*)(&(pNetwork->rawOutputNodes[j]));
            pNetwork->weightsHidden_Output[i][j].childNode = &(pNetwork->hiddenNodes[pNetwork->numHiddenLayers - 1][i]);
        }
    }

    return EXIT_SUCCESS;
}

int calcWeightDerivatives(struct NeuralNetwork* pNetwork)
{
    for (int i = 0; i < pNetwork->numInputNodes; i++)
    {
        for (int j = 0; j < pNetwork->hiddenLayers[0]; j++)
        {
            pNetwork->weightsInput_Hidden[i][j].pDerivativesLength = 1;
            pNetwork->weightsInput_Hidden[i][j].pDerivative = *((float*)pNetwork->weightsInput_Hidden[i][j].childNode);
            pNetwork->weightsInput_Hidden[i][j].gradient = 1.0f;
        }
    }

    for (int i = 0; i < pNetwork->numHiddenLayers - 1; i++)
    {
        for (int j = 0; j < pNetwork->hiddenLayers[i]; j++)
        {
            for (int k = 0; k < pNetwork->hiddenLayers[i + 1]; k++)
            {
                pNetwork->weightsHidden_Hidden[i][j][k].pDerivativesLength = 1;
                pNetwork->weightsHidden_Hidden[i][j][k].pDerivative = *((float*)pNetwork->weightsHidden_Hidden[i][j][k].childNode);
                pNetwork->weightsHidden_Hidden[i][j][k].gradient = 1.0f;
            }
        }
    }

    for (int i = 0; i < pNetwork->hiddenLayers[pNetwork->numHiddenLayers - 1]; i++)
    {
        for (int j = 0; j < pNetwork->numOutputNodes; j++)
        {
            pNetwork->weightsHidden_Output[i][j].pDerivativesLength = 1;
            pNetwork->weightsHidden_Output[i][j].pDerivative = *((float*)pNetwork->weightsHidden_Output[i][j].childNode);
            pNetwork->weightsHidden_Output[i][j].gradient = 1.0f;
        }
    }

    return EXIT_SUCCESS;
}

int setBiases(struct NeuralNetwork* pNetwork)
{
    if (pNetwork == NULL || pNetwork->rawHiddenNodes == NULL || pNetwork->rawOutputNodes == NULL)
    {
        return EXIT_FAILURE;
    }

    for (int i = 0; i < pNetwork->numHiddenLayers; i++)
    {
        for (int j = 0; j < pNetwork->hiddenLayers[i]; j++)
        {
            pNetwork->rawHiddenNodes[i][j].bias.value = 0.0f;
        }
    }

    for (int i = 0; i < pNetwork->numOutputNodes; i++)
    {
        pNetwork->rawOutputNodes[i].bias.value = 0.0f;
    }

    return EXIT_SUCCESS;
}

int setInputNodes(struct NeuralNetwork* pNetwork, float* inputs, unsigned int inputLength)
{
    if (pNetwork->numInputNodes != inputLength)
    {
        return EXIT_FAILURE;
    }

    if (pNetwork == NULL || pNetwork->inputNodes == NULL)
    {
        return EXIT_FAILURE;
    }

    for (int i = 0; i < pNetwork->numInputNodes; i++)
    {
        pNetwork->inputNodes[i].value = inputs[i];
    }
    return EXIT_SUCCESS;
}

int calcHiddenNodes(struct NeuralNetwork* pNetwork)
{
    if (pNetwork == NULL || pNetwork->inputNodes == NULL || pNetwork->rawHiddenNodes == NULL)
    {
        return EXIT_FAILURE;
    }

    for (int i = 0; i < pNetwork->hiddenLayers[0]; i++)
    {
        pNetwork->rawHiddenNodes[0][i].value = 0.0f;

        for (int j = 0; j < pNetwork->numInputNodes; j++)
        {
            pNetwork->rawHiddenNodes[0][i].value += pNetwork->inputNodes[j].value * pNetwork->weightsInput_Hidden[j][i].value;
        }

        pNetwork->rawHiddenNodes[0][i].value += pNetwork->rawHiddenNodes[0][i].bias.value;
        pNetwork->hiddenNodes[0][i].value = relu(pNetwork->rawHiddenNodes[0][i].value);
    }

    for (int i = 0; i < pNetwork->numHiddenLayers - 1; i++)
    {
        for (int j = 0; j < pNetwork->hiddenLayers[i + 1]; j++)
        {
            pNetwork->rawHiddenNodes[i + 1][j].value = 0.0f;

            for (int k = 0; k < pNetwork->hiddenLayers[i]; k++)
            {
                pNetwork->rawHiddenNodes[i + 1][j].value += pNetwork->hiddenNodes[i][k].value * pNetwork->weightsHidden_Hidden[i][k][j].value;
            }

            pNetwork->rawHiddenNodes[i + 1][j].value += pNetwork->rawHiddenNodes[i + 1][j].bias.value;
            pNetwork->hiddenNodes[i + 1][j].value = relu(pNetwork->rawHiddenNodes[i + 1][j].value);
        }
    }

    return EXIT_SUCCESS;
}

int calcHiddenDerivatives(struct NeuralNetwork* pNetwork)
{
    if (pNetwork == NULL || pNetwork->inputNodes == NULL || pNetwork->rawHiddenNodes == NULL)
    {
        return EXIT_FAILURE;
    }

    for (int i = 0; i < pNetwork->numHiddenLayers; i++)
    {
        for (int j = 0; j < pNetwork->hiddenLayers[i]; j++)
        {
            pNetwork->rawHiddenNodes[i][j].pDerivatives = (float*)malloc(sizeof(float));
            pNetwork->rawHiddenNodes[i][j].pDerivativesLength = 1;
            addMallocKill(pNetwork->rawHiddenNodes[i][j].pDerivatives);

            pNetwork->rawHiddenNodes[i][j].parentNodes = (struct baseNode*)(&(pNetwork->hiddenNodes[i][j]));
            printf("hiddenNodes Value: %f\n", ((struct baseNode*)(&(pNetwork->hiddenNodes[i][j])))->value);

            if (pNetwork->rawHiddenNodes[i][j].value > 0)
            {
                pNetwork->rawHiddenNodes[i][j].pDerivatives[0] = 1.0f;
            }
            else
            {
                pNetwork->rawHiddenNodes[i][j].pDerivatives[0] = 0.0f;
            }

            if ((i + 1) < pNetwork->numHiddenLayers)
            {
                pNetwork->hiddenNodes[i][j].pDerivatives = (float*)malloc(sizeof(pNetwork->hiddenNodes[i + 1][0].value) * pNetwork->hiddenLayers[i + 1]);
                pNetwork->hiddenNodes[i][j].pDerivativesLength = pNetwork->hiddenLayers[i + 1];
                addMallocKill(pNetwork->hiddenNodes[i][j].pDerivatives);

                pNetwork->hiddenNodes[i][j].parentNodes = (struct baseNode*)(pNetwork->hiddenNodes[i + 1]);

                for (int k = 0; k < pNetwork->hiddenLayers[i + 1]; k++)
                {
                    pNetwork->hiddenNodes[i][j].pDerivatives[k] = pNetwork->weightsHidden_Hidden[i][j][k].value;
                }
            }
            else
            {
                pNetwork->hiddenNodes[i][j].pDerivatives = (float*)malloc(sizeof(pNetwork->outputNodes[0].value) * pNetwork->numOutputNodes);
                pNetwork->hiddenNodes[i][j].pDerivativesLength = pNetwork->numOutputNodes;
                addMallocKill(pNetwork->hiddenNodes[i][j].pDerivatives);

                pNetwork->hiddenNodes[i][j].parentNodes = (struct baseNode*)(pNetwork->outputNodes);

                for (int k = 0; k < pNetwork->numOutputNodes; k++)
                {
                    pNetwork->hiddenNodes[i][j].pDerivatives[k] = pNetwork->weightsHidden_Output[j][k].value;
                }
            }
        }
    }

    return EXIT_SUCCESS;
}

int calcOutputNodes(struct NeuralNetwork* pNetwork)
{
    if (pNetwork == NULL || pNetwork->rawOutputNodes == NULL || pNetwork->outputNodes == NULL)
    {
        return EXIT_FAILURE;
    }

    for (int i = 0; i < pNetwork->numOutputNodes; i++)
    {
        pNetwork->rawOutputNodes[i].value = 0.0f;
        for (int j = 0; j < pNetwork->hiddenLayers[pNetwork->numHiddenLayers - 1]; j++)
        {
            pNetwork->rawOutputNodes[i].value += pNetwork->hiddenNodes[pNetwork->numHiddenLayers - 1][j].value * pNetwork->weightsHidden_Output[j][i].value;
        }

        pNetwork->rawOutputNodes[i].value += pNetwork->rawOutputNodes[i].bias.value;
    }

    for (int i = 0; i < pNetwork->numOutputNodes; i++)
    {
        pNetwork->outputNodes[i].value = softMax(pNetwork, i);
    }

    return EXIT_SUCCESS;
}

int calcOutputDerivatives(struct NeuralNetwork* pNetwork)
{
    if (pNetwork == NULL || pNetwork->rawOutputNodes == NULL || pNetwork->outputNodes == NULL)
    {
        return EXIT_FAILURE;
    }

    for (int i = 0; i < pNetwork->numOutputNodes; i++)
    {
        pNetwork->rawOutputNodes[i].pDerivatives = (float*)malloc(sizeof(float) * pNetwork->numOutputNodes);
        pNetwork->rawOutputNodes[i].pDerivativesLength = pNetwork->numOutputNodes;
        addMallocKill(pNetwork->rawOutputNodes[i].pDerivatives);

        pNetwork->rawOutputNodes[i].parentNodes = (struct baseNode*)(pNetwork->outputNodes);

        for (int j = 0; j < pNetwork->numOutputNodes; j++)
        {
            if (i == j)
            {
                pNetwork->rawOutputNodes[i].pDerivatives[j] = pNetwork->outputNodes[i].value * (1.0f - pNetwork->outputNodes[i].value);
            }
            else
            {
                pNetwork->rawOutputNodes[i].pDerivatives[j] = -(pNetwork->outputNodes[j].value) * pNetwork->outputNodes[i].value;
            }
        }
    }

    return EXIT_SUCCESS;
}

int calcCrossEntropyDerivatives(struct NeuralNetwork* pNetwork)
{
    if (pNetwork == NULL || pNetwork->outputNodes == NULL)
    {
        return EXIT_FAILURE;
    }

    for (int i = 0; i < pNetwork->numOutputNodes; i++)
    {
        pNetwork->outputNodes[i].pDerivatives = (float*)malloc(sizeof(float));
        pNetwork->outputNodes[i].pDerivativesLength = 1;
        addMallocKill(pNetwork->outputNodes[i].pDerivatives);

        pNetwork->outputNodes[i].parentNodes = NULL;

        pNetwork->outputNodes[i].pDerivatives[0] = -1.0f / pNetwork->outputNodes[i].value;
    }

    return EXIT_SUCCESS;
}

//Function that calculates the gradients for the weights of the neural network (FUNCTION STILL MAY NOT WORK MAKE SURE TO TEST) - Christian Phillips
int calcWeightGradient(struct weight* pWeight, struct baseNode* pNode, int correctOutputNode, float prevTotal)
{
    //check to see if the pointer to the weight is NULL this could be caused by the weight not being assigned memory or being set to NULL - Christian Phillips
    if (pWeight == NULL)
    {
        return EXIT_FAILURE;
    }

    //This code runs if the parentNodes of pNode are NULL this is done so that the code will run up until it reaches the output nodes - Christian Phillips
    if (pNode->parentNodes != NULL)
    {

        printf("pNode->value: %f\n", pNode->value);
        printf("pNode->parentNodes[0].value: %f\n", pNode->parentNodes);

        for (int i = 0; i < pNode->pDerivativesLength; i++)
        {
            //uses recursion to go to the parent node of the current node then makes the prevTotal for the next method call equal to the current prevTotal value times the partial derivative for this node at the current index i - Christian Phillips

            calcWeightGradient(pWeight, &((struct baseNode*)(pNode->parentNodes))[i], correctOutputNode, (prevTotal * pNode->pDerivatives[i]));// Original Error Location
        }
    }
    //This runs once the pNode is an output node - Christian Phillips
    else if (pNode->parentNodes == NULL)
    {
        for (int i = 0; i < pNode->pDerivativesLength; i++)
        {
            //Makes sure that the code does not multiply the wrong cross entropy partial derivates with respect to the wrong node - Christian Phillips
            if (i == correctOutputNode)
            {
                pWeight->gradient *= prevTotal * pNode->pDerivatives[i];
            }
        }
    }

    return EXIT_SUCCESS;
}

//Function that calculates the gradients for the biases of the neural network (FUNCTION STILL MAY NOT WORK MAKE SURE TO TEST) - Christian Phillips
int calcBiasGradient(struct nodeBias* pBias, struct baseNode* pNode, int correctOutputNode, float prevTotal)
{
    //check to see if the pointer to the bias is NULL this could be caused by the bias not being assigned memory or being set to NULL - Christian Phillips
    if (pBias == NULL)
    {
        return EXIT_FAILURE;
    }

    //This code runs if the parentNodes of pNode are NULL this is done so that the code will run up until it reaches the output nodes - Christian Phillips
    if (pNode->parentNodes != NULL)
    {
        for (int i = 0; i < pNode->pDerivativesLength; i++)
        {
            //uses recursion to go to the parent node of the current node then makes the prevTotal for the next method call equal to the current prevTotal value times the partial derivative for this node at the current index i - Christian Phillips
            calcBiasGradient(pBias, &((struct baseNode*)(pNode->parentNodes))[i], correctOutputNode, (prevTotal * pNode->pDerivatives[i]));
        }
    }
    //This runs once the pNode is an output node - Christian Phillips
    else if (pNode->parentNodes == NULL)
    {
        for (int i = 0; i < pNode->pDerivativesLength; i++)
        {
            //Makes sure that the code does not multiply the wrong cross entropy partial derivates with respect to the wrong node - Christian Phillips
            if (i == correctOutputNode)
            {
                pBias->gradient *= prevTotal * pNode->pDerivatives[i];
            }
        }
    }

    return EXIT_SUCCESS;
}

int calcParameters(struct NeuralNetwork* pNetwork)
{
    if (pNetwork == NULL)
    {
        return EXIT_FAILURE;
    }

    //for loops to calculate the new weights of the neural network - Christian Phillips
    for (int i = 0; i < pNetwork->numInputNodes; i++)
    {
        for (int j = 0; j < pNetwork->hiddenLayers[0]; j++)
        {
            if (pNetwork->weightsInput_Hidden[i][j].parentNodes == (struct baseNode*)(&(pNetwork->rawHiddenNodes[0][j])))
            {
                printf("This Ran: %f\n", ((struct baseNode*)(&(pNetwork->rawHiddenNodes[0][j])))->parentNodes->value);
                printf("This Ran: %f\n", pNetwork->weightsInput_Hidden[i][j].parentNodes->parentNodes->value);//Error occurs Here
            }
            calcWeightGradient(&(pNetwork->weightsInput_Hidden[i][j]), pNetwork->weightsInput_Hidden[i][j].parentNodes, pNetwork->correctOutputNode, pNetwork->weightsInput_Hidden[i][j].pDerivative);
            pNetwork->weightsInput_Hidden[i][j].value -= pNetwork->weightsInput_Hidden[i][j].gradient * pNetwork->learningRate;
        }
    }

    for (int i = 0; i < pNetwork->numHiddenLayers - 1; i++)
    {
        for (int j = 0; j < pNetwork->hiddenLayers[i]; j++)
        {
            for (int k = 0; k < pNetwork->hiddenLayers[i + 1]; k++)
            {
                calcWeightGradient(&(pNetwork->weightsHidden_Hidden[i][j][k]), (struct baseNode*)(pNetwork->weightsHidden_Hidden[i][j][k].parentNodes), pNetwork->correctOutputNode, pNetwork->weightsHidden_Hidden[i][j][k].pDerivative);
                pNetwork->weightsHidden_Hidden[i][j][k].value -= pNetwork->weightsHidden_Hidden[i][j][k].gradient * pNetwork->learningRate;
            }
        }
    }

    for (int i = 0; i < pNetwork->hiddenLayers[pNetwork->numHiddenLayers - 1]; i++)
    {
        for (int j = 0; j < pNetwork->numOutputNodes; j++)
        {
            calcWeightGradient(&(pNetwork->weightsHidden_Output[i][j]), (struct baseNode*)(pNetwork->weightsHidden_Output[i][j].parentNodes), pNetwork->correctOutputNode, pNetwork->weightsHidden_Output[i][j].pDerivative);
            pNetwork->weightsHidden_Output[i][j].value -= pNetwork->weightsHidden_Output[i][j].gradient * pNetwork->learningRate;
        }
    }

    //for loops to calculate the new biases for the neural network - Christian Phillips

    for (int i = 0; i < pNetwork->numHiddenLayers; i++)
    {
        for (int j = 0; j < pNetwork->hiddenLayers[i]; j++)
        {
            calcBiasGradient(&(pNetwork->rawHiddenNodes[i][j].bias), &(((struct baseNode*)pNetwork->rawHiddenNodes[i])[j]), pNetwork->correctOutputNode, 1.0f);
            pNetwork->rawHiddenNodes[i][j].bias.value -= pNetwork->rawHiddenNodes[i][j].bias.gradient * pNetwork->learningRate;
        }
    }

    for (int i = 0; i < pNetwork->numOutputNodes; i++)
    {
        calcBiasGradient(&(pNetwork->rawOutputNodes[i].bias), &(((struct baseNode*)pNetwork->rawOutputNodes)[i]), pNetwork->correctOutputNode, 1.0f);
        pNetwork->rawOutputNodes[i].bias.value -= pNetwork->rawOutputNodes[i].bias.gradient * pNetwork->learningRate;
    }

    return EXIT_SUCCESS;
}

//function to calculate and display the loss of the neural network
int calcTotalLoss(struct NeuralNetwork* pNetwork)
{
    pNetwork->loss = -log(pNetwork->outputNodes[pNetwork->correctOutputNode - 1].value);
    printf("test: %f\n", pNetwork->outputNodes[pNetwork->correctOutputNode - 1].value);
    printf("loss: %f\n", pNetwork->loss);

    return EXIT_SUCCESS;
}

int mallocDestroyer()
{
    if (mallocKillList == NULL)
    {
        return EXIT_FAILURE;
    }
    for (int i = mallocKillListLength - 1; i >= 0; i--)
    {
        free(mallocKillList[i]);
    }

    free(mallocKillList);

    return EXIT_SUCCESS;
}

int main()
{
    setRandSeed(99);

    unsigned int* hiddenLayers = (unsigned int*)malloc(sizeof(int) * 2);
    addMallocKill(hiddenLayers);

    float* inputs = (float*)malloc(sizeof(float) * 3);
    addMallocKill(inputs);

    inputs[0] = 0.1f;
    inputs[1] = 0.2f;
    inputs[2] = 0.3f;

    hiddenLayers[0] = 2;
    hiddenLayers[1] = 2;

    struct NeuralNetwork* pNetwork = constructNeuralNetwork(3, hiddenLayers, 3, 2);

    pNetwork->correctOutputNode = 1;
    setWeights(pNetwork);
    setBiases(pNetwork);
    setInputNodes(pNetwork, inputs, 3);
    calcHiddenNodes(pNetwork);
    calcOutputNodes(pNetwork);
    calcTotalLoss(pNetwork);
    calcWeightDerivatives(pNetwork);
    calcHiddenDerivatives(pNetwork);
    calcOutputDerivatives(pNetwork);
    calcCrossEntropyDerivatives(pNetwork);
    printf("This Ran: %d\n", pNetwork->rawHiddenNodes[0][0].pDerivativesLength);
    calcParameters(pNetwork);

    printf("Weigth: %f\n", pNetwork->weightsInput_Hidden[1][0].value);

    printf("Node: %f\n", pNetwork->hiddenNodes[0][0].value);
    printf("Node: %f\n", pNetwork->hiddenNodes[0][1].value);
    printf("Weigth: %f\n", pNetwork->weightsHidden_Hidden[0][0][0].value);
    printf("Weight: %f\n", pNetwork->weightsHidden_Hidden[0][1][0].value);
    printf("Node: %f\n", pNetwork->hiddenNodes[1][0].value);
    printf("Node: %f\n", pNetwork->outputNodes[0].value);

    mallocDestroyer();

    return 0;
}