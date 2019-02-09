/*
 * Some functions used for Reinforcement Task
 * 
 * Agent is learning its way in a grid world from a starting point to a goal 
 * by either Q-Learning or SARSA-Learning. 
 * 
*/

#include "rl_homework_lib.h"


// Function of the World
State getNextState(State s_t, Action a_t){
    State nextState = s_t;
    
    switch(a_t){
        case UP:
            if(s_t.y != 1){
                nextState.y = s_t.y - 1;
            }
            break;
            
        case DOWN:
            if(s_t.y != (world_height-2)){
                nextState.y = s_t.y + 1;
            }
            break;
            
        case LEFT:
            if(s_t.x != 1){
                nextState.x = s_t.x - 1;
            }
            break;
            
        case RIGHT:
            if(s_t.x != (world_width-2)){
                nextState.x = s_t.x + 1;
            }
            break;
    }
    return nextState;
}
float getNextReward(State s_t){
    float nextReward = 0.0;
    if(s_t.y == (world_height-2) && (s_t.x >= 2 && s_t.x <= (world_width-3))){
        nextReward = dropOffPenalty;
    }
    else if(s_t.x == (world_width - 2) && s_t.y == (world_height-2)){
        nextReward = goalReward;
    }
    else{
        nextReward = stepPenalty;
    }
    
    return nextReward;
}
int isTerminalState(State s_t){
    int terminalState = 0;
    if(s_t.y == (world_height-2) && (s_t.x >= 2 && s_t.x <= (world_width-3))){
        terminalState = 1;
    }
    else if(s_t.x == (world_width - 2) && s_t.y == (world_height-2)){
        terminalState = 1;
    }
    else{
        terminalState = 0;
    }
    return terminalState;
}




/*
 * Random Utlity Functions
 */
float randomFloat(float from, float to){
    return ( (float)rand() / (float)RAND_MAX ) * (to - from) + from;
}

int randomInt(int from, int to){
    return ( rand() % (to - from + 1)) + from;  
}


// Function of the Agent:
Action getNextAction(Agent* a, State s_t){
    float random = randomFloat(0,1);
    if(random <= a->epsilon){
        return getRandomAction();
    }
    else{
        return getGreedyAction(a, s_t);
    }
    
    return UP;
}
Action getRandomAction(){
    int randomAction = randomInt(0,3);
    return (Action)randomAction;
}
Action getGreedyAction(Agent* a, State s_t){
    float maxQvalue = -1e10;
    float QloopValue = 0.0;
    int maxQvalueIndex = -1;
    
    for(int i = 0; i < nActions; i++){
        QloopValue = a->QValues[stateAction2idx(s_t, (Action)i )];
        if(QloopValue > maxQvalue){
            maxQvalue = QloopValue;
            maxQvalueIndex = i;
        }
        else if(QloopValue == maxQvalue){
            if( randomFloat(0,1) < 0.5 ){
                maxQvalue = QloopValue;
                maxQvalueIndex = i;
            }
            else{
                //do nothing, keep old maximum
            }
        }
    }
    
    return (Action)maxQvalueIndex;
}


/*
 * Helper Function
 */

float maxActionQvalue(Agent *a, State s){
    float Qmax = -1e10;
    float Qloop = 0.0;
    
    for(int i = 0; i < 4; i++){
        Qloop = a->QValues[ stateAction2idx(s, (Action)i )];
        if(Qloop > Qmax){
            Qmax = Qloop;
        }
    }
    
    return Qmax;
}


Action argmaxActionQvalue(Agent *a, State s){
    Action ActionMax = (Action)0;
    float Qmax = -1e10;
    float Qloop = 0.0;
    
    for(int i = 0; i < 4; i++){
        Qloop = a->QValues[ stateAction2idx(s, (Action)i )];
        if(Qloop > Qmax){
            Qmax = Qloop;
            ActionMax = (Action)i;
        }
    }
    
    return ActionMax;
}

// Function of Gameplay:
Agent* trainAgent(int nIter, int verbose, Agent* a){
    //gets untrained agent
    //performs nIter training iterations
    //computes avgReward per episode and saves into agent
    //returns trained agent
    a->avgReward = 0.0;
    srand((unsigned) time(NULL));
    for(int iter = 1; iter <= nIter; iter++){
        a->accumReward = 0.0;
        playEpisode(nIter, verbose, a);
    }
    
    return a;
}
Agent* playEpisode(int nEpisode, int verbose, Agent* a){
    //playsEpisode 
    //starts at starting grid points
    //while terminal state not found: select next state and update Q-values
    //compute accumulated reward by adding reward to agent
    
    
    State s_t1 = {.x = 1, .y = world_height - 2};
    State s_t2 = {.x = 0, .y = 0};
    Action a_t = UP; //only initialized with UP
    float r_t = 0.0;
    
    
    while( isTerminalState(s_t1) == 0 ){
        a_t = getNextAction(a, s_t1);
        s_t2 = getNextState(s_t1, a_t);
        r_t = getNextReward(s_t2);
        a->accumReward += r_t;
        
        switch(a->type){
            case QLearning:
                qLearningUpdate(a, s_t1, s_t2, a_t, r_t);
                break;
            case SARSA:
                sarsaUpdate(a, s_t1, s_t2, a_t, r_t);
                break;
        }
        
        s_t1 = s_t2;
    }
    a->avgReward += a->accumReward / (float)nEpisode;

    return a;
}


// SARSA and Q-Learning Updates:

Agent* sarsaUpdate(Agent* a, State s_t1, State s_t2, Action a_t1, float r_t){
    Action a_t2 = getNextAction(a, s_t2);
    
    a->QValues[ stateAction2idx(s_t1, a_t1)  ] = (1-a->alpha) * a->QValues[ stateAction2idx(s_t1, a_t1)  ] + 
            a->alpha*( r_t + a->gamma * a->QValues[ stateAction2idx(s_t2, a_t2)]  );
    return a;
}
Agent* qLearningUpdate(Agent* a, State s_t1, State s_t2, Action a_t1, float r_t){
    a->QValues[ stateAction2idx(s_t1, a_t1)  ] = (1-a->alpha) * a->QValues[ stateAction2idx(s_t1, a_t1)  ] + 
            a->alpha*( r_t + a->gamma * maxActionQvalue(a, s_t2 )  );
    return a;
}





/*
 * Difference between SARSA and QLearning
 * 
 * The average reward of SARSA will be lower than the average reward of QLearning. 
 * This is because, QLearning updates the QValues assuming an optimal policy from the next step on, regardless of the actual
 * action taken. So if the agent gets a negative reward in the next step (e.g. falls of the cliff, random action), this does not change the 
 * QValue dramatically, because a way better solution out of this next state may exist. Hence, QLearnig will find a better path, 
 * as it will explore more of a world. In the cliff-world, this also means that it will take a higher risk (will fall off the cliff
 * more often). QLearnining in a cliff world will converge to a solution, where the agent walks just on the edge of the cliff to a 
 * a goal state. 
 * SARSA on the other hand, updates its QValues by accounting the QValue reached that is actually reached in the next state by 
 * following the policy. If the agent then gets a negative reward in the following state following the policy (for example falling off by just random action selection), 
 * the state before will get a decreased QValue. The algorithm will then find a way around this low value. In the cliff world an agent
 * will walk a certain distance away from the edge until it reaches the goal state. Since a bad random choice amplifies this behavior.
 * Following an epsilon-greedy-policy a high epsilon-value will lead to an agent walking far away from the edge. With an epsilon value, 
 * tending to 0, the agent will also explore more and more the optimal solution, since bad random choices causing the detours will appear less often. 
 * This also explains why SARSA and QLearning behave similar when using the greedy policy. 
 */

