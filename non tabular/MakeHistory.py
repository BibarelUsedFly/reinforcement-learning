import numpy as np
np.set_printoptions(precision=2)
from auxiliary import optimalize
from DemonModel import action_value, get_actions, is_terminal, take_action

def greedy_policy(state, actions, weights):
    '''Elige una acción de acuerdo a una política greedy'''
    action_scores = optimalize(
        np.array([action_value(state, act, weights) for act in actions]))
    return np.random.choice(actions, p=action_scores)


def make_history(initial_state, weights):
    ## S_0, A_0
    history = []
    R = 0.0 ## Initial reward
    S = initial_state
    A = greedy_policy(S, get_actions(S), weights)
    history.append((R, S, A)) ## Tupla inicial RSA
    while not is_terminal(S): ## Mientras dure el episodio
        R, S1 = take_action(S, A)
        if is_terminal(S1):
            history.append((R, S1, None)) ## Tupla final RSA
            break ## Episode end
        A1 = greedy_policy(S1, get_actions(S1), weights)
        history.append((R, S1, A1))
        S, A = S1, A1
    return history

def print_weights(actions, weights):
    print(" "*14, (" "*6).join(actions[:4]), " "*4, (" "*5).join(actions[4:]), end='')
    print(
    '''
    DDistX:  {0:5.2f}  {8:5.2f}  {16:5.2f}  {24:5.2f}  {32:5.2f}  {40:5.2f}  {48:5.2f}  {56:5.2f}
    DDistY:  {1:5.2f}  {9:5.2f}  {17:5.2f}  {25:5.2f}  {33:5.2f}  {41:5.2f}  {49:5.2f}  {57:5.2f}
    H1DistX: {2:5.2f}  {10:5.2f}  {18:5.2f}  {26:5.2f}  {34:5.2f}  {42:5.2f}  {50:5.2f}  {58:5.2f}
    H1DistY: {3:5.2f}  {11:5.2f}  {19:5.2f}  {27:5.2f}  {35:5.2f}  {43:5.2f}  {51:5.2f}  {59:5.2f}
    H2DistX: {4:5.2f}  {12:5.2f}  {20:5.2f}  {28:5.2f}  {36:5.2f}  {44:5.2f}  {52:5.2f}  {60:5.2f}
    H2DistY: {5:5.2f}  {13:5.2f}  {21:5.2f}  {29:5.2f}  {37:5.2f}  {45:5.2f}  {53:5.2f}  {61:5.2f}
    GDistX:  {6:5.2f}  {14:5.2f}  {22:5.2f}  {30:5.2f}  {38:5.2f}  {46:5.2f}  {54:5.2f}  {62:5.2f}
    GDistY:  {7:5.2f}  {15:5.2f}  {23:5.2f}  {31:5.2f}  {39:5.2f}  {47:5.2f}  {55:5.2f}  {63:5.2f}
    '''.format(*weights)
    )

def print_weightsOLD(actions, weights):
    # print('''
    #          {0}  {1}  {2}  {3}  {4}  {5}  {6}  {7}
    # '''.format(*actions))
    print(" "*14, (" "*6).join(actions[:4]), " "*4, (" "*5).join(actions[4:]), end='')
    print(
    '''
    X:       {0:.2f}  {18:.2f}  {36:.2f}  {54:.2f}  {72:.2f}  {90:.2f}  {108:.2f}  {126:.2f}
    Y:       {1:.2f}  {19:.2f}  {37:.2f}  {55:.2f}  {73:.2f}  {91:.2f}  {109:.2f}  {127:.2f}
    DX:      {2:.2f}  {20:.2f}  {38:.2f}  {56:.2f}  {74:.2f}  {92:.2f}  {110:.2f}  {128:.2f}
    DY:      {3:.2f}  {21:.2f}  {39:.2f}  {57:.2f}  {75:.2f}  {93:.2f}  {111:.2f}  {129:.2f}
    ROLL:     {4:.2f}   {22:.2f}   {40:.2f}   {58:.2f}  {76:.2f}  {94:.2f}  {112:.2f}  {130:.2f}
    H1X:     {5:.2f}  {23:.2f}  {41:.2f}  {59:.2f}  {77:.2f}  {95:.2f}  {113:.2f}  {131:.2f}
    H1Y:     {6:.2f}  {24:.2f}  {42:.2f}  {60:.2f}  {78:.2f}  {96:.2f}  {114:.2f}  {132:.2f}
    H2X:     {7:.2f}  {25:.2f}  {43:.2f}  {61:.2f}  {79:.2f}  {97:.2f}  {115:.2f}  {133:.2f}
    H2Y:     {8:.2f}  {26:.2f}  {44:.2f}  {62:.2f}  {80:.2f}  {98:.2f}  {116:.2f}  {134:.2f}
    HP:      {9:.2f}  {27:.2f}  {45:.2f}  {63:.2f}  {81:.2f}  {99:.2f}  {117:.2f}  {135:.2f}
    DDistX:  {10:.2f}  {28:.2f}  {46:.2f}  {64:.2f}  {82:.2f}  {100:.2f}  {118:.2f}  {136:.2f}
    DDistY:  {11:.2f}  {29:.2f}  {47:.2f}  {65:.2f}  {83:.2f}  {101:.2f}  {119:.2f}  {137:.2f}
    H1DistX: {12:.2f}  {30:.2f}  {48:.2f}  {66:.2f}  {84:.2f}  {102:.2f}  {120:.2f}  {138:.2f}
    H1DistY: {13:.2f}  {31:.2f}  {49:.2f}  {67:.2f}  {85:.2f}  {103:.2f}  {121:.2f}  {139:.2f}
    H2DistX: {14:.2f}  {32:.2f}  {50:.2f}  {68:.2f}  {86:.2f}  {104:.2f}  {122:.2f}  {140:.2f}
    H2DistY: {15:.2f}  {33:.2f}  {51:.2f}  {69:.2f}  {87:.2f}  {105:.2f}  {123:.2f}  {141:.2f}
    GDistX:  {16:.2f}  {34:.2f}  {52:.2f}  {70:.2f}  {88:.2f}  {106:.2f}  {124:.2f}  {142:.2f}
    GDistY:  {17:.2f}  {35:.2f}  {53:.2f}  {71:.2f}  {89:.2f}  {107:.2f}  {125:.2f}  {143:.2f}
    '''.format(*weights)
    )