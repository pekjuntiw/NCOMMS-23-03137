import numpy as np


def LC_neuron_spike(Vin):
    up = np.repeat(0., len(Vin))
    down = np.repeat(0., len(Vin))
    gout = np.repeat(0., len(Vin))
    gout2 = np.repeat(0., len(Vin))
    V_r = np.repeat(0., len(Vin))

    V_r[0] = Vin[0]

    # VO2 parameters
    Vth = 3.4
    Roff = 14000
    # voltage-dividing resistor
    Rs = 1500

    # voltage divider factor
    fenyaxishu = Roff/(Roff+Rs)

    # gain of intermediate op-amp
    beta = 200
    # delta increment
    delta = Vth / (beta * fenyaxishu)

    for i in range(1,len(Vin)):
        gout[i] = gout[i - 1] - (Vin[i] - Vin[i - 1])  # calculate output of op-amp 1
        gout2[i] = gout[i] * (-beta)  # calculate output of intermediate op-amp

        if gout2[i] * fenyaxishu >= Vth or gout2[i] * fenyaxishu <= -Vth:
            if gout2[i] * fenyaxishu >= Vth:
                up[i] = 1
                down[i] = 0
                gout[i] = gout[i] + delta  # reset gout
            elif gout2[i] * fenyaxishu <= -Vth:
                up[i] = 0
                down[i] = 1
                gout[i] = gout[i] - delta  # reset gout
        else:
            up[i] = 0
            down[i] = 0

    # signal reconstruction
    for i in range(1, len(Vin)):
        if up[i] == 1:
            V_r[i] = V_r[i - 1] + delta
        elif down[i] == 1:
            V_r[i] = V_r[i - 1] - delta

        else:
            V_r[i] = V_r[i - 1]

    return up, down, V_r, gout, gout2
