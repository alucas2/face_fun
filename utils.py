from constants import *

def face_variables_to_str(fv):
    s = ""
    s += "M_open_x   {:.2f}".format(fv[MOUTH_OPEN_X]) + "\n"
    s += "M_open_y   {:.2f}".format(fv[MOUTH_OPEN_Y]) + "\n"
    s += "M_smile    {:.2f}".format(fv[MOUTH_SMILE]) + "\n"
    s += "M_pos_x    {:.2f}".format(fv[MOUTH_POSITION_X]) + "\n"
    s += "M_pos_y    {:.2f}".format(fv[MOUTH_POSITION_Y]) + "\n"
    s += "LEY_open_x {:.2f}".format(fv[LEFT_EYE_OPEN_X]) + "\n"
    s += "LEY_open_y {:.2f}".format(fv[LEFT_EYE_OPEN_Y]) + "\n"
    s += "REY_open_x {:.2f}".format(fv[RIGHT_EYE_OPEN_X]) + "\n"
    s += "REY_open_y {:.2f}".format(fv[RIGHT_EYE_OPEN_Y]) + "\n"
    s += "LEB_raise  {:.2f}".format(fv[LEFT_EYEBROW_RAISE]) + "\n"
    s += "REB_raise  {:.2f}".format(fv[RIGHT_EYEBROW_RAISE]) + "\n"
    return s

def smoothstep(x, x_min, x_max, y_min, y_max):
    if x < x_min:
        return y_min
    elif x > x_max:
        return y_max
    else:
        t = (x - x_min) / (x_max - x_min)
        return y_min + (3*t**2 - 2*t**3) * (y_max - y_min)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np

    x = np.linspace(-1, 1, 500)
    y = [smoothstep(xx, -0.1, 0.5, -0.75, 0.75) for xx in x]
    plt.plot(x, y)
    plt.show()