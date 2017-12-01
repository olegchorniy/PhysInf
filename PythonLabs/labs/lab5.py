from lorenz import modelling

S = 10
b = 8.0 / 3.0
r = 28

INITIAL = (1, 1, 1)
DT = 1e-3 * 2
TIME = 250

if __name__ == '__main__':
    x, y, z = modelling(S, b, r, INITIAL, TIME, DT)
    print x
    print y
    print z