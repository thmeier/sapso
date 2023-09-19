import numpy as np
import matplotlib.pyplot as plt

# OPTIONS
INTERACTIVE = False
INTERPOLATION = 'exp' # 'cubic' # 'exp'
#INTERPOLATION = 'exp'
VERBOSE = False
NORMALIZE = True


def interpolate_cubic(points):
    """
    THIS INTERPOLATION METHOD REQUIRES THAT x-coord OF FIRST POINT IS 0.0
    WHILE THE LAST POINT HAS x-coord 1.0
    """

    coeffs = np.polyfit(*points.T, deg=3)
    xs = np.linspace(0, 1, 100)
    ys = np.sum([ c * xs ** (3-i) for i,c in enumerate(coeffs) ], axis=0)
    return xs, ys

def interpolate_exp(points):
    """
    THIS INTERPOLATION METHOD REQUIRES THAT x-coord OF FIRST POINT IS 0.0
    WHILE THE LAST POINT HAS x-coord 1.0
    """

    log_points = np.vstack([
        points[:, 0], np.log(np.maximum(points[:, 1], 2e-3))
    ]).T

    xs = np.linspace(0, 1, 100)
    ys = list()

    for i in range(3):
        _points = log_points[i:i+2]
        coeffs = np.polyfit(*_points.T, deg=1)
        lo, hi = _points[:, 0]
        x = xs[np.where((lo <= xs) & (xs <= hi))]
        y = np.exp(coeffs[0] * x + coeffs[1])
        ys.append(y)

        if VERBOSE:
            print(f'points      = ({points[i:i+2][0,0]:.2f}, {points[i:i+2][0,1]:.2f}), ({points[i:i+2][1,0]:.2f}, {points[i:i+2][1,1]:.2f})')
            print(f'points (ln) = ({_points[0,0]:.2f}, {_points[0,1]:.2f}), ({_points[1,0]:.2f}, {_points[1,1]:.2f})')
            print(f'coeffs      = [{coeffs[0]:.2f}, {coeffs[1]:.2f}]')
            print()


    ys = np.concatenate(ys)
    return xs, ys

def interpolate_const(points):
    xs = np.linspace(0, 1, 100)
    ys = list()

    for i in range(3):
        _points = points[i:i+2]
        coeffs = np.polyfit(*_points.T, deg=0)
        lo, hi = _points[:, 0]
        x = xs[np.where((lo <= xs) & (xs <= hi))]
        y = np.ones_like(x) * coeffs[0]
        ys.append(y)

    ys = np.concatenate(ys)
    return xs, ys

def plot_param_interpolation(xs, ys, points):
    plt.plot(xs, ys)
    plt.plot(*points.T, 'o', color='tab:orange')
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.0)
    plt.show()

def demo():
    if INTERACTIVE:
        plt.xlim(0.0, 1.0)
        plt.ylim(0.0, 1.0)
        plt.title('Choose 4 Points for Interpolation')
        points = np.array(plt.ginput(4))
        plt.close()
        print(f'points:\n{points}')

    else:
        points = np.array([
            (0.00, 0.70),
            (0.33, 0.33),
            (0.75, 0.50),
            (1.00, 0.25)
        ])

    if INTERPOLATION == 'cubic':
        xs, ys = interpolate_cubic(points)

    elif INTERPOLATION == 'exp':
        xs, ys = interpolate_exp(points)

    else:
        raise ValueError(
            f'Interpolation method {INTERPOLATION} not recognized!'
        )

    plot_param_interpolation(xs, ys, points)

if __name__ == '__main__':
    # demo()
    # exit(0)

    component1 = np.array([
        (0.00, 1.00),
        (0.33, 0.33),
        (0.67, 0.33),
        (1.00, 0.33)
    ])

    component2 = np.array([
        (0.00, 0.00),
        (0.33, 0.67),
        (0.67, 0.33),
        (1.00, 0.33)
    ])

    component3 = np.array([
        (0.00, 0.00),
        (0.33, 0.00),
        (0.67, 0.33),
        (1.00, 0.33)
    ])

    if INTERPOLATION == 'cubic':
        interpolate = interpolate_cubic

    elif INTERPOLATION == 'exp':
        interpolate = interpolate_exp

    elif INTERPOLATION == 'const':
        interpolate = interpolate_const

    components = [component1, component2, component3]

    # interpolate between points
    xs_ys = [ interpolate(c) for c in components ]

    # init plot
    fig, ax = plt.subplots(figsize=(6.5, 6.5))

    # mark phases
    for x in [0.00, 0.33, 0.67, 1.00]:
        ax.axvline(x, ls='dashed', color='darkgray')

    # ensure compoents sum to 1.0
    total = np.sum([ ys for (_, ys) in xs_ys ], axis=0)
    if NORMALIZE:
        xs_ys = [ (xs, ys/total) for (xs, ys) in xs_ys ]
        total = np.sum([ ys for (_, ys) in xs_ys ], axis=0)

    # plot sum of components
    ax.plot(xs_ys[0][0], total, label='sum', color='tab:red')

    # plot interpolated lines and points
    colors = ['tab:blue', 'tab:orange', 'tab:green']
    for i, ((xs, ys), points) in enumerate(zip(xs_ys, components)):
        print(f'component {i} in range [{np.min(ys):.3f}, {np.max(ys):.3f}]')

        ax.plot(xs, ys, label=f'component {i}', color=colors[i])
        ax.plot(*points.T, 'o', color='gray')

    # label phases in plot
    for x, phase in zip([1/6, 3/6, 5/6], ['phase 0', 'phase 1', 'phase 2']):
        ax.text(x, 0.9, phase, ha='center', va='center')

    # adjust plot
    #plt.gca().set(xlim=(-0.025, 1.025), ylim=(0.0, 1.025))
    ax.set(xlabel='iteration progress', ylabel='weight')
    ax.legend()

    plt.savefig(f'velocity_component_weighting__{INTERPOLATION}.pdf', bbox_inches='tight', dpi=600)

    plt.show()
