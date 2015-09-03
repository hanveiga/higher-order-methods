import numpy as np
import scipy
import scipy.optimize
import matplotlib.pyplot as plt

""" Analytical solutions live here """

class AnalyticalSolution(object):
    def __init__(self, function):
        self.function = function

    def get_analytical_solution(self, N, limits):
        np.linspace(limits[0],limits[1], N)

        values = self.function(np.linspace)

        return values

class ShockRodAnal(AnalyticalSolution):
    """ Using the code from https://gitlab.com/fantaz/Riemann_exact/tree/master """
    def __init__(self, initial_state, geometry, gamma = 1.4):
        self.left_state = initial_state[0]
        self.right_state = initial_state[1]

        self.geometry = geometry 

        self.gamma = gamma

    def get_analytical_solution(self, N, t):
        # solve function from fantaz
        """ returns the analytical value of the function at timestep t evaluated at N nodes
        with limits lim"""

        p_l, rho_l, u_l = self.left_state
        p_r, rho_r, u_r = self.right_state
        x_l, x_r, x_i = self.geometry

        # basic checking
        if x_l >= x_r:
            print('xl has to be less than xr!')
            exit()
        if x_i >= x_r or x_i <= x_l:
            print('xi has in between xl and xr!')
            exit()

        # calculate regions
        reg1, reg3, reg4, reg5, w = self.__calculate_regions(p_l, u_l, rho_l, p_r, u_r, rho_r)

        regions = self.__region_states(p_l, p_r, reg1, reg3, reg4, reg5)

        # calculate positions
        x_positions = self.__calc_positions(p_l, p_r, reg1, reg3, w, x_i, t)

        pos_description = ('Head of Rarefaction', 'Foot of Rarefaction',
                           'Contact Discontinuity', 'Shock')
        positions = dict(zip(pos_description, x_positions))

        # create arrays
        x, p, rho, u = self.__create_arrays(p_l, p_r, x_l, x_r, x_positions,
                                     reg1, reg3, reg4, reg5,
                                     N, t, x_i)

        val_names = ('x', 'p', 'rho', 'u')
        val_dict = dict(zip(val_names, (x, p, rho, u)))

        return val_dict

    def __shock_tube_function(self, p4, p1, p5, rho1, rho5):
        """
        Shock tube equation
        """
        z = (p4 / p5 - 1.)
        c1 = np.sqrt(self.gamma * p1 / rho1)
        c5 = np.sqrt(self.gamma * p5 / rho5)

        gm1 = self.gamma - 1.
        gp1 = self.gamma + 1.
        g2 = 2. * self.gamma

        fact = gm1 / g2 * (c5 / c1) * z / np.sqrt(1. + gp1 / g2 * z)
        fact = (1. - fact) ** (g2 / gm1)

        return p1 * fact - p4

    def __calculate_regions(self, p_l, u_l, rho_l, p_r, u_r, rho_r):
        """
        Compute regions
        :rtype : tuple
        :return: returns p, rho and u for regions 1,3,4,5 as well as the shock speed
        """
        rho1 = rho_l
        p1 = p_l
        u1 = u_l
        rho5 = rho_r
        p5 = p_r
        u5 = u_r

        if p_l < p_r:
            rho_1 = rho_r
            p1 = p_r
            u1 = u_r
            rho5 = rho_l
            p5 = p_l
            u5 = u_l

        # solve for post-shock pressure
        p4 = scipy.optimize.fsolve(self.__shock_tube_function, p1, (p1, p5, rho1, rho5))[0]

        # compute post-shock density and velocity
        z = (p4 / p5 - 1.)
        c5 = np.sqrt(self.gamma * p5 / rho5)

        gm1 = self.gamma - 1.
        gp1 = self.gamma + 1.
        gmfac1 = 0.5 * gm1 / self.gamma
        gmfac2 = 0.5 * gp1 / self.gamma

        fact = np.sqrt(1. + gmfac2 * z)

        u4 = c5 * z / (self.gamma * fact)
        rho4 = rho5 * (1. + gmfac2 * z) / (1. + gmfac1 * z)

        # shock speed
        w = c5 * fact

        # compute values at foot of rarefaction
        p3 = p4
        u3 = u4
        rho3 = rho1 * (p3 / p1)**(1. / self.gamma)

        return (p1, rho1, u1), (p3, rho3, u3), (p4, rho4, u4), (p5, rho5, u5), w

    def __calc_positions(self, pl, pr, region1, region3, w, xi, t):
        """
        :return: tuple of positions in the following order ->
                Head of Rarefaction: xhd,  Foot of Rarefaction: xft,
                Contact Discontinuity: xcd, Shock: xsh
        """
        p1, rho1 = region1[:2]  # don't need velocity
        p3, rho3, u3 = region3
        c1 = np.sqrt(self.gamma * p1 / rho1)
        c3 = np.sqrt(self.gamma * p3 / rho3)
        if pl > pr:
            xsh = xi + w * t
            xcd = xi + u3 * t
            xft = xi + (u3 - c3) * t
            xhd = xi - c1 * t
        else:
            # pr > pl
            xsh = xi - w * t
            xcd = xi - u3 * t
            xft = xi - (u3 - c3) * t
            xhd = xi + c1 * t

        return xhd, xft, xcd, xsh


    def __region_states(self, pl, pr, region1, region3, region4, region5):
        """
        :return: dictionary (region no.: p, rho, u), except for rarefaction region
        where the value is a string, obviously
        """
        if pl > pr:
            return {'Region 1': region1,
                    'Region 2': 'RAREFACTION',
                    'Region 3': region3,
                    'Region 4': region4,
                    'Region 5': region5}
        else:
            return {'Region 1': region5,
                    'Region 2': region4,
                    'Region 3': region3,
                    'Region 4': 'RAREFACTION',
                    'Region 5': region1}

    def __create_arrays(self, pl, pr, xl, xr, positions, state1, state3, state4, state5, N, t, xi):
        """
        :return: tuple of x, p, rho and u values across the domain of interest
        """
        xhd, xft, xcd, xsh = positions
        p1, rho1, u1 = state1
        p3, rho3, u3 = state3
        p4, rho4, u4 = state4
        p5, rho5, u5 = state5
        gm1 = self.gamma - 1.
        gp1 = self.gamma + 1.

        dx = (xr-xl)/(N-1)
        x_arr = np.arange(xl, xr, dx)
        rho = np.zeros(N, dtype=float)
        p = np.zeros(N, dtype=float)
        u = np.zeros(N, dtype=float)
        c1 = np.sqrt(self.gamma * p1 / rho1)
        if pl > pr:
            for i, x in enumerate(x_arr):
                if x < xhd:
                    rho[i] = rho1
                    p[i] = p1
                    u[i] = u1
                elif x < xft:
                    u[i] = 2. / gp1 * (c1 + (x - xi) / t)
                    fact = 1. - 0.5 * gm1 * u[i] / c1
                    rho[i] = rho1 * fact ** (2. / gm1)
                    p[i] = p1 * fact ** (2. * self.gamma / gm1)
                elif x < xcd:
                    rho[i] = rho3
                    p[i] = p3
                    u[i] = u3
                elif x < xsh:
                    rho[i] = rho4
                    p[i] = p4
                    u[i] = u4
                else:
                    rho[i] = rho5
                    p[i] = p5
                    u[i] = u5
        else:
            for i, x in enumerate(x_arr):
                if x < xsh:
                    rho[i] = rho5
                    p[i] = p5
                    u[i] = -u1
                elif x < xcd:
                    rho[i] = rho4
                    p[i] = p4
                    u[i] = -u4
                elif x < xft:
                    rho[i] = rho3
                    p[i] = p3
                    u[i] = -u3
                elif x < xhd:
                    u[i] = -2. / gp1 * (c1 + (xi - x) / t)
                    fact = 1. + 0.5 * gm1 * u[i] / c1
                    rho[i] = rho1 * fact ** (2. / gm1)
                    p[i] = p1 * fact ** (2. * self.gamma / gm1)
                else:
                    rho[i] = rho1
                    p[i] = p1
                    u[i] = -u1

        return x_arr, p, rho, u

if __name__ == '__main__':
    gamma = 1.4
    AnalyticalSol = ShockRodAnal(initial_state=[(1, 1, 0), (0.1, 0.125, 0.)],
                                       geometry=(0., 1., 0.5))
    values = AnalyticalSol.get_analytical_solution(N=500, t=0.2)

    # Finally, let's plot solutions
    p = values['p']
    rho = values['rho']
    u = values['u']

    # Energy and temperature
    E = p/(gamma-1.) + 0.5*u**2
    T = p/rho

    plt.figure(1)
    plt.plot(values['x'], p, linewidth=1.5, color='b')
    plt.ylabel('pressure')
    plt.xlabel('x')
    plt.axis([0, 1, 0, 1.1])

    plt.figure(2)
    plt.plot(values['x'], rho, linewidth=1.5, color='r')
    plt.ylabel('density')
    plt.xlabel('x')
    plt.axis([0, 1, 0, 1.1])

    plt.figure(3)
    plt.plot(values['x'], u, linewidth=1.5, color='g')
    plt.ylabel('velocity')
    plt.xlabel('x')

    plt.figure(4)
    plt.plot(values['x'], E, linewidth=1.5, color='k')
    plt.ylabel('Energy')
    plt.xlabel('x')
    plt.axis([0, 1, 0, 2.6])

    plt.figure(5)
    plt.plot(values['x'], T, linewidth=1.5, color='c')
    plt.ylabel('Temperature')
    plt.xlabel('x')
    plt.show()