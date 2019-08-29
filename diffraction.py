"""Functions for wavefront propagation."""


def generate_probe(N, lam, dx, Ls, setup):
    """Generate the transmission function from APS beamlines.

    Parameters
    ----------
    N      -> number of pixels
    lam -> the wave length
    dx     -> pixel size (in sample plane)
    Ls     -> distance from focal plane to sample

    """
    # Fresnel Zone Plate
    if setup == "velo":
        Rn = 90e-6
        dRn = 50e-9
    elif setup == "barry":
        Rn = 80e-6
        dRn = 70e-9
    elif setup == "barry2":
        Rn = 70e-6
        dRn = 160e-9
    else:
        Rn = 90e-6
        dRn = 50e-9

    fl = 2 * Rn * dRn / lam  # focal length corresponding to central wavelength
    D_FZP = 180e-6  # dimeter of the FZP
    D_H = 60e-6  # central beamstop

    # pixel size on FZP plane
    dx_fzp = lam * fl / N / dx
    # Coordinate on FZP plane
    lx_fzp = linspace(-dx_fzp * N / 2, dx_fzp * N / 2, N)
    [x_fzp, y_fzp] = meshgrid(lx_fzp)
    # Transmission function of the FZP
    T = exp(-1j * 2 * pi / lam * (x_fzp**2 + y_fzp**2) / 2 / fl)
    C = double(
        sqrt(x_fzp**2 + y_fzp**2) <= (D_FZP / 2)
    )  # Cercular function of FZP
    H = double(sqrt(x_fzp**2 + y_fzp**2) >= (D_H / 2))  # cental block

    # probe on sample plane
    return fresnel_propagation(C * T * H, dx_fzp, (fl + Ls), lam)


def fresnel_propagation_ac(t, z, dx=1e-6, wavelength=1e-10):
    """
    Propagate a plane wave using the method of Adrian Carbajal-Dominguez et al.

    Carbajal-Domínguez, Adrián, Jorge Bernal Arroyo, Jesus E. Gomez Correa,
    and Gabriel Martínez Niconoff. "Numerical calculation of near field
    scalar diffraction using angular spectrum of plane waves theory and FFT."
    Revista mexicana de física E 56, no. 2 (2010): 159-164.
    https://arxiv.org/abs/1002.1999v1

    Parameters
    ----------
    t : complex 2D array
        the complex transmission function
    dx : float [m]
        the width of a square pixel on the transmission function grid
    z : float [m]
        the distance of the propagation
    wavelength : float [m]
        the wavelength of the light

    """
    M, N = t.shape
    # u, v are the locations of pixel centers of t
    u, v = np.meshgrid(
        np.linspace(dx * (1 - M) / 2, dx * (1 + M) / 2, M, endpoint=False),
        np.linspace(dx * (1 - N) / 2, dx * (1 + N) / 2, N, endpoint=False),
        indexing='ij',
    )

    # p is the propagation direction. See Eq. 9
    p = np.empty_like(t)
    k = 1 / wavelength  # wavenumber / 2 / PI
    case1 = k**2 > u**2 + v**2
    p[case1] = np.sqrt(k**2 - u[case1]**2 - v[case1]**2)
    case2 = k**2 < u**2 + v**2
    p[case2] = 1j * np.sqrt(u[case2]**2 + v[case2]**2 - k**2)

    # See Eq. 17
    A = np.fft.fftshift(np.fft.fft2(t))

    # See Eq. 18
    G = np.exp(1j * 2 * np.pi * z * p)

    return np.ifft2(A * G)
