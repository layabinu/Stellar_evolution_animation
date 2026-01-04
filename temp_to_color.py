import numpy as np
import h5py as h5
from pathlib import Path

# ------------------------------
# Paths
# ------------------------------
BASE_DIR = Path(__file__).parent
FRAMES_NPZ = BASE_DIR / "frames_data.npz"
HDF5_PATH = BASE_DIR / "BSE_Detailed_Output_0.h5"
OUTPUT_NPZ = BASE_DIR / "frames_data.npz"


import math
import numpy
import ciexyz
import colormodels

# Physical constants in mks units
PLANCK_CONSTANT = 6.6237e-34      # J-sec
SPEED_OF_LIGHT = 2.997925e+08    # m/sec
BOLTZMAN_CONSTANT = 1.3802e-23      # J/K
SUN_TEMPERATURE = 5778.0          # K


def blackbody_specific_intensity(wl_nm, T_K):
    '''Get the monochromatic specific intensity for a blackbody -
        wl_nm = wavelength [nm]
        T_K   = temperature [K]
    This is the energy radiated per second per unit wavelength per unit solid angle.
    Reference - Shu, eq. 4.6, p. 78.'''
    # precalculations that could be made global
    a = (PLANCK_CONSTANT * SPEED_OF_LIGHT) / (BOLTZMAN_CONSTANT)
    b = (2.0 * PLANCK_CONSTANT * SPEED_OF_LIGHT * SPEED_OF_LIGHT)
    wl_m = wl_nm * 1.0e-9
    try:
        exponent = a / (wl_m * T_K)
    except ZeroDivisionError:
        # treat same as large exponent
        return 0.0
    if exponent > 500.0:
        # so large that the final result is nearly zero - avoid the giant intermediate
        return 0.0
    specific_intensity = b / (math.pow(wl_m, 5) * (math.exp(exponent) - 1.0))
    return specific_intensity


def blackbody_spectrum(T_K):
    '''Get the spectrum of a blackbody, as a numpy array.'''
    spectrum = ciexyz.empty_spectrum()
    (num_rows, num_cols) = spectrum.shape
    for i in range(0, num_rows):
        specific_intensity = blackbody_specific_intensity(spectrum[i][0], T_K)
        # scale by size of wavelength interval
        spectrum[i][1] = specific_intensity * ciexyz.delta_wl_nm * 1.0e-9
    return spectrum


def blackbody_color(T_K):
    '''Given a temperature (K), return the xyz color of a thermal blackbody.'''
    spectrum = blackbody_spectrum(T_K)
    xyz = ciexyz.xyz_from_spectrum(spectrum)
    return xyz

import math
import numpy

# The xyz constructors have some special versions to handle some common situations


def xyz_color(x, y, z=None):
    '''Construct an xyz color.  If z is omitted, set it so that x+y+z = 1.0.'''
    if z == None:
        # choose z so that x+y+z = 1.0
        z = 1.0 - (x + y)
    rtn = numpy.array([x, y, z])
    return rtn


def xyz_normalize(xyz):
    '''Scale so that all values add to 1.0.
    This both modifies the passed argument and returns the normalized result.'''
    sum_xyz = xyz[0] + xyz[1] + xyz[2]
    if sum_xyz != 0.0:
        scale = 1.0 / sum_xyz
        xyz[0] *= scale
        xyz[1] *= scale
        xyz[2] *= scale
    return xyz


def xyz_normalize_Y1(xyz):
    '''Scale so that the y component is 1.0.
    This both modifies the passed argument and returns the normalized result.'''
    if xyz[1] != 0.0:
        scale = 1.0 / xyz[1]
        xyz[0] *= scale
        xyz[1] *= scale
        xyz[2] *= scale
    return xyz


def xyz_color_from_xyY(x, y, Y):
    '''Given the 'little' x,y chromaticity, and the intensity Y,
    construct an xyz color.  See Foley/Van Dam p. 581, eq. 13.21.'''
    return xyz_color(
        (x/y) * Y,
        Y,
        (1.0-x-y)/(y) * Y)

# Simple constructors for the remaining models


def rgb_color(r, g, b):
    '''Construct a linear rgb color from components.'''
    rtn = numpy.array([r, g, b])
    return rtn


def irgb_color(ir, ig, ib):
    '''Construct a displayable integer irgb color from components.'''
    rtn = numpy.array([ir, ig, ib], int)
    return rtn


def luv_color(L, u, v):
    '''Construct a Luv color from components.'''
    rtn = numpy.array([L, u, v])
    return rtn


def lab_color(L, a, b):
    '''Construct a Lab color from components.'''
    rtn = numpy.array([L, a, b])
    return rtn

#
# Definitions of some standard values for colors and conversions
#

# Chromaticities of various standard phosphors and white points.


# sRGB (ITU-R BT.709) standard phosphor chromaticities
SRGB_Red = xyz_color(0.640, 0.330)
SRGB_Green = xyz_color(0.300, 0.600)
SRGB_Blue = xyz_color(0.150, 0.060)
SRGB_White = xyz_color(0.3127, 0.3290)  # D65

# HDTV standard phosphors, from Poynton [Color FAQ] p. 9
#   These are claimed to be similar to typical computer monitors
HDTV_Red = xyz_color(0.640, 0.330)
HDTV_Green = xyz_color(0.300, 0.600)
HDTV_Blue = xyz_color(0.150, 0.060)
# use D65 as white point for HDTV

# SMPTE phosphors
#   However, Hall [p. 188] notes that TV expects values calibrated for NTSC
#   even though actual phosphors are as below.
# From Hall p. 118, and Kasson p. 400
SMPTE_Red = xyz_color(0.630, 0.340)
SMPTE_Green = xyz_color(0.310, 0.595)
SMPTE_Blue = xyz_color(0.155, 0.070)
# use D65 as white point for SMPTE

# NTSC phosphors [original standard for TV, but no longer used in TV sets]
# From Hall p. 119 and Foley/Van Dam p. 589
NTSC_Red = xyz_color(0.670, 0.330)
NTSC_Green = xyz_color(0.210, 0.710)
NTSC_Blue = xyz_color(0.140, 0.080)
# use D65 as white point for NTSC

# Typical short persistence phosphors from Foley/Van Dam p. 583
FoleyShort_Red = xyz_color(0.61, 0.35)
FoleyShort_Green = xyz_color(0.29, 0.59)
FoleyShort_Blue = xyz_color(0.15, 0.063)

# Typical long persistence phosphors from Foley/Van Dam p. 583
FoleyLong_Red = xyz_color(0.62, 0.33)
FoleyLong_Green = xyz_color(0.21, 0.685)
FoleyLong_Blue = xyz_color(0.15, 0.063)

# Typical TV phosphors from Judd/Wyszecki p. 239
Judd_Red = xyz_color(0.68, 0.32)       # Europium Yttrium Vanadate
Judd_Green = xyz_color(0.28, 0.60)       # Zinc Cadmium Sulfide
Judd_Blue = xyz_color(0.15, 0.07)       # Zinc Sulfide

# White points [all are for CIE 1931 for small field of view]
#   These are from Judd/Wyszecki
WhiteA = xyz_color(0.4476, 0.4074)      # approx 2856 K
WhiteB = xyz_color(0.3484, 0.3516)      # approx 4874 K
WhiteC = xyz_color(0.3101, 0.3162)      # approx 6774 K
WhiteD55 = xyz_color(0.3324, 0.3475)      # approx 5500 K
WhiteD65 = xyz_color(0.3127, 0.3290)      # approx 6500 K
WhiteD75 = xyz_color(0.2990, 0.3150)      # approx 7500 K

# Blackbody white points [this empirically gave good results]
Blackbody6500K = xyz_color(0.3135, 0.3237)
Blackbody6600K = xyz_color(0.3121, 0.3223)
Blackbody6700K = xyz_color(0.3107, 0.3209)
Blackbody6800K = xyz_color(0.3092, 0.3194)
Blackbody6900K = xyz_color(0.3078, 0.3180)
Blackbody7000K = xyz_color(0.3064, 0.3166)

# MacBeth Color Checker white patch
#   Using this as white point will force MacBeth chart entry to equal machine RGB
MacBethWhite = xyz_color(0.30995, 0.31596, 0.37409)

# Also see Judd/Wyszecki p.164 for colors of Planck Blackbodies

# Some standard xyz/rgb conversion matricies, which assume particular phosphors.
# These are useful for testing.

# sRGB, from http://www.color.org/sRGB.xalter
srgb_rgb_from_xyz_matrix = numpy.array([
    [3.2410, -1.5374, -0.4986],
    [-0.9692,  1.8760,  0.0416],
    [0.0556, -0.2040,  1.0570]
])

# SMPTE conversions, from Kasson p. 400
smpte_xyz_from_rgb_matrix = numpy.array([
    [0.3935, 0.3653, 0.1916],
    [0.2124, 0.7011, 0.0865],
    [0.0187, 0.1119, 0.9582]
])
smpte_rgb_from_xyz_matrix = numpy.array([
    [3.5064, -1.7400, -0.5441],
    [-1.0690,  1.9777,  0.0352],
    [0.0563, -0.1970,  1.0501]
])

#
# Conversions between CIE XYZ and RGB colors.
#     Assumptions must be made about the specific device to construct the conversions.
#

# public - xyz colors of the monitor phosphors (and full white)
PhosphorRed = None
PhosphorGreen = None
PhosphorBlue = None
PhosphorWhite = None

rgb_from_xyz_matrix = None
xyz_from_rgb_matrix = None


def init(
        phosphor_red=SRGB_Red,
        phosphor_green=SRGB_Green,
        phosphor_blue=SRGB_Blue,
        white_point=SRGB_White):
    '''Setup the conversions between CIE XYZ and linear RGB spaces.
    Also do other initializations (gamma, conversions with Luv and Lab spaces, clipping model).

    The default arguments correspond to the sRGB standard RGB space.

    The conversion is defined by supplying the chromaticities of each of
    the monitor phosphors, as well as the resulting white color when all
    of the phosphors are at full strength.

    See [Foley/Van Dam, p.587, eqn 13.27, 13.29] and [Hall, p. 239].
    '''
    global PhosphorRed, PhosphorGreen, PhosphorBlue, PhosphorWhite
    PhosphorRed = phosphor_red
    PhosphorGreen = phosphor_green
    PhosphorBlue = phosphor_blue
    PhosphorWhite = white_point
    global xyz_from_rgb_matrix, rgb_from_xyz_matrix
    phosphor_matrix = numpy.column_stack(
        (phosphor_red, phosphor_green, phosphor_blue))
    # normalize white point to Y=1.0
    normalized_white = white_point.copy()
    xyz_normalize_Y1(normalized_white)
    # Determine intensities of each phosphor by solving:
    #     phosphor_matrix * intensity_vector = white_point
    intensities = numpy.linalg.solve(phosphor_matrix, normalized_white)
    # construct xyz_from_rgb matrix from the results
    xyz_from_rgb_matrix = numpy.column_stack(
        (phosphor_red * intensities[0],
         phosphor_green * intensities[1],
         phosphor_blue * intensities[2]))
    # invert to get rgb_from_xyz matrix
    rgb_from_xyz_matrix = numpy.linalg.inv(xyz_from_rgb_matrix)
    #  print('xyz_from_rgb', str (xyz_from_rgb_matrix))
    #  print('rgb_from_xyz', str (rgb_from_xyz_matrix))

    # conversions between the (almost) perceptually uniform
    # spaces (Luv, Lab) require the definition of a white point.
    init_Luv_Lab_white_point(white_point)

    # init gamma correction functions to default
    init_gamma_correction()

    # init color clipping method to default
    init_clipping()


def rgb_from_xyz(xyz):
    '''Convert an xyz color to rgb.'''
    return numpy.dot(rgb_from_xyz_matrix, xyz)


def xyz_from_rgb(rgb):
    '''Convert an rgb color to xyz.'''
    return numpy.dot(xyz_from_rgb_matrix, rgb)

#
# Color model conversions to (nearly) perceptually uniform spaces Luv and Lab.
#

# Luv/Lab conversions depend on the specification of a white point.


_reference_white = None
_reference_u_prime = None
_reference_v_prime = None


def init_Luv_Lab_white_point(white_point):
    '''Specify the white point to use for Luv/Lab conversions.'''
    global _reference_white, _reference_u_prime, _reference_v_prime
    _reference_white = white_point.copy()
    xyz_normalize_Y1(_reference_white)
    (_reference_u_prime, _reference_v_prime) = uv_primes(_reference_white)

# Luminance function [of Y value of an XYZ color] used in Luv and Lab. See [Kasson p.399] for details.
# The linear range coefficient L_LUM_C has more digits than in the paper,
# this makes the function more continuous over the boundary.


L_LUM_A = 116.0
L_LUM_B = 16.0
L_LUM_C = 903.29629551307664
L_LUM_CUTOFF = 0.008856


def L_luminance(y):
    '''L coefficient for Luv and Lab models.'''
    if y > L_LUM_CUTOFF:
        return L_LUM_A * math.pow(y, 1.0/3.0) - L_LUM_B
    else:
        # linear range
        return L_LUM_C * y


def L_luminance_inverse(L):
    '''Inverse of L_luminance().'''
    if L <= (L_LUM_C * L_LUM_CUTOFF):
        # linear range
        y = L / L_LUM_C
    else:
        t = (L + L_LUM_B) / L_LUM_A
        y = math.pow(t, 3)
    return y

# Utility function for Luv


def uv_primes(xyz):
    '''Luv utility.'''
    x = xyz[0]
    y = xyz[1]
    z = xyz[2]
    w_denom = x + 15.0 * y + 3.0 * z
    if w_denom != 0.0:
        u_prime = 4.0 * x / w_denom
        v_prime = 9.0 * y / w_denom
    else:
        # this should only happen when x=y=z=0 [i.e. black] since xyz values are positive
        u_prime = 0.0
        v_prime = 0.0
    return (u_prime, v_prime)


def uv_primes_inverse(u_prime, v_prime, y):
    '''Inverse of form_uv_primes(). We will always have y known when this is called.'''
    if v_prime != 0.0:
        # normal
        w_denom = (9.0 * y) / v_prime
        x = 0.25 * u_prime * w_denom
        y = y
        z = (w_denom - x - 15.0 * y) / 3.0
    else:
        # should only happen when color is totally black
        x = 0.0
        y = 0.0
        z = 0.0
    xyz = xyz_color(x, y, z)
    return xyz

# Utility function for Lab
#     See [Kasson p.399] for details.
#     The linear range coefficient has more digits than in the paper,
#     this makes the function more continuous over the boundary.


LAB_F_A = 7.7870370302851422
LAB_F_B = (16.0/116.0)
# same cutoff as L_luminance()


def Lab_f(t):
    '''Lab utility function.'''
    if t > L_LUM_CUTOFF:
        return math.pow(t, 1.0/3.0)
    else:
        # linear range
        return LAB_F_A * t + LAB_F_B


def Lab_f_inverse(F):
    '''Inverse of Lab_f().'''
    if F <= (LAB_F_A * L_LUM_CUTOFF + LAB_F_B):
        # linear range
        t = (F - LAB_F_B) / LAB_F_A
    else:
        t = math.pow(F, 3)
    return t

# Conversions between standard device independent color space (CIE XYZ)
# and the almost perceptually uniform space Luv.


def luv_from_xyz(xyz):
    '''Convert CIE XYZ to Luv.'''
    x = xyz[0]
    y = xyz[1]
    z = xyz[2]
    # actually reference_white [1] is probably always 1.0
    y_p = y / _reference_white[1]
    (u_prime, v_prime) = uv_primes(xyz)
    L = L_luminance(y_p)
    u = 13.0 * L * (u_prime - _reference_u_prime)
    v = 13.0 * L * (v_prime - _reference_v_prime)
    luv = luv_color(L, u, v)
    return luv


def xyz_from_luv(luv):
    '''Convert Luv to CIE XYZ.  Inverse of luv_from_xyz().'''
    L = luv[0]
    u = luv[1]
    v = luv[2]
    # invert L_luminance() to get y
    y = L_luminance_inverse(L)
    if L != 0.0:
        # color is not totally black
        # get u_prime, v_prime
        L13 = 13.0 * L
        u_prime = _reference_u_prime + (u / L13)
        v_prime = _reference_v_prime + (v / L13)
        # get xyz color
        xyz = uv_primes_inverse(u_prime, v_prime, y)
    else:
        # color is black
        xyz = xyz_color(0.0, 0.0, 0.0)
    return xyz

# Conversions between standard device independent color space (CIE XYZ)
# and the almost perceptually uniform space Lab.


def lab_from_xyz(xyz):
    '''Convert color from CIE XYZ to Lab.'''
    x = xyz[0]
    y = xyz[1]
    z = xyz[2]

    x_p = x / _reference_white[0]
    y_p = y / _reference_white[1]
    z_p = z / _reference_white[2]

    f_x = Lab_f(x_p)
    f_y = Lab_f(y_p)
    f_z = Lab_f(z_p)

    L = L_luminance(y_p)
    a = 500.0 * (f_x - f_y)
    b = 200.0 * (f_y - f_z)
    Lab = lab_color(L, a, b)
    return Lab


def xyz_from_lab(Lab):
    '''Convert color from Lab to CIE XYZ.  Inverse of lab_from_xyz().'''
    L = Lab[0]
    a = Lab[1]
    b = Lab[2]
    # invert L_luminance() to get y_p
    y_p = L_luminance_inverse(L)
    # calculate f_y
    f_y = Lab_f(y_p)
    # solve for f_x and f_z
    f_x = f_y + (a / 500.0)
    f_z = f_y - (b / 200.0)
    # invert Lab_f() to get x_p and z_p
    x_p = Lab_f_inverse(f_x)
    z_p = Lab_f_inverse(f_z)
    # multiply by reference white to get xyz
    x = x_p * _reference_white[0]
    y = y_p * _reference_white[1]
    z = z_p * _reference_white[2]
    xyz = xyz_color(x, y, z)
    return xyz

# Gamma correction
#
# Non-gamma corrected rgb values, also called non-linear rgb values,
# correspond to palette register entries [although here they are kept
# in the range 0.0 to 1.0.]  The numerical values are not proportional
# to the amount of light energy present.
#
# Gamma corrected rgb values, also called linear rgb values,
# do not correspond to palette entries.  The numerical values are
# proportional to the amount of light energy present.
#
# This effect is particularly significant with CRT displays.
# With LCD displays, it is less clear (at least to me), what the genuinely
# correct correction should be.


# Gamma correction functions
display_from_linear_component = None
linear_from_display_component = None
gamma_exponent = None

# sRGB standard effective gamma.  This exponent is not applied explicitly.
STANDARD_GAMMA = 2.2

# Although NTSC specifies a gamma of 2.2 as standard, this is designed
# to account for the dim viewing environments typical of TV, but not
# computers.  Well-adjusted CRT displays have a true gamma in the range
# 2.35 through 2.55.  We use the physical gamma value here, not 2.2,
# thus not correcting for a dim viewing environment.
# [Poynton, Gamma FAQ p.5, p.9, Hall, p. 121]
POYNTON_GAMMA = 2.45

# Simple power laws for gamma correction


def simple_gamma_invert(x):
    '''Simple power law for gamma inverse correction.'''
    if x <= 0.0:
        return x
    else:
        return math.pow(x, 1.0 / gamma_exponent)


def simple_gamma_correct(x):
    '''Simple power law for gamma correction.'''
    if x <= 0.0:
        return x
    else:
        return math.pow(x, gamma_exponent)

# sRGB gamma correction - http://www.color.org/sRGB.xalter
# The effect of the equations is to closely fit a straightforward
# gamma 2.2 curve with an slight offset to allow for invertability in
# integer math. Therefore, we are maintaining consistency with the
# gamma 2.2 legacy images and the video industry.


def srgb_gamma_invert(x):
    '''sRGB standard for gamma inverse correction.'''
    if x <= 0.00304:
        rtn = 12.92 * x
    else:
        rtn = 1.055 * math.pow(x, 1.0/2.4) - 0.055
    return rtn


def srgb_gamma_correct(x):
    '''sRGB standard for gamma correction.'''
    if x <= 0.03928:
        rtn = x / 12.92
    else:
        rtn = math.pow((x + 0.055) / 1.055, 2.4)
    return rtn


def init_gamma_correction(
        display_from_linear_function=srgb_gamma_invert,
        linear_from_display_function=srgb_gamma_correct,
        gamma=STANDARD_GAMMA):
    '''Setup gamma correction.
    The functions used for gamma correction/inversion can be specified,
    as well as a gamma value.

    The specified display_from_linear_function should convert a
    linear (rgb) component [proportional to light intensity] into
    displayable component [proportional to palette values].

    The specified linear_from_display_function should convert a
    displayable (rgb) component [proportional to palette values]
    into a linear component [proportional to light intensity].

    The choices for the functions:
    display_from_linear_function -
        srgb_gamma_invert [default] - sRGB standard 
        simple_gamma_invert - simple power function, can specify gamma.
    linear_from_display_function -
        srgb_gamma_correct [default] - sRGB standard
        simple_gamma_correct - simple power function, can specify gamma.

    The gamma parameter is only used for the simple() functions,
    as sRGB implies an effective gamma of 2.2.'''
    global display_from_linear_component, linear_from_display_component, gamma_exponent
    display_from_linear_component = display_from_linear_function
    linear_from_display_component = linear_from_display_function
    gamma_exponent = gamma

#
# Color clipping - Physical color values may exceed the what the display can show,
#   either because the color is too pure (indicated by negative rgb values), or
#   because the color is too bright (indicated by rgb values > 1.0).
#   These must be clipped to something displayable.
#


_clip_method = None

# possible color clipping methods
CLIP_CLAMP_TO_ZERO = 0
CLIP_ADD_WHITE = 1


def init_clipping(clip_method=CLIP_ADD_WHITE):
    '''Specify the color clipping method.'''
    global _clip_method
    _clip_method = clip_method


def clip_rgb_color(rgb_color):
    '''Convert a linear rgb color (nominal range 0.0 - 1.0), into a displayable
    irgb color with values in the range (0 - 255), clipping as necessary.

    The return value is a tuple, the first element is the clipped irgb color,
    and the second element is a tuple indicating which (if any) clipping processes were used.
    '''
    clipped_chromaticity = False
    clipped_intensity = False

    rgb = rgb_color.copy()

    # clip chromaticity if needed (negative rgb values)
    if _clip_method == CLIP_CLAMP_TO_ZERO:
        # set negative rgb values to zero
        if rgb[0] < 0.0:
            rgb[0] = 0.0
            clipped_chromaticity = True
        if rgb[1] < 0.0:
            rgb[1] = 0.0
            clipped_chromaticity = True
        if rgb[2] < 0.0:
            rgb[2] = 0.0
            clipped_chromaticity = True
    elif _clip_method == CLIP_ADD_WHITE:
        # add enough white to make all rgb values nonnegative
        # find max negative rgb (or 0.0 if all non-negative), we need that much white
        rgb_min = min(0.0, min(rgb))
        # get max positive component
        rgb_max = max(rgb)
        # get scaling factor to maintain max rgb after adding white
        scaling = 1.0
        if rgb_max > 0.0:
            scaling = rgb_max / (rgb_max - rgb_min)
        # add enough white to cancel this out, maintaining the maximum of rgb
        if rgb_min < 0.0:
            rgb[0] = scaling * (rgb[0] - rgb_min)
            rgb[1] = scaling * (rgb[1] - rgb_min)
            rgb[2] = scaling * (rgb[2] - rgb_min)
            clipped_chromaticity = True
    else:
        raise(ValueError, 'Invalid color clipping method %s' %
              (str(_clip_method)))

    # clip intensity if needed (rgb values > 1.0) by scaling
    rgb_max = max(rgb)
    # we actually don't overflow until 255.0 * intensity > 255.5, so instead of 1.0 use ...
    intensity_cutoff = 1.0 + (0.5 / 255.0)
    if rgb_max > intensity_cutoff:
        # must scale intensity, so max value is intensity_cutoff
        scaling = intensity_cutoff / rgb_max
        rgb *= scaling
        clipped_intensity = True

    # gamma correction
    for index in range(0, 3):
        rgb[index] = display_from_linear_component(rgb[index])

    # scale to 0 - 255
    ir = round(255.0 * rgb[0])
    ig = round(255.0 * rgb[1])
    ib = round(255.0 * rgb[2])
    # ensure that values are in the range 0-255
    ir = min(255, max(0, ir))
    ig = min(255, max(0, ig))
    ib = min(255, max(0, ib))
    irgb = irgb_color(ir, ig, ib)
    return (irgb, (clipped_chromaticity, clipped_intensity))

#
# Conversions between linear rgb colors (range 0.0 - 1.0, values proportional to light intensity)
# and displayable irgb colors (range 0 - 255, values corresponding to hardware palette values).
#
# Displayable irgb colors can be represented as hex strings, like '#AB05B4'.
#


def irgb_string_from_irgb(irgb):
    '''Convert a displayable irgb color (0-255) into a hex string.'''
    # ensure that values are in the range 0-255
    for index in range(0, 3):
        irgb[index] = min(255, max(0, irgb[index]))
    # convert to hex string
    irgb_string = '#%02X%02X%02X' % (irgb[0], irgb[1], irgb[2])
    return irgb_string


def irgb_from_irgb_string(irgb_string):
    '''Convert a color hex string (like '#AB13D2') into a displayable irgb color.'''
    strlen = len(irgb_string)
    if strlen != 7:
        raise(ValueError, 'irgb_string_from_irgb(): Expecting 7 character string like #AB13D2')
    if irgb_string[0] != '#':
        raise(ValueError, 'irgb_string_from_irgb(): Expecting 7 character string like #AB13D2')
    irs = irgb_string[1:3]
    igs = irgb_string[3:5]
    ibs = irgb_string[5:7]
    ir = int(irs, 16)
    ig = int(igs, 16)
    ib = int(ibs, 16)
    irgb = irgb_color(ir, ig, ib)
    return irgb


def irgb_from_rgb(rgb):
    '''Convert a (linear) rgb value (range 0.0 - 1.0) into a 0-255 displayable integer irgb value (range 0 - 255).'''
    result = clip_rgb_color(rgb)
    (irgb, (clipped_chrom, clipped_int)) = result
    return irgb


def rgb_from_irgb(irgb):
    '''Convert a displayable (gamma corrected) irgb value (range 0 - 255) into a linear rgb value (range 0.0 - 1.0).'''
    # scale to 0.0 - 1.0
    r0 = float(irgb[0]) / 255.0
    g0 = float(irgb[1]) / 255.0
    b0 = float(irgb[2]) / 255.0
    # gamma adjustment
    r = linear_from_display_component(r0)
    g = linear_from_display_component(g0)
    b = linear_from_display_component(b0)
    rgb = rgb_color(r, g, b)
    return rgb


def irgb_string_from_rgb(rgb):
    '''Clip the rgb color, convert to a displayable color, and convert to a hex string.'''
    return irgb_string_from_irgb(irgb_from_rgb(rgb))

# Multi-level conversions, for convenience


def irgb_from_xyz(xyz):
    '''Convert an xyz color directly into a displayable irgb color.'''
    return irgb_from_rgb(rgb_from_xyz(xyz))


def irgb_string_from_xyz(xyz):
    '''Convert an xyz color directly into a displayable irgb color hex string.'''
    return irgb_string_from_rgb(rgb_from_xyz(xyz))

#
# Initialization - Initialize to sRGB at module startup.
#   If a different rgb model is needed, then the startup can be re-done to set the new conditions.
#


init()
# Default conversions setup on module load

# ------------------------------
# Load HDF5 temperature data
# ------------------------------
def load_hdf5_temperatures(path):
    f = h5.File(str(path), "r")

    mask = f["Record_Type"][()] == 4

    time = f["Time"][()][mask]
    teff1 = f["Teff(1)"][()][mask]
    teff2 = f["Teff(2)"][()][mask]

    f.close()
    return time, teff1, teff2

# ------------------------------
# Interpolation helper
# ------------------------------
def interp_from_hdf5(frame_time, hdf5_time, values):
    """
    Uses exact value if possible, otherwise linear interpolation.
    """
    if frame_time <= hdf5_time[0]:
        return float(values[0])
    if frame_time >= hdf5_time[-1]:
        return float(values[-1])

    i = np.searchsorted(hdf5_time, frame_time)

    if hdf5_time[i] == frame_time:
        return float(values[i])

    t0, t1 = hdf5_time[i-1], hdf5_time[i]
    v0, v1 = values[i-1], values[i]

    alpha = (frame_time - t0) / (t1 - t0)
    return (1 - alpha) * v0 + alpha * v1

# ------------------------------
# Temperature → displayable irgb
# ------------------------------
def temperature_to_rgb(T_K):
    """
    Convert effective temperature [K] into a displayable irgb color (0–255).

    Pipeline:
        Teff → blackbody spectrum → XYZ → linear RGB → gamma corrected irgb
    """
    # guard against invalid temperatures
    if T_K <= 0 or not np.isfinite(T_K):
        return irgb_color(0, 0, 0)

    # blackbody → XYZ
    xyz = blackbody_color(T_K)

    # XYZ → displayable irgb (includes clipping + gamma)
    irgb = irgb_from_xyz(xyz)

    return irgb

# ------------------------------
# Main augmentation
# ------------------------------
def add_temperatures_and_rgb():
    print("Loading frames...")
    frames = np.load(FRAMES_NPZ, allow_pickle=True)["frames"].tolist()

    print("Loading HDF5 temperatures...")
    hdf5_time, teff1, teff2 = load_hdf5_temperatures(HDF5_PATH)

    print("Assigning Teff and RGB values...")
    for f in frames:
        t = f["Time"]

        T1 = interp_from_hdf5(t, hdf5_time, teff1)
        T2 = interp_from_hdf5(t, hdf5_time, teff2)

        f["Teff1"] = float(T1)
        f["Teff2"] = float(T2)

        f["RGB1"] = temperature_to_rgb(T1)
        f["RGB2"] = temperature_to_rgb(T2)

    print(f"Saving augmented frames → {OUTPUT_NPZ}")
    np.savez_compressed(OUTPUT_NPZ, frames=frames)

    print("Done.")

# ------------------------------
# Entry point
# ------------------------------
if __name__ == "__main__":
    add_temperatures_and_rgb()



