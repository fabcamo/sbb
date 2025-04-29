import copy
import numpy as np
from shapely.geometry import Point

from geolib_plus.gef_cpt import GefCpt
from geolib_plus.robertson_cpt_interpretation import RobertsonCptInterpretation
from geolib_plus.robertson_cpt_interpretation import UnitWeightMethod, ShearWaveVelocityMethod, OCRMethod


def calc_qn(cpt: GefCpt) -> np.ndarray:
    """
    Calculate the qn value based on the CPT data.

    params:
        cpt (GefCpt): The CPT object with the interpreted data.

    returns:
        np.ndarray: The calculated qn value.
    """
    # Calculate qn
    qn = (cpt.qt - cpt.total_stress)
    # Set negative values to 0
    qn[qn < 0] = 0

    return qn


def calculate_vs_with_different_methods(cpt, interpreter):
    """
    Calculate Vs, G0, and E0 using different methods without modifying the original CPT.

    Args:
        cpt (GefCpt): The CPT object.
        interpreter (RobertsonCptInterpretation): The interpreter object.

    Returns:
        dict: Dictionary where keys are method names and values are dicts of vs, G0, E0.
    """
    vs_results = {}

    methods = {
        "Robertson": ShearWaveVelocityMethod.ROBERTSON,
        "Mayne": ShearWaveVelocityMethod.MAYNE,
        "Zang": ShearWaveVelocityMethod.ZANG,
        "Ahmed": ShearWaveVelocityMethod.AHMED,
    }

    for method_name, method_enum in methods.items():
        # Create a deep copy to avoid overwriting the original CPT
        cpt_copy = copy.deepcopy(cpt)

        interpreter.shearwavevelocitymethod = method_enum
        cpt_copy.interpret_cpt(interpreter)

        vs_results[method_name] = {
            "vs": cpt_copy.vs.copy(),
            "G0": cpt_copy.G0.copy(),
            "E0": cpt_copy.E0.copy(),
        }

    return vs_results


def calc_Bq(cpt: GefCpt) -> np.ndarray:
    """
    Calculate the Bq value based on the CPT data.

    params:
        cpt (GefCpt): The CPT object with the interpreted data.

    returns:
        np.ndarray: The calculated Bq value.
    """
    # Calculate Bq
    # Remember to multiply by 1000 to convert from MPa to kPa
    qn = calc_qn(cpt)
    qn_safe = np.maximum(qn, 1e-6)  # Avoid division by zero
    Bq = (cpt.pore_pressure_u2 * 1000 - cpt.hydro_pore_pressure) / qn_safe
    return Bq


def calc_Nkt(cpt: GefCpt) -> float:
    """
    Calculate Nkt based on Fr and Bq according to Robertson (2012) and Mayne & Peuchen (2022).

    Params:
        cpt (GefCpt): The CPT object with the interpreted data.

    Returns:
        Nkt_Fr (float): The calculated Nkt value based on Fr.
        Nkt_Bq (float): The calculated Nkt value based on Bq.
    """
    # 1. Calculate Nkt based on Fr (Robertson, 2012)
    # Make sure Fr is not zero or negative for log calculation
    Fr_safe = np.where(cpt.Fr <= 0, 0.01, cpt.Fr)  # Avoid log(0)
    Nkt_Fr = 10.5 + 7 * np.log10(Fr_safe)

    # 2. Calculate Nkt based on Bq (Mayne and Peuchen, 2022)
    # You need to calculate Bq first if it's not given
    Bq = calc_Bq(cpt)
    # Make sure Bq is not negative or zero for log
    Bq_safe = np.where(Bq + 0.1 <= 0, 0.01, Bq + 0.1)
    Nkt_Bq = 10.5 - 4.6 * np.log(Bq_safe)

    return Nkt_Fr, Nkt_Bq


def calc_Su(cpt: GefCpt, Nkt) -> float:
    """
    Calculate the undrained shear strength (Su) based on the Nkt and the cpt data.

    params:
        cpt (GefCpt): The CPT object with the interpreted data.
        Nkt (float): The Nkt value.

    returns:
        Su (float): The calculated undrained shear strength.
    """
    # Calculate Su
    Su = calc_qn(cpt) / Nkt

    return Su


def calc_St(cpt: GefCpt, Nkt) -> np.ndarray:
    """
    Correct calculation of Sensitivity-St using dynamic Nkt and fs.

    Params:
        cpt (GefCpt): The CPT object with the interpreted data.
        Nkt (array or scalar): Calculated Nkt.

    Returns:
        np.ndarray: Sensitivity values.
    """
    qn = calc_qn(cpt)
    fs = np.maximum(cpt.friction, 1e-6)  # Avoid zero or very small fs
    with np.errstate(divide='ignore', invalid='ignore'):
        St = qn / (Nkt * fs)

    return St


def calc_psi(cpt):
    """
    Calculate the Psi (state parameter) for liquefaction assessment from CPTu data.

    Params:
        cpt: CPT object with necessary attributes (qt, friction_nbr (Fr), total_stress, effective_stress, pore_pressure_u2, hydro_pore_pressure).

    Returns:
        psi: Array of Psi values (dimensionless).
    """

    # Safety: avoid division by zero later
    epsilon = 1e-6

    # Step 1: lambda from Fr (%)
    Fr_safe = np.where(cpt.friction_nbr <= 0, epsilon, cpt.friction_nbr)
    lambda_ = Fr_safe / 10

    # Step 2: m from lambda
    m = 11.9 + 13.3 * lambda_

    # Step 3: Estimate phi' from qt and effective stress
    qt_over_sigma_v_eff = np.where(cpt.effective_stress > epsilon, (cpt.qt / cpt.effective_stress), np.nan)
    phi_deg = np.degrees(np.arctan(0.1 + 0.38 * np.log10(np.maximum(qt_over_sigma_v_eff, epsilon))))
    phi_rad = np.radians(phi_deg)

    # Step 4: M from phi
    M = (6 * np.sin(phi_rad)) / (3 - np.sin(phi_rad))

    # Step 5: k from lambda and M
    k = (3 + (0.85 / np.maximum(lambda_, epsilon))) * M

    # Step 6: Qp
    Bq = calc_Bq(cpt)
    Qp = ((cpt.qt - cpt.total_stress) / (cpt.effective_stress + epsilon)) * (1 - Bq)

    # Step 7: Psi calculation
    with np.errstate(divide='ignore', invalid='ignore'):
        psi = -np.log(np.maximum(Qp / np.maximum(k, epsilon), epsilon)) / np.maximum(m, epsilon)

    return psi


def L24R10_lithology(cpt, polygons):
    """
    Classify a CPT point-by-point according to custom polygons (Lengkeek 2024).

    Args:
        cpt (GefCpt): CPT object.
        polygons (dict): Dictionary with shapely polygons.

    Returns:
        List: List of classified zone names (e.g., '2a', '5', etc.)
    """
    classifications = []
    for i in range(len(cpt.depth)):
        # Get coordinates
        x_coord = cpt.friction_nbr[i]
        y_coord = cpt.qt[i] / 100

        # Apply limits
        x_coord = np.clip(x_coord, 0.1001, 20)
        y_coord = np.clip(y_coord, 1.01, 999.99)

        point = Point(x_coord, y_coord)

        zone = None
        for zone_name, polygon in polygons.items():
            if polygon.contains(point):
                # Remove 'zone ' if it exists
                cleaned_zone = zone_name.replace('zone ', '').strip()
                zone = cleaned_zone
                break

        classifications.append(zone if zone else None)

    return classifications


def calc_IB(cpt):
    """
    Calculate the IB index based on the CPT data.

    Params:
        cpt (GefCpt): The CPT object with the interpreted data.

    Returns:
        IB (float): The calculated IB index.
    """
    # Calculate IB
    IB = (100 * (cpt.Qtn + 10)) / (70 + (cpt.Qtn + cpt.Fr))

    return IB


def filter_by_IB(Su_Fr, Su_Bq, St_Fr, St_Bq, psi_manual, IB):
    """
    Apply filtering rules based on IB index.

    Args:
        Su_Fr, Su_Bq, St_Fr, St_Bq (np.ndarray): Su and St values.
        psi_manual, psi_dgeolib (np.ndarray): Psi values.
        IB (np.ndarray): IB index.

    Returns:
        tuple: Filtered (Su_Fr, Su_Bq, St_Fr, St_Bq, psi_manual, psi_dgeolib)
    """
    # Initialize filtered copies
    Su_Fr_filtered = Su_Fr.copy()
    Su_Bq_filtered = Su_Bq.copy()
    St_Fr_filtered = St_Fr.copy()
    St_Bq_filtered = St_Bq.copy()
    psi_manual_filtered = psi_manual.copy()

    # Create masks
    mask_high = IB > 32  # IB > 32 → blank Su, St
    mask_low = IB < 22  # IB < 22 → blank Psi
    mask_transition = (IB >= 22) & (IB <= 32)  # IB in 22–32 → keep all

    # Apply the rules
    Su_Fr_filtered[mask_high] = np.nan
    Su_Bq_filtered[mask_high] = np.nan
    St_Fr_filtered[mask_high] = np.nan
    St_Bq_filtered[mask_high] = np.nan

    psi_manual_filtered[mask_low] = np.nan

    return Su_Fr_filtered, Su_Bq_filtered, St_Fr_filtered, St_Bq_filtered, psi_manual_filtered
