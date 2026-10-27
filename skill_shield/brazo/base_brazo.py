"""
Orientación de la BASE del brazo y mapeo ángulos humanos → joint1..3 (27-oct).

Con la peana el FR5 estaba VERTICAL y el mapeo era lineal (tabla ARM_IMITACION de
config.py): joint1 = π/2 − azimut, joint2 = π/2 − elevación, joint3 = −codo.

En el soporte nuevo el FR5 va en la placa del HOMBRO DERECHO, inclinado 45° hacia
fuera y hacia abajo: el eje de joint1 ya no es vertical y la tabla lineal deja de
valer. Aquí se calcula el mapeo EXACTO con la cinemática del FR5:

    dirección del brazo en ejes de la base  d_b = R_baseᵀ · d_mundo
    joint2 = −asin(d_b·z)          (0 = perpendicular al eje de joint1)
    joint1 = atan2(−d_b·y, −d_b·x) (rama "normal": joint2 entre −90° y +90°)

y el codo se dobla hacia el lado de su plano que queda más "arriba y hacia fuera"
(el del humano en la T-pose; con el brazo colgando, hacia fuera, lejos de la columna).

La orientación de la base se LEE de fairino_mjcf.xml (atributo quat de base_link),
así que si se vuelve a la peana (base sin girar) todo funciona exactamente como antes:
BASE_INCLINADA = False y angulos_humano.py usa la tabla ARM_IMITACION de siempre.

Direcciones del humano en el mundo (las mismas de siempre):
    fuera = −X (T-pose del brazo derecho) · arriba = +Z · delante = −Y
"""
import math
import xml.etree.ElementTree as ET

import numpy as np

from config import *

FUERA = np.array([-1.0, 0.0, 0.0])
ARRIBA = np.array([0.0, 0.0, 1.0])
DELANTE = np.array([0.0, -1.0, 0.0])

# Hacia dónde se dobla el codo dentro de su plano (ver cabecera): arriba y algo hacia fuera
_P_CODO = np.array([-0.5, 0.0, 1.0]) / math.sqrt(1.25)

# joint2 con el brazo ALINEADO con el eje de joint1 (cinemática del FR5: d_b·z = −sin q2)
Q2_EJE_BASE = -math.pi / 2


def _quat_a_mat(q):
    w, x, y, z = np.asarray(q, dtype=float) / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def _leer_base(ruta):
    """Matriz de giro de base_link según el MJCF (identidad si no tiene quat)."""
    try:
        for b in ET.parse(ruta).getroot().iter("body"):
            if b.get("name") == "base_link":
                if b.get("euler") or b.get("axisangle") or b.get("xyaxes") or b.get("zaxis"):
                    print("⚠ base_brazo: base_link usa euler/axisangle/xyaxes; solo se entiende "
                          "quat → se supone la base SIN girar.")
                    return np.eye(3)
                q = b.get("quat")
                return _quat_a_mat([float(v) for v in q.split()]) if q else np.eye(3)
    except Exception as e:                       # sin XML legible: comportamiento de siempre
        print(f"⚠ base_brazo: no puedo leer {ruta} ({e}); se supone la base SIN girar.")
    return np.eye(3)


R_BASE = _leer_base(RUTA_XML)
BASE_INCLINADA = not np.allclose(R_BASE, np.eye(3), atol=1e-6)
EJE_BASE = R_BASE[:, 2].copy()                   # eje de joint1 en el mundo


def _wrap(a):
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def dir_mundo(azimut_rad, elevacion_rad):
    """Dirección unitaria del brazo humano en el MUNDO.
    elevación: 0 = colgando · π/2 = horizontal · π = arriba. azimut: 0 = delante · π/2 = fuera."""
    s = math.sin(elevacion_rad)
    return (s * math.sin(azimut_rad) * FUERA - math.cos(elevacion_rad) * ARRIBA
            + s * math.cos(azimut_rad) * DELANTE)


def q12_desde_dir(d_mundo, q1_ref=None):
    """joint1, joint2 (rama normal) para que el brazo del FR5 apunte a d_mundo."""
    d_b = R_BASE.T @ np.asarray(d_mundo, dtype=float)
    d_b = d_b / max(float(np.linalg.norm(d_b)), 1e-12)
    q2 = -math.asin(float(np.clip(d_b[2], -1.0, 1.0)))
    if math.hypot(d_b[0], d_b[1]) < 1e-6:          # alineado con el eje de joint1
        q1 = 0.0 if q1_ref is None else float(q1_ref)
    else:
        q1 = math.atan2(-d_b[1], -d_b[0])
    return q1, q2


def signo_codo(d_mundo):
    """−1 → joint3 = −codo · +1 → joint3 = +codo (ver cabecera)."""
    d = np.asarray(d_mundo, dtype=float)
    e = EJE_BASE - float(EJE_BASE @ d) * d          # hacia dónde dobla q3 < 0 (lado del eje)
    return -1.0 if float(e @ _P_CODO) >= 0.0 else 1.0


def q_desde_angulos(azimut_rad, elevacion_rad, codo_rad, q1_ref=None):
    """(joint1, joint2, joint3) SIN recortar, con la base que tenga el modelo."""
    d = dir_mundo(azimut_rad, elevacion_rad)
    q1, q2 = q12_desde_dir(d, q1_ref)
    return np.array([q1, q2, signo_codo(d) * codo_rad])


def rangos_q12(rango_az, e_min_rad, e_max_rad, n=41):
    """[min, max] de joint1 y joint2 cuando el brazo humano recorre su rango: elevación
    de e_min a e_max y, para cada elevación, el azimut que da rango_az(e) → (min, max)
    en rad (para los límites de limites_humanos.completar_hib)."""
    q1s, q2s = [], []
    for e in np.linspace(e_min_rad, e_max_rad, n):
        az_min_rad, az_max_rad = rango_az(e)
        for a in np.linspace(az_min_rad, az_max_rad, n):
            d = dir_mundo(a, e)
            if abs(float(R_BASE[:, 2] @ d)) > 0.995:      # sobre el eje: joint1 indefinido
                continue
            q1, q2 = q12_desde_dir(d)
            q1s.append(q1)
            q2s.append(q2)
    return (min(q1s), max(q1s)), (min(q2s), max(q2s))
