"""
Teleoperación del ARCTOS (hombro IZQUIERDO del soporte) con la MISMA cadena de la del FR5 (29-oct).

El FR5 (hombro derecho) se teleopera con tu brazo DERECHO; el ARCTOS, con tu brazo IZQUIERDO y
exactamente el mismo proceso: fusión de las dos OAK-D (fusion_muneca_codo), límites humanos
reales (retargeting), IK diferencial de una sola rama con postura humana, colisiones con
cápsulas (ik_diferencial) y orquestación por fotograma (teleop_brazo). Lo único propio del
ARCTOS es su cadena (cinematica.CadenaArctos: cinemática, IK analítica, codo recto, muñeca
recta, cápsulas) y los ajustes de abajo, elegidos con el diagnóstico de IK:

    python -m skill_shield.teleop_fr5.diagnostico ik --brazo izquierdo
    python -m skill_shield.teleop_fr5.diagnostico ik --brazo ambos          # FR5 y ARCTOS a la vez

Diagnóstico de IK (29-oct, misma trayectoria que el FR5, media de 4 semillas):

                         limpia            ruido 1 cm + saltos 3 cm     ruido 2 cm + saltos 3 cm
    FR5 (derecho)     80 % · p95 33 cm      77 % · med 4,8 · p95 37 cm   62 % · med 7,9 · p95 38 cm
    ARCTOS (izq.)     91 % · p95  2 cm      86 % · med 2,8 · p95  6 cm   66 % · med 4,5 · p95 10 cm
    (% de fotogramas "exacta"; error de la muñeca de la OrcaHand frente a la verdad)

El p95 del FR5 sale del "hueco" de su hombro con el brazo colgando; el ARCTOS no lo tiene. El
ARCTOS es más pequeño (escala ≈ 1,14 frente a 1,65), así que el mismo ruido le mueve menos la
mano. La orientación de la mano queda igual (p95 10-15° limpia, 12-17° con ruido).
"""
import numpy as np

from skill_shield.teleop_fr5.cinematica import CadenaArctos
from skill_shield.teleop_fr5.ik_diferencial import ParametrosIKDif
from skill_shield.teleop_fr5.ik_teleop import Capsula, ParametrosIK, entorno_soporte_hombros
from skill_shield.teleop_fr5.retargeting import ConfigRetargeting, LimitesAnatomicos
from skill_shield.teleop_fr5.teleop_brazo import ConfigTeleop, TeleopBrazoFR5

# Ajustes por defecto del ARCTOS (config.py los puede cambiar con TELEOP_IZQ_*)
DEFECTO = dict(
    TELEOP_IZQ_ESCALA=None,              # None = automática (TELEOP_IZQ_FRACCION del alcance del ARCTOS)
    TELEOP_IZQ_FRACCION=0.85,
    TELEOP_IZQ_ALCANCE_MAX=0.88,
    TELEOP_IZQ_Z_MIN=None,               # None = suelo + 0.25 (como el FR5)
    TELEOP_IZQ_LIMITES={},               # límites extra {joint 0..5: (lo, hi)} rad
    TELEOP_IZQ_V_MAX=3.0,
    TELEOP_IZQ_POSTURA_HUMANA=True,
    TELEOP_IZQ_CODO_SIGNO=None,
    # muñeca "humana" del ARCTOS: |joint4| (giro del antebrazo) y |joint5| (flexión) respecto a recta
    TELEOP_IZQ_MUNECA_DEG=(110.0, 100.0),
    # ajustes de la IK diferencial propios del ARCTOS (el resto, los del FR5)
    TELEOP_IZQ_IK_DIF=dict(w_reg=0.05, w_dir=0.30, banda_q1_deg=(-170.0, 170.0), banda_q2_deg=(-140.0, 140.0),
                           margen_duro_joint_deg={3: 45.0}),   # giro del antebrazo: menos saltos de rama
    TELEOP_IZQ_ROLL_RANGO_DEG=120.0,
)


def _g(cfg, nombre, defecto=None):
    """Valor de config (módulo o dict) o el de DEFECTO."""
    if cfg is not None:
        if isinstance(cfg, dict):
            if nombre in cfg:
                return cfg[nombre]
        elif hasattr(cfg, nombre):
            return getattr(cfg, nombre)
    return DEFECTO.get(nombre, defecto)


def entorno_arctos(z_suelo=-0.975):
    """Obstáculos fijos del ARCTOS: los mismos del soporte que usa el FR5 (columna, cabezal,
    bastidor) + suelo. El FR5 entra como obstáculo dinámico cada fotograma
    (compartir_obstaculos, más abajo)."""
    return entorno_soporte_hombros(z_suelo)


def construir_teleop_arctos(ruta_mjcf, camaras=None, cfg=None, par_fusion=None, z_suelo=-0.975,
                            longitudes_manual=None, n_calibracion=30, verbose=True,
                            auto_extrinseca=False, vigilar_extrinseca=False, vertical_auto=False,
                            entorno=None, cadena=None):
    """TeleopBrazoFR5 del ARCTOS + mano izquierda para el brazo IZQUIERDO humano.

    camaras / par_fusion: los MISMOS objetos que usa la teleop del FR5 (se comparten la extrínseca
    A↔B, que calibra y vigila la del FR5, y la vertical del tronco que mide en la T-pose).
    """
    c = cadena if cadena is not None else CadenaArctos.desde_mjcf(ruta_mjcf)
    z_min = _g(cfg, "TELEOP_IZQ_Z_MIN")
    if z_min is None:
        z_min = z_suelo + 0.25
    muneca = _g(cfg, "TELEOP_IZQ_MUNECA_DEG")
    dif = dict(DEFECTO["TELEOP_IZQ_IK_DIF"])
    dif.update(_g(cfg, "TELEOP_IZQ_IK_DIF") or {})
    lim = LimitesAnatomicos(
        hombro_az_min=_g(cfg, "HOMBRO_AZ_MIN", -45.0), hombro_az_max=_g(cfg, "HOMBRO_AZ_MAX", 130.0),
        hombro_ext_elev=_g(cfg, "HOMBRO_EXT_ELEV", 55.0), hombro_az_libre_elev=_g(cfg, "HOMBRO_AZ_LIBRE_ELEV", 20.0),
        codo_flex_max=_g(cfg, "CODO_FLEX_MAX", 145.0), humero_rot_ext=_g(cfg, "HUMERO_ROT_EXT", 95.0),
        humero_rot_int=_g(cfg, "HUMERO_ROT_INT", 110.0), activos=_g(cfg, "LIMITES_HUMANOS", True))
    teleop = TeleopBrazoFR5(
        cadena=c, camaras=camaras or {}, verbose=verbose,
        cfg=ConfigTeleop(brazo="izquierdo", longitudes_manual=longitudes_manual, n_calibracion=n_calibracion,
                         ik="diferencial", auto_extrinseca=auto_extrinseca,
                         vigilar_extrinseca=vigilar_extrinseca, vertical_auto=vertical_auto),
        par_fusion=par_fusion,
        # ejes del robot IZQUIERDO en su mundo: T-pose del brazo izquierdo = +X, arriba = +Z, delante = −Y
        cfg_retarget=ConfigRetargeting(fuera=(1.0, 0.0, 0.0), arriba=(0.0, 0.0, 1.0), delante=(0.0, -1.0, 0.0),
                                       escala=_g(cfg, "TELEOP_IZQ_ESCALA"),
                                       fraccion_alcance=float(_g(cfg, "TELEOP_IZQ_FRACCION")),
                                       z_min=z_min, prono_escala_z=_g(cfg, "PRONO_Z_ESCALA", 1.6),
                                       roll_rango_deg=float(_g(cfg, "TELEOP_IZQ_ROLL_RANGO_DEG")),
                                       alcance_max=float(_g(cfg, "TELEOP_IZQ_ALCANCE_MAX"))),
        lim_anatomicos=lim,
        par_ik=ParametrosIK(limites_extra=dict(_g(cfg, "TELEOP_IZQ_LIMITES") or {}),
                            v_seguimiento=float(_g(cfg, "TELEOP_IZQ_V_MAX")),
                            postura_humana=bool(_g(cfg, "TELEOP_IZQ_POSTURA_HUMANA")),
                            codo_signo=_g(cfg, "TELEOP_IZQ_CODO_SIGNO"),
                            muneca_q4_deg=float(muneca[0]), muneca_q5_deg=float(muneca[1])),
        par_ik_dif=ParametrosIKDif(**dif),
        entorno=entorno if entorno is not None else entorno_arctos(z_suelo))
    return teleop


def capsulas_de(teleop, q):
    """Cápsulas (mundo) del brazo de 'teleop' en la postura q, renombradas para los HUD/avisos
    del otro brazo ("FR5:mano", "ARCTOS:antebrazo"...)."""
    if q is None:
        return []
    caps = teleop.ik.col.capsulas(np.asarray(q, dtype=float))
    pref = getattr(teleop.cadena, "robot", "otro")
    return [Capsula(f"{pref}:{c_.nombre}", c_.a, c_.b, c_.r) for c_ in caps]


def compartir_obstaculos(teleop_a, q_a, teleop_b, q_b):
    """Cada brazo ve al otro como obstáculo (su postura actual). Llamar ANTES de resolver cada
    fotograma (con TeleopEnParalelo: entre recoger() y lanzar())."""
    if teleop_a is not None:
        teleop_a.ik.col.entorno.dinamicas = capsulas_de(teleop_b, q_b) if teleop_b is not None else None
    if teleop_b is not None:
        teleop_b.ik.col.entorno.dinamicas = capsulas_de(teleop_a, q_a) if teleop_a is not None else None
