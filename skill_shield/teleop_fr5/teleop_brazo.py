"""
Orquestador por fotograma: cámaras → fusión → retargeting (límites humanos) → IK → 6 joints.

Uso mínimo (ver la integración en Skill_Shield_Nivel_6.py y el simulador):

    teleop = TeleopBrazoFR5(ruta_mjcf, camaras={"A": Camara("A"), "B": Camara("B", R_ab, t_ab)})
    teleop.iniciar_calibracion()                # con el brazo estirado en T-pose (~1 s)
    ...
    q, info = teleop.actualizar({"A": obs_a, "B": obs_b}, t, mano=(f_h, l_h), q_actual=q_robot)
    if q is not None: enviar q (6 joints) al generador suave / al robot

Estados (info["estado"]): "calibrando", "sin_medida", "hold", "exacta", "aprox",
"transicion", "bloqueado".
"""
import time
from dataclasses import dataclass

import numpy as np

from skill_shield.teleop_fr5.cinematica import CadenaFR5, cargar_cadena
from skill_shield.teleop_fr5.fusion_muneca_codo import CalibradorHuesos, Fusor2Cam, Parametros
from skill_shield.teleop_fr5.ik_teleop import ControlIKFR5, Entorno, ModeloColision, ParametrosIK
from skill_shield.teleop_fr5.retargeting import Anatomia, ConfigRetargeting, LimitesAnatomicos, Retargeting


@dataclass
class ConfigTeleop:
    brazo: str = "derecho"
    zona_muerta_m: float = 0.004          # la muñeca del robot no se mueve por temblores menores
    hold_s: float = 0.5                   # s que se mantiene el último objetivo sin medida nueva
    longitudes_manual: tuple = None       # (brazo, antebrazo) m si no se quiere calibrar
    n_calibracion: int = 30


class TeleopBrazoFR5:
    def __init__(self, ruta_mjcf=None, camaras=None, cfg: ConfigTeleop = None,
                 par_fusion: Parametros = None, cfg_retarget: ConfigRetargeting = None,
                 lim_anatomicos: LimitesAnatomicos = None, par_ik: ParametrosIK = None,
                 entorno: Entorno = None, cadena: CadenaFR5 = None, verbose=True):
        self.cfg = cfg or ConfigTeleop()
        self.cadena = cadena if cadena is not None else cargar_cadena(ruta_mjcf, verbose=verbose)
        self.fusor = Fusor2Cam(camaras or {}, self.cfg.brazo, par_fusion)
        self.calib = CalibradorHuesos(self.cfg.n_calibracion, self.cfg.longitudes_manual)
        self.retarget = Retargeting(self.cadena, cfg_retarget, Anatomia(lim_anatomicos))
        # la postura humana del robot usa los mismos ejes y el mismo cono de hombro que el retargeting
        par_ik = par_ik or ParametrosIK()
        rc, la = self.retarget.cfg, self.retarget.anatomia.lim
        par_ik.ejes = (tuple(rc.fuera), tuple(rc.arriba), tuple(rc.delante))
        par_ik.hombro_az_min, par_ik.hombro_az_max = la.hombro_az_min, la.hombro_az_max
        par_ik.hombro_ext_elev = la.hombro_ext_elev
        par_ik.codo_flex_max_deg = la.codo_flex_max
        self.ik = ControlIKFR5(self.cadena, ModeloColision(self.cadena, entorno), par_ik)
        if verbose and self.ik.postura is not None:
            lo, hi = np.degrees(self.ik.lo), np.degrees(self.ik.hi)
            print("🧍 Postura humana del FR5: codo joint3 "
                  f"{'>' if self.ik.postura.codo_signo > 0 else '<'} 0 (0…{par_ik.codo_flex_max_deg:.0f}°), "
                  f"muñeca |q4|≤{par_ik.muneca_q4_deg:.0f}° |q5−90°|≤{par_ik.muneca_q5_deg:.0f}°, "
                  f"hombro q1 [{lo[0]:.0f}, {hi[0]:.0f}]° q2 [{lo[1]:.0f}, {hi[1]:.0f}]° + cono")
        self.q = None
        self.obj_prev = None
        self.t_obj = None
        self.p_w_filtrada = None
        self.info = {"estado": "sin_medida"}
        if self.calib.longitudes is not None:
            self.fusor.longitudes = self.calib.longitudes

    # ── calibración de huesos (T-pose) ───────────────────────────────────────
    def iniciar_calibracion(self):
        self.calib.iniciar()
        self.fusor.reiniciar()
        self.retarget.reiniciar_roll()
        self.ik.q = None                      # la IK arranca de la postura real del robot (q_actual)

    @property
    def calibrado(self):
        return self.calib.longitudes is not None

    # ── fotograma ────────────────────────────────────────────────────────────
    def actualizar(self, observaciones: dict, t=None, mano=None, roll=None, q_actual=None, dt=None):
        """
        observaciones: {nombre_cam: ObservacionCamara | None}
        mano: (f_h, l_h) vectores de la mano en ejes de la cámara A (MediaPipe Hands) o None
        roll: pronosupinación (rad) de otra fuente (IMU del guante) o None
        q_actual: joints reales del robot (6) para arrancar la IK sin saltos
        Devuelve (q o None, info).
        """
        t = time.perf_counter() if t is None else float(t)
        if self.ik.q is None and q_actual is not None:
            self.ik.q = np.asarray(q_actual, dtype=float).copy()
            self.ik.rama = None
            self.retarget.reiniciar_roll(self.cadena.fk(self.ik.q)["T_tool"][:3, :3])
        res = self.fusor.actualizar(observaciones, t)
        if res is not None and self.calib.calibrando:
            self.calib.alimentar(res["hombro"], res["codo"], res["muneca"])
            if self.calib.longitudes is not None:
                self.fusor.longitudes = self.calib.longitudes
        if not self.calibrado:
            self.info = dict(estado="calibrando", progreso=self.calib.progreso, fuentes=self.fusor.resumen)
            return None, self.info

        obj = None
        if res is not None:
            obj = self.retarget.objetivo(res, self.calib.longitudes, mano=mano, roll=roll, t=t)
        if obj is not None:
            # zona muerta sobre la muñeca
            p = obj["p_muneca"]
            if self.p_w_filtrada is not None and np.linalg.norm(p - self.p_w_filtrada) < self.cfg.zona_muerta_m:
                obj["p_muneca"] = self.p_w_filtrada.copy()
            else:
                self.p_w_filtrada = p.copy()
            self.obj_prev, self.t_obj = obj, t
        elif self.obj_prev is not None and (t - self.t_obj) <= self.cfg.hold_s:
            obj = self.obj_prev                     # sin medida nueva: se mantiene el objetivo
        if obj is None:
            self.info = dict(estado="sin_medida", fuentes=self.fusor.resumen)
            return self.q, self.info

        q, info_ik = self.ik.resolver(obj, t, dt)
        self.q = q
        self.info = dict(info_ik)
        self.info.update(fuentes=self.fusor.resumen, codo_reconstruido=res["codo_reconstruido"] if res else None,
                         hold=res is None, escala=obj["escala"], roll=obj["roll"],
                         anatomia=(self.retarget.anatomia.n_hombro, self.retarget.anatomia.n_codo))
        return q, self.info
