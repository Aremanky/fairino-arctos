"""
Medida humana (hombro, codo, muñeca, mano) → objetivo del FR5, con LÍMITES HUMANOS reales.

1) Direcciones del brazo (hombro→codo, codo→muñeca) en el marco del TORSO (derecha, arriba,
   delante), de modo que no cambian si el operador se gira o se mueve delante de las cámaras.
2) Límites anatómicos sobre la medida (lo que un brazo no puede hacer es ruido de las cámaras):
     · hombro: azimut permitido según la elevación (cruzar por delante del pecho, abrir hacia
       atrás, extensión con el brazo bajo)
     · codo: flexión 0..145° y rotación del húmero acotada
   Son los mismos criterios de skill_shield/brazo/limites_humanos.py, reescritos sin
   dependencias de config/MuJoCo.
3) Paso al mundo del robot (fuera, arriba, delante del FR5) y ESCALADO del alcance: el robot
   con la OrcaHand llega mucho más lejos que un brazo humano, así que la muñeca del robot se
   coloca en  hombro_robot + k · (muñeca − hombro)_humano, con k = ESCALA (por defecto, el 85 %
   del alcance del robot sobre el alcance humano calibrado). El codo humano se usa como
   SUGERENCIA (de qué lado doblar el codo del robot), no como objetivo exacto.
4) Orientación de la mano: el eje "adelante" de la OrcaHand sigue la dirección de tu ANTEBRAZO
   (la flexión de la muñeca la hace la articulación de la propia OrcaHand) y el giro alrededor
   de ese eje (pronosupinación) sale de la dirección índice→meñique de la mano si se mide
   (MediaPipe Hands o el IMU del guante); si no, se mantiene el último.

Convención de la OrcaHand derecha (medida en orcahand_right_body.xml, marco mano_mount):
   +Y = adelante (dedos) · −Z = lateral índice→meñique · −X = normal de la palma (hacia donde
   se cierran los dedos). R_mano_mount = [ −n | f | −l ] con n = f × l.
OrcaHand IZQUIERDA (29-oct, ARCTOS, marco mano_mount_izq): espejo de la derecha, así que
   +Y = dedos · +Z = índice→meñique · −X = palma. R = [ −n | f | l ] con n = l × f.
   El lado lo dice la cadena (cadena.mano); para el brazo izquierdo, además, los ejes "fuera"
   del robot apuntan a +X (ConfigRetargeting.fuera) y la fusión mide tu brazo IZQUIERDO.
"""
import math
from dataclasses import dataclass

import numpy as np

from skill_shield.teleop_fr5.cinematica import rot_eje


def _u(v):
    v = np.asarray(v, dtype=float)
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-9 else None


# ══════════════════════════════════════════════════════════════════════════════
# límites anatómicos (direcciones en ejes del torso: fuera, arriba, delante)
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class LimitesAnatomicos:
    hombro_az_min: float = -45.0     # ° cruzando el brazo por delante del pecho
    hombro_az_max: float = 130.0     # ° abriendo hacia atrás con el brazo alto
    hombro_ext_elev: float = 55.0    # ° por debajo de esta elevación se puede llevar el brazo atrás
    hombro_az_libre_elev: float = 20.0   # ° con el brazo casi vertical el azimut es ruido: no se limita
    codo_flex_max: float = 145.0     # ° flexión máxima del codo
    humero_rot_ext: float = 95.0     # ° rotación externa máx. del húmero
    humero_rot_int: float = 110.0    # ° rotación interna máx. del húmero
    activos: bool = True


def az_elev(d):
    """(azimut, elevación) en grados de una dirección (fuera, arriba, delante).
    elevación: 0 = colgando, 90 = horizontal, 180 = arriba · azimut: 0 = delante, 90 = fuera."""
    elev = math.degrees(math.acos(float(np.clip(-d[1], -1.0, 1.0))))
    az = math.degrees(math.atan2(float(d[0]), float(d[2])))
    return az, elev


def dir_desde_az_elev(az, elev):
    a, e = math.radians(az), math.radians(elev)
    s = math.sin(e)
    return np.array([s * math.sin(a), -math.cos(e), s * math.cos(a)])


class Anatomia:
    """Proyecta (d_brazo, d_antebrazo) al rango anatómico. Cuenta cuántas veces actúa."""

    def __init__(self, lim: LimitesAnatomicos = None):
        self.lim = lim or LimitesAnatomicos()
        self.n_hombro = 0
        self.n_codo = 0

    def rango_azimut(self, elev):
        L = self.lim
        if elev <= L.hombro_ext_elev:
            az_max = 180.0
        elif elev >= 90.0:
            az_max = L.hombro_az_max
        else:
            t = (elev - L.hombro_ext_elev) / (90.0 - L.hombro_ext_elev)
            az_max = 180.0 + t * (L.hombro_az_max - 180.0)
        return L.hombro_az_min, az_max

    def limitar_hombro(self, d_b):
        L = self.lim
        az, el = az_elev(d_b)
        if el < L.hombro_az_libre_elev or el > 180.0 - L.hombro_az_libre_elev:
            return np.asarray(d_b, dtype=float), False
        lo, hi = self.rango_azimut(el)
        if lo <= az <= hi:
            return np.asarray(d_b, dtype=float), False
        d_lo = (lo - az) % 360.0
        d_hi = (az - hi) % 360.0
        return dir_desde_az_elev(lo if d_lo <= d_hi else hi, el), True

    @staticmethod
    def _referencia_neutra(d_b):
        """Hacia dónde iría el antebrazo al flexionar el codo SIN rotar el húmero."""
        abajo = np.array([0.0, -1.0, 0.0])
        c = float(np.clip(abajo @ d_b, -1.0, 1.0))
        if c < -0.85:
            return None
        eje = np.cross(abajo, d_b)
        n = float(np.linalg.norm(eje))
        if n < 1e-6:
            return np.array([0.0, 0.0, 1.0])
        return rot_eje(eje / n, math.acos(c)) @ np.array([0.0, 0.0, 1.0])

    def limitar_codo(self, d_b, d_a):
        L = self.lim
        d_b = np.asarray(d_b, dtype=float)
        d_a = np.asarray(d_a, dtype=float)
        flex = math.degrees(math.acos(float(np.clip(d_b @ d_a, -1.0, 1.0))))
        perp = _u(d_a - (d_a @ d_b) * d_b)
        if perp is None or flex < 15.0:
            return d_a, False
        tocado = False
        if flex > L.codo_flex_max:
            flex, tocado = L.codo_flex_max, True
        ref = self._referencia_neutra(d_b)
        if ref is not None:
            ref = _u(ref - (ref @ d_b) * d_b)
        if ref is not None:
            rot = -math.degrees(math.atan2(float(np.cross(ref, perp) @ d_b), float(ref @ perp)))
            rot_n = float(np.clip(rot, -L.humero_rot_int, L.humero_rot_ext))
            if rot_n != rot:
                perp = rot_eje(d_b, -math.radians(rot_n)) @ ref
                tocado = True
        if not tocado:
            return d_a, False
        f = math.radians(flex)
        return math.cos(f) * d_b + math.sin(f) * perp, True

    def __call__(self, d_b, d_a):
        if not self.lim.activos:
            return np.asarray(d_b, dtype=float), np.asarray(d_a, dtype=float)
        d_b0 = np.asarray(d_b, dtype=float)
        d_b, th = self.limitar_hombro(d_b0)
        if th:                                   # el antebrazo gira CON el brazo
            eje = np.cross(d_b0, d_b)
            n = float(np.linalg.norm(eje))
            if n > 1e-9:
                d_a = rot_eje(eje / n, math.acos(float(np.clip(d_b0 @ d_b, -1.0, 1.0)))) @ np.asarray(d_a, dtype=float)
        d_a, tc = self.limitar_codo(d_b, d_a)
        self.n_hombro += th
        self.n_codo += tc
        return d_b, d_a


# ══════════════════════════════════════════════════════════════════════════════
# retargeting
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class ConfigRetargeting:
    fuera: tuple = (-1.0, 0.0, 0.0)      # ejes del ROBOT en su mundo (MJCF): T-pose del brazo derecho
    arriba: tuple = (0.0, 0.0, 1.0)
    delante: tuple = (0.0, -1.0, 0.0)
    escala: float = None                 # None = 0.85·alcance_robot / alcance_humano
    fraccion_alcance: float = 0.85       # parte del alcance del robot que equivale a tu brazo estirado
    alcance_max: float = 0.92            # nunca pedir más de esta fracción del alcance (singularidad)
    z_min: float = None                  # m: altura mínima de la muñeca del robot (None = sin límite)
    z_min_suave: float = 0.15            # m: tramo de compresión suave por encima de z_min
    z_mano_scale: float = 1.0            # escala extra de la altura (1 = isótropo)
    largo_mano_m: float = 0.24           # muñeca de la OrcaHand → punta de los dedos (+ radio): con la
                                         # mano hacia abajo, z_min se aplica a la PUNTA, no a la muñeca
    comprimir_abajo: bool = True         # media inferior comprimida (el hombro del robot está bajo)
    prono_escala_z: float = 1.6          # MediaPipe Hands aplasta la profundidad de la mano
    roll_tau_s: float = 0.12             # s: suavizado exponencial de la pronosupinación medida
    roll_max_vel: float = 4.0            # rad/s máx. de cambio del roll (filtra saltos de MediaPipe)
    roll_rango_deg: float = 120.0        # pronosupinación máxima a cada lado del neutro (humano ≈ ±90°)
    offset_muneca_robot: tuple = None    # OrcaHand: articulación de la muñeca (mano_mount); None = la de
                                         # la cadena (FR5 + derecha: (0, 0.0923, 0.010))


class Retargeting:
    def __init__(self, cadena, cfg: ConfigRetargeting = None, anatomia: Anatomia = None):
        self.cadena = cadena
        self.cfg = cfg or ConfigRetargeting()
        self.anatomia = anatomia or Anatomia()
        self.ejes = np.column_stack([np.asarray(self.cfg.fuera, float), np.asarray(self.cfg.arriba, float),
                                     np.asarray(self.cfg.delante, float)])     # (fuera, arriba, delante) → mundo
        self.hombro_robot = cadena.fk(np.zeros(6))["hombro"]
        self.izquierda = getattr(cadena, "mano", "derecha") == "izquierda"
        self.alcance_robot = self._alcance_robot()
        self.L1_robot = float(np.linalg.norm(cadena.fk(np.zeros(6))["codo"] - self.hombro_robot))
        self.roll = 0.0                        # último giro de pronosupinación (rad)
        self.roll_valido = False
        self._t_roll = None
        self.roll_neutro = None                # roll humano de referencia (el de la calibración)
        self.roll_robot0 = None                # roll del robot al activar (teleop_brazo lo fija)
        self._rolls_calib = []

    def _alcance_robot(self):
        """Máxima distancia hombro → articulación de la muñeca de la OrcaHand (brazo recto, muñeca
        recta: joint4 = 0, joint5 = +90°)."""
        c = self.cadena
        q = np.array(getattr(c, "q_recto", [0.0, 0.0, 0.0, 0.0, math.pi / 2, 0.0]), dtype=float)
        p = c.fk(q)["T_tool"]
        off = self.cfg.offset_muneca_robot
        off = getattr(c, "off_muneca", (0.0, 0.0923, 0.010)) if off is None else off
        pw = p[:3, :3] @ np.asarray(off, dtype=float) + p[:3, 3]
        return float(np.linalg.norm(pw - self.hombro_robot))

    # ── de la cámara A al torso y al robot ────────────────────────────────────
    @staticmethod
    def a_torso(v, torso):
        derecha, arriba, delante, lado = torso
        v = np.asarray(v, dtype=float)
        return np.array([lado * float(v @ derecha), float(v @ arriba), float(v @ delante)])

    def a_robot(self, v_torso):
        return self.ejes @ np.asarray(v_torso, dtype=float)

    # ── objetivo ──────────────────────────────────────────────────────────────
    def _filtrar_roll(self, roll_med, t):
        """Pronosupinación RELATIVA: tu mano en la T-pose de calibración (neutro = media circular
        de las primeras medidas) equivale a la mano del robot tal como estaba al activar
        (roll_robot0, p. ej. joint6 = 0 en reposo); después, cada grado tuyo es un grado del
        robot, acotado a ±roll_rango_deg. Suavizado exponencial + límite de velocidad."""
        cfg = self.cfg
        if t is None or self._t_roll is None:
            self._t_roll = t
            if not self.roll_valido:
                self.roll, self.roll_valido = (self.roll_robot0 if self.roll_robot0 is not None else roll_med), True
            return
        dt = max(1e-3, float(t - self._t_roll))
        self._t_roll = t
        if self.roll_neutro is None:           # las primeras medidas fijan el neutro (mediana circular)
            self._rolls_calib.append(roll_med)
            if len(self._rolls_calib) >= 15:
                self.roll_neutro = float(np.angle(np.mean(np.exp(1j * np.array(self._rolls_calib)))))
            else:
                return                         # hasta entonces la mano del robot no gira
        lim = math.radians(cfg.roll_rango_deg)
        rel = (roll_med - self.roll_neutro + math.pi) % (2 * math.pi) - math.pi
        base = self.roll_robot0 if self.roll_robot0 is not None else self.roll_neutro
        roll_obj = base + float(np.clip(rel, -lim, lim))
        d = (roll_obj - self.roll + math.pi) % (2 * math.pi) - math.pi
        a = 1.0 - math.exp(-dt / max(cfg.roll_tau_s, 1e-3))
        d = float(np.clip(a * d, -cfg.roll_max_vel * dt, cfg.roll_max_vel * dt))
        self.roll = self.roll + d

    def objetivo(self, res, longitudes, mano=None, roll=None, t=None):
        """
        res: salida de Fusor2Cam.actualizar (hombro/codo/muñeca en marco A + torso).
        longitudes: (brazo, antebrazo) humanos calibrados (m).
        mano: opcional, (f_h, l_h) = vectores adelante (muñeca→corazón MCP) e índice→meñique de
              la mano en ejes de la cámara A (MediaPipe Hands world landmarks); o None.
        roll: opcional, pronosupinación en rad (p. ej. del IMU del guante); manda sobre 'mano'.
        Devuelve dict: p_muneca (mundo), R_mano (3x3, marco mano_mount), p_codo_sugerido,
                       d_brazo, d_antebrazo (mundo robot), flags.
        """
        cfg = self.cfg
        S, E, W = res["hombro"], res["codo"], res["muneca"]
        torso = res["torso"]
        d_b = _u(self.a_torso(E - S, torso))
        d_a = _u(self.a_torso(W - E, torso))
        if d_b is None or d_a is None:
            return None
        d_b, d_a = self.anatomia(d_b, d_a)           # límites humanos reales
        L1, L2 = longitudes
        v_w = L1 * d_b + L2 * d_a                    # hombro→muñeca humano (torso), ya coherente
        alcance_h = L1 + L2
        k = cfg.escala if cfg.escala else cfg.fraccion_alcance * self.alcance_robot / alcance_h
        v_r = k * self.a_robot(v_w)
        # alcance máximo (lejos de la singularidad del brazo estirado)
        r = float(np.linalg.norm(v_r))
        r_max = cfg.alcance_max * self.alcance_robot
        if r > r_max:
            v_r *= r_max / r
        p_w = self.hombro_robot + v_r
        d_b_r = self.a_robot(d_b)
        d_a_r = self.a_robot(d_a)
        # altura mínima (suelo / mesa / bastidor): compresión suave. Se aplica al punto más bajo
        # de la mano: con la mano apuntando hacia abajo, los dedos quedan largo_mano_m por debajo
        # de la muñeca (con la escala ~1,65 el brazo colgando del robot no cabe hasta el suelo).
        if cfg.z_min is not None:
            f_ = _u(d_a_r)
            bajada = max(0.0, -float(f_[2])) * cfg.largo_mano_m if f_ is not None else 0.0
            z_rel = p_w[2] - (cfg.z_min + bajada)
            if z_rel < cfg.z_min_suave:
                s = cfg.z_min_suave
                # de z_rel = s hacia abajo: z' = s · exp((z_rel − s)/s) ∈ (0, s]
                p_w[2] = cfg.z_min + bajada + s * math.exp(min((z_rel - s) / s, 0.0))
        # codo sugerido: dirección del brazo humano con la longitud del brazo del robot
        p_e = self.hombro_robot + self.L1_robot * d_b_r
        # orientación: adelante = antebrazo; giro = pronosupinación
        f = _u(d_a_r)
        roll_med = None
        if roll is not None:
            roll_med = float(roll)
        elif mano is not None:
            f_h, l_h = mano
            l_t = self.a_torso(np.asarray(l_h, float) * np.array([1.0, 1.0, cfg.prono_escala_z]), torso)
            l_r = self.a_robot(l_t)
            l_p = _u(l_r - float(l_r @ f) * f)
            if l_p is not None and np.linalg.norm(l_r - float(l_r @ f) * f) > 0.3 * np.linalg.norm(l_r):
                ref = self._ref_roll(f)
                roll_med = math.atan2(float(np.cross(ref, l_p) @ f), float(ref @ l_p))
        if roll_med is not None:
            self._filtrar_roll(roll_med, t)
        R_mano = self.R_mano(f, self.roll)
        return {
            "p_muneca": p_w, "R_mano": R_mano, "p_codo_sugerido": p_e,
            "d_brazo": d_b_r, "d_antebrazo": d_a_r, "escala": k,
            "roll": self.roll, "roll_valido": self.roll_valido,
            "d_brazo_torso": d_b, "d_antebrazo_torso": d_a,
        }

    @staticmethod
    def _ref_roll(f):
        """Referencia del giro: lateral 'natural' ⟂ al antebrazo (horizontal, hacia fuera-derecha)."""
        up = np.array([0.0, 0.0, 1.0])
        ref = np.cross(f, up)            # horizontal, ⟂ al antebrazo
        if np.linalg.norm(ref) < 0.2:    # antebrazo vertical: otra referencia
            ref = np.cross(f, np.array([0.0, 1.0, 0.0]))
        return ref / np.linalg.norm(ref)

    def R_mano(self, f, roll):
        """Orientación de mano_mount: adelante = f; lateral índice→meñique = ref girada 'roll'."""
        f = _u(f)
        ref = self._ref_roll(f)
        l = rot_eje(f, roll) @ ref
        if self.izquierda:               # mano izquierda (espejo): palma n = l × f, R = [−n | f | l]
            return np.column_stack([-np.cross(l, f), f, l])
        n = np.cross(f, l)               # normal de la palma (hacia donde cierran los dedos)
        return np.column_stack([-n, f, -l])

    def reiniciar_roll(self, R_robot=None):
        """Al activar: el neutro humano se vuelve a medir y, si se da la orientación actual de la
        mano del robot, su roll pasa a ser la referencia (roll_robot0)."""
        self.roll_neutro, self._rolls_calib, self.roll_valido, self._t_roll = None, [], False, None
        if R_robot is not None:
            self.roll_robot0 = self.roll_desde_R(R_robot)
            self.roll = self.roll_robot0

    def roll_desde_R(self, R):
        """Inversa de R_mano: pronosupinación de una orientación dada (para arrancar sin saltos)."""
        f = R[:, 1]
        l = R[:, 2] if self.izquierda else -R[:, 2]
        ref = self._ref_roll(f)
        return math.atan2(float(np.cross(ref, l) @ f), float(ref @ l))
