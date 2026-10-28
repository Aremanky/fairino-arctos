"""
Hombro, codo y muñeca en 3D con DOS cámaras OAK-D, robusto a la oclusión de la palma.

Por qué hacía falta otra fusión
───────────────────────────────
Con el brazo estirado HACIA una de las cámaras pasan tres cosas a la vez en esa cámara:
  1. la palma y los dedos tapan la muñeca y el codo → MediaPipe los "inventa" (y con
     visibilidad alta): el píxel del codo cae dentro de la mano;
  2. la profundidad estéreo en el píxel de la muñeca es la de la PALMA (10-20 cm más cerca);
  3. muñeca y codo quedan casi en la misma línea de vista (escorzo): un error de 2 px en la
     imagen es un error enorme en la dirección del antebrazo.
La fusión antigua promediaba "si se parecen" y elegía una cámara "si no", pero no sabía
POR QUÉ discrepaban. Aquí cada cámara mide cada articulación con una incertidumbre que
depende explícitamente de esas tres causas, y lo que no es fiable se marca como SOSPECHOSO:
si la otra cámara mide bien esa articulación, lo sospechoso se degrada a "solo rayo"
(posición en la imagen sin profundidad) o se descarta; si no hay nada mejor, se usa con la
profundidad muy "ensanchada" (para que el error quede acotado y no derive).

Cómo funciona (por fotograma)
─────────────────────────────
Por cámara y articulación (observar_camara):
  · píxel + visibilidad de MediaPipe Pose, profundidad estéreo en un parche (percentil bajo)
  · ESCORZO: ángulo entre el antebrazo previsto (filtro) y el rayo de vista de la muñeca.
      < 20°  → el antebrazo apunta a la cámara: codo descartado, muñeca con z sospechosa
      20-40° → transición suave
  · PALMA: con los 21 puntos de MediaPipe Hands (si los hay) se construye la silueta 2D de la
    mano; si no, una elipse a partir de la muñeca y la dirección codo→muñeca. Codo dentro de
    la silueta → descartado. Profundidad de la muñeca más cerca de lo previsto → sospechosa.
  · COHERENCIA: profundidad muy distinta a la prevista (fondo) o huesos de longitud
    imposible → sospechosa.
  · Cada medida es un punto 3D con covarianza ANISÓTROPA (lateral = σ_pix·z/f; profundidad =
    k·z² si es válida, ensanchada si es sospechosa, "infinita" si es solo rayo). Dos rayos de
    dos cámaras = triangulación.
Fusión (Fusor2Cam):
  · filtro de Kalman por articulación (posición + velocidad, marco de la cámara A), con las
    medidas de las dos cámaras combinadas en forma de información y puerta χ² (Mahalanobis)
  · sin medidas válidas, predice (coasting) hasta hold_s y luego marca "perdido"
  · tras el filtro, coherencia con las longitudes de los huesos (calibradas en T-pose):
    muñeca no más lejos del hombro que brazo+antebrazo; codo poco fiable → reconstruido en
    la circunferencia compatible, en el lado del último codo bueno.
Salida: hombro, codo, muñeca (marco A, metros), covarianzas, marco del torso, diagnóstico.

Este módulo no importa config.py ni MuJoCo: se le pasan los parámetros (clase Parametros).
"""
import math
import time
from dataclasses import dataclass, field

import numpy as np

_I3 = np.eye(3)

# Índices de MediaPipe Pose
PS_HOMBRO_IZQ, PS_HOMBRO_DER = 11, 12
PS_CODO_IZQ, PS_CODO_DER = 13, 14
PS_MUNECA_IZQ, PS_MUNECA_DER = 15, 16
PS_CADERA_IZQ, PS_CADERA_DER = 23, 24


@dataclass
class Parametros:
    vis_min: float = 0.35              # visibilidad mínima de MediaPipe para usar un punto
    sigma_pix: float = 4.0             # error típico de MediaPipe en la imagen (px)
    sigma_z_k: float = 0.006           # σ_z = k·z²  (m)
    sigma_z_sospecha: float = 0.12     # m: σ_z de una profundidad sospechosa cuando no hay nada mejor
    z_min: float = 0.25                # m: por debajo el estéreo no mide
    z_max: float = 5.0
    prof_radio_px: int = 4             # parche de profundidad (radio)
    prof_percentil: float = 15.0
    prof_min_pix: int = 4
    escorzo_ang: tuple = (20.0, 40.0)  # ° (codo descartado / fiable) del antebrazo respecto al rayo
    mano_largo_m: float = 0.19         # m: muñeca → punta del corazón (silueta aproximada)
    mano_ancho_m: float = 0.10
    margen_silueta_px: float = 8.0
    prof_gate_cerca_m: float = 0.06    # z medida más cerca que la prevista por más de esto → palma
    prof_gate_lejos_m: float = 0.20    # más lejos → fondo
    chi2_gate: float = 11.3            # χ² 3 g.l. al 99 %
    acel_sigma: dict = field(default_factory=lambda: {"hombro": 1.0, "codo": 4.0, "muneca": 8.0})  # m/s²
    hold_s: float = 0.8                # s sin medidas antes de dar la articulación por perdida
    tol_huesos: float = 0.25           # ±25 % de la longitud calibrada
    hombro_hacia_torso: float = 0.6    # el hombro se mide en el pecho (la mano tapa el hombro)
    arriba_ref: str = "camara"         # "camara" | "caderas"
    codo_sigma_fiable: float = 0.06    # m: σ del codo por debajo de la cual no se reconstruye
    # ── triangulación A+B (28-oct): con las dos cámaras viendo la articulación, su posición sale
    #    de los RAYOS de los dos píxeles (cámaras separadas ~56° y 2 m: mucho más precisa que el
    #    estéreo de cada OAK a 2 m, y no depende de la ropa); el estéreo queda como apoyo débil ──
    triangular: bool = True
    tri_ang_min_deg: float = 8.0       # ángulo mínimo entre los dos rayos (si no, mal condicionado)
    tri_sigma_z_min: float = 0.08      # m: el estéreo como apoyo a lo largo del rayo, nunca más fino que esto
    tri_res_max: float = 3.5           # residuo máx. de cada rayo (en σ) para aceptar la triangulación
    z_nominal: float = 2.0             # m: profundidad provisional de un rayo sin estéreo ni predicción
    arriba_fija: tuple = None          # vertical calibrada (marco A); la usa arriba_ref = "calibrada"


# ══════════════════════════════════════════════════════════════════════════════
# geometría de cámara
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class Intrinsecos:
    fx: float
    fy: float
    cx: float
    cy: float
    ancho: int
    alto: int

    @classmethod
    def desde_K(cls, K, ancho, alto):
        K = np.asarray(K, dtype=float)
        return cls(float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2]), int(ancho), int(alto))

    @classmethod
    def desde_focal(cls, f_px, ancho, alto):
        return cls(float(f_px), float(f_px), ancho / 2.0, alto / 2.0, int(ancho), int(alto))

    @property
    def f(self):
        return 0.5 * (self.fx + self.fy)

    def rayo(self, u, v):
        """Dirección (no unitaria, z=1) del rayo que pasa por el píxel (u, v)."""
        return np.array([(u - self.cx) / self.fx, (v - self.cy) / self.fy, 1.0])

    def proyectar(self, p):
        """Punto 3D (marco cámara) → píxel. None si está detrás."""
        if p[2] <= 1e-6:
            return None
        return np.array([self.fx * p[0] / p[2] + self.cx, self.fy * p[1] / p[2] + self.cy])


def profundidad_parche(depth_mm, u, v, radio, percentil, min_pix):
    """Profundidad (m) con percentil bajo en un parche; None si no hay datos válidos."""
    h, w = depth_mm.shape
    u, v = int(round(u)), int(round(v))
    if not (0 <= u < w and 0 <= v < h):
        return None
    reg = depth_mm[max(0, v - radio):min(h, v + radio + 1), max(0, u - radio):min(w, u + radio + 1)]
    val = reg[reg > 0]
    if len(val) < min_pix:
        return None
    return float(np.percentile(val, percentil)) / 1000.0


def _suave01(x, lo, hi):
    t = float(np.clip((x - lo) / max(hi - lo, 1e-9), 0.0, 1.0))
    return t * t * (3.0 - 2.0 * t)


def _punto_en_poligono(p, poli):
    """Test par-impar de (x, y) dentro del polígono (N, 2)."""
    x, y = p
    dentro = False
    n = len(poli)
    j = n - 1
    for i in range(n):
        xi, yi = poli[i]
        xj, yj = poli[j]
        if (yi > y) != (yj > y) and x < (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi:
            dentro = not dentro
        j = i
    return dentro


def _casco_convexo(pts):
    """Casco convexo 2D (cadena monótona) de (N, 2) → (M, 2)."""
    pts = sorted({(float(x), float(y)) for x, y in pts})
    if len(pts) < 3:
        return np.array(pts)

    def cruz(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    inf, sup = [], []
    for p in pts:
        while len(inf) >= 2 and cruz(inf[-2], inf[-1], p) <= 0:
            inf.pop()
        inf.append(p)
    for p in reversed(pts):
        while len(sup) >= 2 and cruz(sup[-2], sup[-1], p) <= 0:
            sup.pop()
        sup.append(p)
    return np.array(inf[:-1] + sup[:-1])


def _expandir_poligono(poli, margen):
    c = poli.mean(axis=0)
    out = []
    for p in poli:
        d = p - c
        n = np.linalg.norm(d)
        out.append(p + (d / n * margen if n > 1e-9 else 0.0))
    return np.array(out)


# ══════════════════════════════════════════════════════════════════════════════
# medida de una cámara
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class Medida3D:
    """Una medida de una articulación por una cámara, en el marco de ESA cámara."""
    p: np.ndarray            # punto (a la profundidad medida o, sin ella, a la prevista)
    rayo: np.ndarray         # dirección unitaria del rayo de vista (marco cámara)
    s_lat: float             # σ lateral (m)
    s_z: float               # σ en profundidad si la profundidad es válida (m)
    tiene_z: bool            # True si hay profundidad medida
    sospechosa: bool         # True si la profundidad puede ser de la palma / del fondo
    peso: float              # 0..1 (confianza global; 0 = descartada)
    nota: str = ""

    def covarianza(self, modo):
        """modo: "normal" | "sospecha" (σ_z ensanchada) | "rayo" (sin profundidad)."""
        if modo == "rayo" or not self.tiene_z:
            s_z = 1e3
        elif modo == "sospecha":
            s_z = max(self.s_z, self._s_z_sospecha)
        else:
            s_z = self.s_z
        rz = self.rayo
        tmp = np.array([1.0, 0.0, 0.0]) if abs(rz[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        rx = np.cross(rz, tmp)
        rx /= np.linalg.norm(rx)
        ry = np.cross(rz, rx)
        B = np.column_stack([rx, ry, rz])
        return B @ np.diag([self.s_lat ** 2, self.s_lat ** 2, s_z ** 2]) @ B.T

    _s_z_sospecha: float = 0.12


class ObservacionCamara:
    """
    Entrada de UNA cámara en un fotograma:
      lm:        {índice Pose: (u_norm, v_norm, visibilidad)}  (o el objeto pose_landmarks de MediaPipe)
      depth_mm:  mapa de profundidad uint16 (mm) alineado con el color, o None
      intr:      Intrinsecos
      mano_lm:   (21, 2) píxeles de MediaPipe Hands de la mano teleoperada, o None
      z_func:    alternativa a depth_mm: función (u, v) → z (m) o None (para simulación)
    """

    def __init__(self, lm, depth_mm, intr, mano_lm=None, z_func=None, t=None):
        if hasattr(lm, "landmark"):
            lm = {i: (l.x, l.y, l.visibility) for i, l in enumerate(lm.landmark)}
        self.lm = lm
        self.depth = depth_mm
        self.intr = intr
        self.mano_lm = None if mano_lm is None else np.asarray(mano_lm, dtype=float).reshape(-1, 2)
        self.z_func = z_func
        self.t = time.perf_counter() if t is None else t

    def pixel(self, i):
        l = self.lm.get(i)
        if l is None:
            return None, 0.0
        u, v, vis = l
        if not (0.0 <= u < 1.0 and 0.0 <= v < 1.0):
            return None, 0.0
        return np.array([u * self.intr.ancho, v * self.intr.alto]), float(vis)

    def z_en(self, u, v, par):
        if self.z_func is not None:
            return self.z_func(u, v)
        if self.depth is None:
            return None
        for radio in (par.prof_radio_px, 2 * par.prof_radio_px):
            z = profundidad_parche(self.depth, u, v, radio, par.prof_percentil, par.prof_min_pix)
            if z is not None and par.z_min < z < par.z_max:
                return z
        return None


def _ids(brazo):
    if brazo == "derecho":
        return dict(h=PS_HOMBRO_DER, c=PS_CODO_DER, m=PS_MUNECA_DER, h2=PS_HOMBRO_IZQ,
                    cad=PS_CADERA_DER, cad2=PS_CADERA_IZQ, lado=1.0)
    return dict(h=PS_HOMBRO_IZQ, c=PS_CODO_IZQ, m=PS_MUNECA_IZQ, h2=PS_HOMBRO_DER,
                cad=PS_CADERA_IZQ, cad2=PS_CADERA_DER, lado=-1.0)


class Camara:
    """Cámara con sus extrínsecos respecto a A: p_A = R · p_cam + t."""

    def __init__(self, nombre, R=None, t=None):
        self.nombre = nombre
        self.R = np.eye(3) if R is None else np.asarray(R, dtype=float).reshape(3, 3)
        self.t = np.zeros(3) if t is None else np.asarray(t, dtype=float).reshape(3)

    def a_A(self, p):
        return self.R @ p + self.t

    def de_A(self, p):
        return self.R.T @ (p - self.t)


def observar_camara(obs: ObservacionCamara, cam: Camara, brazo: str, par: Parametros,
                    prediccion: dict, longitudes=None):
    """
    Medidas 3D (marco de la cámara) de hombro, codo, muñeca (+ hombro opuesto y caderas si se
    ven). 'prediccion' = {nombre: (punto en marco A, σ a lo largo del rayo o None)} del filtro.
    Devuelve ({nombre: Medida3D}, notas).
    """
    ids = _ids(brazo)
    notas = []
    out = {}
    pred = {}
    for k, v in prediccion.items():
        if v is None:
            continue
        p, sig = v if isinstance(v, tuple) else (v, None)
        pred[k] = (cam.de_A(p), sig)

    def medida(nombre, i, z_pix=None, vis_extra=1.0, sospechosa=False, nota=""):
        """Medida de la articulación 'i' (índice Pose). z_pix = (u, v) donde medir la profundidad
        (si no, en el propio píxel)."""
        pix, vis = obs.pixel(i)
        if pix is None or vis < par.vis_min:
            return None
        u, v = pix
        zu, zv = pix if z_pix is None else z_pix
        z = obs.z_en(zu, zv, par)
        zp = pred.get(nombre)
        z_prev, s_prev = (float(zp[0][2]), zp[1]) if zp is not None and zp[0][2] > par.z_min else (None, None)
        motivo = nota
        if z is not None and z_prev is not None:
            holg = 2.0 * float(s_prev) if s_prev is not None else 0.0
            if z < z_prev - (par.prof_gate_cerca_m + holg):
                sospechosa, motivo = True, motivo + "z<prev(palma?) "
            elif z > z_prev + (par.prof_gate_lejos_m + holg):
                sospechosa, motivo = True, motivo + "z>prev(fondo) "
        r = obs.intr.rayo(u, v)
        rz = r / np.linalg.norm(r)
        if z is not None:
            z_uso = z
        elif z_prev is not None:
            z_uso = z_prev
        elif par.triangular:
            z_uso = par.z_nominal          # solo rayo: vale para triangular con la otra cámara
        else:
            return None                    # sin profundidad ni predicción no hay punto
        p = r * z_uso
        w = max(vis, 0.2) * max(vis_extra, 1e-3)
        s_lat = par.sigma_pix * z_uso / obs.intr.f / w
        s_z = par.sigma_z_k * z_uso * z_uso / w
        m = Medida3D(p, rz, s_lat, s_z, z is not None, bool(sospechosa), float(vis_extra), motivo)
        m._s_z_sospecha = par.sigma_z_sospecha
        return m

    # ── hombros (medidos en el pecho) y caderas ───────────────────────────────
    ph, vh = obs.pixel(ids["h"])
    ph2, vh2 = obs.pixel(ids["h2"])
    if ph is not None and ph2 is not None and vh >= par.vis_min and vh2 >= par.vis_min:
        med = (ph + ph2) / 2.0
        m = medida("hombro", ids["h"], z_pix=ph + par.hombro_hacia_torso * (med - ph))
        if m is not None:
            out["hombro"] = m
        m = medida("hombro2", ids["h2"], z_pix=ph2 + par.hombro_hacia_torso * (med - ph2))
        if m is not None:
            out["hombro2"] = m
    elif ph is not None and vh >= par.vis_min:
        m = medida("hombro", ids["h"])
        if m is not None:
            out["hombro"] = m
    for nombre, i in (("cadera", ids["cad"]), ("cadera2", ids["cad2"])):
        m = medida(nombre, i)
        if m is not None:
            out[nombre] = m

    # ── escorzo del antebrazo en esta cámara ──────────────────────────────────
    pm, vm = obs.pixel(ids["m"])
    pc, vc = obs.pixel(ids["c"])
    w_escorzo = 1.0
    if "codo" in pred and "muneca" in pred and pm is not None:
        d_ante = pred["muneca"][0] - pred["codo"][0]
        n = np.linalg.norm(d_ante)
        if n > 1e-6:
            d_ante /= n
            rm = obs.intr.rayo(*pm)
            rm /= np.linalg.norm(rm)
            ang = math.degrees(math.acos(min(1.0, abs(float(d_ante @ rm)))))
            w_escorzo = _suave01(ang, *par.escorzo_ang)
            if w_escorzo < 0.99:
                notas.append(f"escorzo {ang:.0f}°")

    # ── silueta de la mano en la imagen ───────────────────────────────────────
    silueta = None
    if obs.mano_lm is not None and len(obs.mano_lm) >= 5:
        silueta = _expandir_poligono(_casco_convexo(obs.mano_lm), par.margen_silueta_px)
    elif pm is not None and vm >= par.vis_min:
        # elipse aproximada: la mano sale de la muñeca en la dirección codo→muñeca (2D)
        zm = pred["muneca"][0][2] if "muneca" in pred and pred["muneca"][0][2] > par.z_min else None
        if zm is None:
            zm = obs.z_en(*pm, par)
        if zm is not None:
            largo = par.mano_largo_m * obs.intr.f / zm
            ancho = par.mano_ancho_m * obs.intr.f / zm
            if pc is not None and vc >= par.vis_min and np.linalg.norm(pm - pc) > 4.0:
                d2 = (pm - pc) / np.linalg.norm(pm - pc)
            else:
                d2 = np.array([0.0, -1.0])
            if w_escorzo < 0.5:                      # mano hacia la cámara: la silueta es un disco
                largo = max(largo, ancho) * 0.6
                ancho = largo
            c2 = pm + 0.5 * largo * d2
            n2 = np.array([-d2[1], d2[0]])
            ang = np.linspace(0, 2 * math.pi, 12, endpoint=False)
            silueta = np.array([c2 + 0.5 * largo * math.cos(a) * d2 + 0.5 * ancho * math.sin(a) * n2
                                for a in ang])
            silueta = _expandir_poligono(silueta, par.margen_silueta_px)

    # ── codo ──────────────────────────────────────────────────────────────────
    codo_en_mano = bool(pc is not None and silueta is not None and _punto_en_poligono(pc, silueta))
    if codo_en_mano:
        notas.append("codo tapado por la mano")
    w_codo = w_escorzo * (0.0 if codo_en_mano else 1.0)
    if w_codo > 0.02:
        m = medida("codo", ids["c"], vis_extra=w_codo, sospechosa=w_escorzo < 0.7,
                   nota="escorzo " if w_escorzo < 0.7 else "")
        if m is not None and longitudes is not None and "hombro" in out and m.tiene_z and not m.sospechosa:
            L1 = float(np.linalg.norm(m.p - out["hombro"].p))
            if abs(L1 - longitudes[0]) > par.tol_huesos * longitudes[0]:
                m.sospechosa, m.nota = True, m.nota + "L1 incoherente "
        if m is not None:
            out["codo"] = m
    elif pc is not None:
        notas.append("codo descartado")

    # ── muñeca ────────────────────────────────────────────────────────────────
    if pm is not None and vm >= par.vis_min:
        m = medida("muneca", ids["m"], vis_extra=max(w_escorzo, 0.35), sospechosa=w_escorzo < 0.5,
                   nota="escorzo " if w_escorzo < 0.5 else "")
        if m is not None and m.tiene_z and not m.sospechosa and longitudes is not None and "hombro" in out:
            d = float(np.linalg.norm(m.p - out["hombro"].p))
            if d > (longitudes[0] + longitudes[1]) * (1.0 + par.tol_huesos * 0.5):
                m.sospechosa, m.nota = True, m.nota + "muñeca más lejos que el brazo "
        if m is not None:
            out["muneca"] = m
    return out, notas


# ══════════════════════════════════════════════════════════════════════════════
# filtro de Kalman por articulación
# ══════════════════════════════════════════════════════════════════════════════

class KalmanPunto:
    """Posición + velocidad (6 estados), modelo de velocidad constante, en el marco A."""

    def __init__(self, acel_sigma):
        self.sa = float(acel_sigma)
        self.x = None
        self.P = None
        self.t = None
        self.t_ult_medida = None
        self.n_medidas = 0

    @property
    def p(self):
        return None if self.x is None else self.x[:3].copy()

    @property
    def v(self):
        return None if self.x is None else self.x[3:].copy()

    @property
    def Pp(self):
        return None if self.P is None else self.P[:3, :3].copy()

    def sigma_en(self, direccion):
        """σ (m) de la posición a lo largo de una dirección unitaria."""
        if self.P is None:
            return None
        d = np.asarray(direccion, dtype=float)
        return math.sqrt(max(float(d @ self.P[:3, :3] @ d), 0.0))

    def reset(self):
        self.x = self.P = self.t = self.t_ult_medida = None
        self.n_medidas = 0

    def predecir(self, t):
        if self.x is None:
            return
        dt = max(0.0, min(float(t - self.t), 0.2))
        F = np.eye(6)
        F[:3, 3:] = dt * np.eye(3)
        G = np.vstack([0.5 * dt * dt * np.eye(3), dt * np.eye(3)])
        Q = G @ G.T * self.sa ** 2 + 1e-9 * np.eye(6)
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q
        self.t = t

    def actualizar(self, p, S, t, chi2_gate):
        """Actualiza con la medida (p, S). Devuelve (aceptada, d² de Mahalanobis)."""
        p = np.asarray(p, dtype=float)
        S = np.asarray(S, dtype=float)
        if self.x is None:
            self.x = np.concatenate([p, np.zeros(3)])
            self.P = np.zeros((6, 6))
            self.P[:3, :3] = S + 1e-4 * np.eye(3)
            self.P[3:, 3:] = 4.0 * np.eye(3)
            self.t = self.t_ult_medida = t
            self.n_medidas = 1
            return True, 0.0
        H = np.hstack([np.eye(3), np.zeros((3, 3))])
        Sk = H @ self.P @ H.T + S
        y = p - H @ self.x
        try:
            Sk_inv = np.linalg.inv(Sk)
        except np.linalg.LinAlgError:
            return False, float("inf")
        d2 = float(y @ Sk_inv @ y)
        if d2 > chi2_gate:
            return False, d2
        K = self.P @ H.T @ Sk_inv
        self.x = self.x + K @ y
        self.P = (np.eye(6) - K @ H) @ self.P
        self.P = 0.5 * (self.P + self.P.T)
        self.t_ult_medida = t
        self.n_medidas += 1
        return True, d2


# ══════════════════════════════════════════════════════════════════════════════
# fusor
# ══════════════════════════════════════════════════════════════════════════════

class Fusor2Cam:
    """
    Uso por fotograma:
        res = fusor.actualizar({"A": obs_a, "B": obs_b}, t)
    obs_* = ObservacionCamara o None. Las cámaras se declaran al construir (extrínsecos).
    res = None si no hay hombro+muñeca fiables; si no, dict con:
        "hombro", "codo", "muneca"   np.array (marco A, m)
        "P_hombro", ...              covarianzas 3x3 (confianza)
        "torso"                      (derecha, arriba, delante, lado) en marco A
        "codo_reconstruido"          True si el codo salió de los huesos, no de las cámaras
        "fuentes"                    texto de diagnóstico
    """
    ARTICULACIONES = ("hombro", "hombro2", "cadera", "cadera2", "codo", "muneca")

    def __init__(self, camaras: dict, brazo="derecho", par: Parametros = None, longitudes=None):
        self.cams = camaras                       # {nombre: Camara}
        self.brazo = brazo
        self.par = par or Parametros()
        self.longitudes = longitudes              # (brazo, antebrazo) m, o None
        self.kf = {k: KalmanPunto(self.par.acel_sigma.get(k.rstrip("2"), 2.0))
                   for k in self.ARTICULACIONES}
        self.ultimo_codo_rel = None               # último vector hombro→codo fiable
        self.resumen = "-"
        self.diag = {}                            # diagnóstico del último fotograma (ver actualizar)
        self.t_ult = None
        self.rechazos = {k: 0 for k in self.ARTICULACIONES}   # rechazos χ² consecutivos
        self.max_rechazos = 3                     # tras estos, el filtro se reinicia a la medida

    def reiniciar(self):
        for k in self.kf.values():
            k.reset()
        self.ultimo_codo_rel = None
        self.rechazos = {k: 0 for k in self.ARTICULACIONES}

    def _actualizar_kf(self, art, pA, SA, t, con_z):
        """Actualiza el filtro de 'art'; si lleva max_rechazos fotogramas rechazando medidas
        con profundidad, es que la predicción está mal (movimiento brusco): se reinicia."""
        kf = self.kf[art]
        ok, _ = kf.actualizar(pA, SA, t, self.par.chi2_gate)
        if ok:
            self.rechazos[art] = 0
            return True
        self.rechazos[art] += 1
        if con_z and self.rechazos[art] >= self.max_rechazos:
            kf.reset()
            kf.actualizar(pA, SA, t, self.par.chi2_gate)
            self.rechazos[art] = 0
            return True
        return False

    def prediccion(self, t=None):
        """{art: punto} de las articulaciones con predicción vigente (medida reciente)."""
        out = {}
        for k, f in self.kf.items():
            if f.p is None:
                continue
            if t is not None and f.t_ult_medida is not None and (t - f.t_ult_medida) > self.par.hold_s:
                continue
            out[k] = f.p
        return out

    def _prediccion_para(self, cam: Camara, t):
        """Predicción con σ a lo largo del rayo de ESA cámara (para las puertas de profundidad)."""
        out = {}
        for k, f in self.kf.items():
            if f.p is None or f.t_ult_medida is None or (t - f.t_ult_medida) > self.par.hold_s:
                continue
            pc = cam.de_A(f.p)
            n = np.linalg.norm(pc)
            if n < 1e-6:
                continue
            out[k] = (f.p, f.sigma_en(cam.R @ (pc / n)))
        return out

    # ── un fotograma ────────────────────────────────────────────────────────
    def actualizar(self, observaciones: dict, t=None):
        t = time.perf_counter() if t is None else float(t)
        par = self.par
        for f in self.kf.values():
            f.predecir(t)
        medidas = {k: [] for k in self.ARTICULACIONES}       # {art: [(nombre_cam, Medida3D)]}
        notas = []
        for nombre, obs in observaciones.items():
            if obs is None:
                continue
            cam = self.cams[nombre]
            meds, nn = observar_camara(obs, cam, self.brazo, par, self._prediccion_para(cam, t),
                                       self.longitudes)
            for art, m in meds.items():
                if m.peso > 0.02:
                    medidas[art].append((nombre, m))
            if nn:
                notas.append(f"{nombre}:" + ",".join(nn))

        fuentes = {}
        triangulados = set()
        # Diagnóstico (no cambia la fusión): por articulación, la medida de cada cámara en el
        # marco A (con o sin profundidad), el χ² entre cámaras y la cámara elegida si discrepan.
        # Lo leen diagnostico.RegistroQuieto y el HUD.
        self.diag = {}
        for art, lista in medidas.items():
            kf = self.kf[art]
            if not lista:
                continue
            self.diag[art] = dict(
                meds={n: (self.cams[n].a_A(m.p), bool(m.tiene_z), bool(m.sospechosa)) for n, m in lista},
                d2=None, elegida=None, tri=None)
            # ── 1º) triangulación con los rayos de las dos cámaras ──
            if par.triangular and len({n for n, _ in lista}) >= 2:
                tri = self._triangular(lista)
                if tri is not None:
                    X, S_X, res_max = tri
                    self.diag[art]["tri"] = res_max
                    ok = self._actualizar_kf(art, X, S_X, t, True)
                    triangulados.add(art)
                    fuentes[art] = "+".join(n for n, _ in lista) + ":T" + ("" if ok else "x")
                    continue
            hay_buena = any(m.tiene_z and not m.sospechosa for _, m in lista)
            # modo de cada medida: buena → normal; sospechosa → rayo si hay otra buena, si no
            # ensanchada; sin profundidad → rayo
            usar = []
            for nombre, m in lista:
                cam = self.cams[nombre]
                if not m.tiene_z:
                    modo = "rayo"
                elif m.sospechosa:
                    modo = "rayo" if hay_buena else "sospecha"
                else:
                    modo = "normal"
                if modo == "rayo" and kf.p is None and len(lista) < 2:
                    continue                                   # un rayo solo no da un punto
                pA = cam.a_A(m.p)
                SA = cam.R @ m.covarianza(modo) @ cam.R.T
                usar.append((nombre, m, pA, SA, modo))
            if not usar:
                continue
            etiqueta = "+".join(n + {"normal": "E", "sospecha": "S", "rayo": "R"}[modo] for n, _, _, _, modo in usar)
            if len(usar) >= 2:
                # ¿discrepan las dos cámaras más de lo que permite su incertidumbre?
                (_, ma, pa, Sa, _), (_, mb, pb, Sb, _) = usar[0], usar[1]
                d = pa - pb
                try:
                    d2 = float(d @ np.linalg.solve(Sa + Sb, d))
                except np.linalg.LinAlgError:
                    d2 = 0.0
                self.diag[art]["d2"] = d2
                if d2 > par.chi2_gate and kf.p is not None:
                    # elegir la más compatible con la predicción
                    cands = []
                    for nombre, m, pA, SA, modo in usar:
                        y = pA - kf.p
                        try:
                            dm = float(y @ np.linalg.solve(SA + kf.Pp, y))
                        except np.linalg.LinAlgError:
                            dm = float("inf")
                        cands.append((dm, nombre, pA, SA, modo))
                    cands.sort(key=lambda c: c[0])
                    _, nombre, pA, SA, modo = cands[0]
                    self.diag[art]["elegida"] = nombre
                    ok = self._actualizar_kf(art, pA, SA, t, modo != "rayo")
                    fuentes[art] = f"{etiqueta}!{nombre}" + ("" if ok else "x")
                    continue
                W = sum(np.linalg.inv(SA) for _, _, _, SA, _ in usar)
                b = sum(np.linalg.inv(SA) @ pA for _, _, pA, SA, _ in usar)
                try:
                    Sc = np.linalg.inv(W)
                    pc = Sc @ b
                except np.linalg.LinAlgError:
                    pc, Sc = usar[0][2], usar[0][3]
                con_z = any(modo != "rayo" for _, _, _, _, modo in usar) or len(usar) >= 2
                ok = self._actualizar_kf(art, pc, Sc, t, con_z)
                fuentes[art] = etiqueta + ("" if ok else "x")
            else:
                nombre, m, pA, SA, modo = usar[0]
                ok = self._actualizar_kf(art, pA, SA, t, modo != "rayo")
                fuentes[art] = etiqueta + ("" if ok else "x")

        # ── estado de cada articulación ────────────────────────────────────────
        def valido(art):
            kf = self.kf[art]
            return kf.p is not None and kf.t_ult_medida is not None and (t - kf.t_ult_medida) <= par.hold_s

        for art in self.ARTICULACIONES:
            if art not in fuentes:
                kf = self.kf[art]
                if kf.p is not None:
                    fuentes[art] = "pred" if valido(art) else "perdido"
                else:
                    fuentes[art] = "-"
        self.resumen = " ".join(f"{a[0]}:{fuentes[a]}" for a in ("hombro", "codo", "muneca")) \
            + (("  [" + " ".join(notas) + "]") if notas else "")
        if not (valido("hombro") and valido("muneca")):
            return None

        S = self.kf["hombro"].p
        W = self.kf["muneca"].p
        E = self.kf["codo"].p if valido("codo") else None
        P_E = self.kf["codo"].Pp if E is not None else None
        codo_rec = False

        # ── coherencia con los huesos ──────────────────────────────────────────
        if self.longitudes is not None:
            L1, L2 = self.longitudes
            Ltot = L1 + L2
            d = float(np.linalg.norm(W - S))
            if d > Ltot * (1.0 + 0.5 * par.tol_huesos):
                W = S + (W - S) / d * Ltot          # nunca más lejos que el brazo estirado
            codo_fiable = (E is not None and P_E is not None
                           and math.sqrt(max(float(np.trace(P_E)) / 3.0, 0.0)) < par.codo_sigma_fiable
                           and abs(np.linalg.norm(E - S) - L1) <= par.tol_huesos * L1
                           and abs(np.linalg.norm(W - E) - L2) <= par.tol_huesos * L2)
            if not codo_fiable:
                E_rec = reconstruir_codo(S, W, L1, L2, self.ultimo_codo_rel if self.ultimo_codo_rel is not None
                                         else (E - S if E is not None else None))
                if E_rec is not None:
                    E, codo_rec = E_rec, True
            if E is not None and not codo_rec:
                self.ultimo_codo_rel = E - S
        if E is None:
            return None

        torso = self._marco_torso(S)
        return {
            "hombro": S, "codo": E, "muneca": W,
            "P_hombro": self.kf["hombro"].Pp, "P_codo": P_E, "P_muneca": self.kf["muneca"].Pp,
            "v_muneca": self.kf["muneca"].v,
            "torso": torso,
            "codo_reconstruido": codo_rec,
            "fuentes": self.resumen,
            "triangulado": frozenset(triangulados),
        }

    def _triangular(self, lista):
        """Punto 3D (marco A) a partir de los rayos de dos o más cámaras: mínimos cuadrados de
        la distancia perpendicular a cada rayo, pesada por su σ angular (σ_pix/f, peor con
        visibilidad o escorzo) × distancia; el estéreo de cada cámara, si es bueno, entra como
        apoyo débil a lo largo de su rayo. Devuelve (X, covarianza, residuo máx. en σ) o None si
        los rayos son casi paralelos, el punto queda detrás de una cámara o los rayos no se
        cortan (un landmark mal en una de las dos: se deja a la lógica de una cámara)."""
        par = self.par
        rayos = []
        for nombre, m in lista:
            cam = self.cams[nombre]
            d = cam.R @ m.rayo
            d = d / math.sqrt(float(d @ d))
            z_ref = max(math.sqrt(float(m.p @ m.p)), 1e-3)
            pz = cam.a_A(m.p) if (m.tiene_z and not m.sospechosa) else None
            sz = max(m.s_z, par.tri_sigma_z_min) if pz is not None else None
            rayos.append((cam.t, d, m.s_lat / z_ref, z_ref, pz, sz, np.outer(d, d)))
        # ángulo entre rayos (el mayor de los pares)
        cmin = 1.0
        for i in range(len(rayos)):
            for j in range(i + 1, len(rayos)):
                cmin = min(cmin, abs(float(rayos[i][1] @ rayos[j][1])))
        if math.degrees(math.acos(min(1.0, cmin))) < par.tri_ang_min_deg:
            return None
        X = None
        for _ in range(2):
            A = np.zeros((3, 3))
            b = np.zeros(3)
            for c, d, s_ang, z_ref, pz, sz, D in rayos:
                if X is not None:
                    v = X - c
                    dist = math.sqrt(float(v @ v))
                else:
                    dist = max(z_ref, 0.3)
                w = 1.0 / max(s_ang * dist, 1e-4) ** 2
                P = _I3 - D
                A += w * P
                b += w * (P @ c)
                if pz is not None:
                    wz = 1.0 / (sz * sz)
                    A += wz * D
                    b += wz * (D @ pz)
            try:
                X = np.linalg.solve(A, b)
            except np.linalg.LinAlgError:
                return None
        res_max = 0.0
        for c, d, s_ang, z_ref, pz, sz, D in rayos:
            v = X - c
            prof = float(v @ d)
            if prof < par.z_min:
                return None
            perp = v - prof * d
            sig = max(s_ang * math.sqrt(float(v @ v)), 1e-4)
            res_max = max(res_max, math.sqrt(float(perp @ perp)) / sig)
        if res_max > par.tri_res_max:
            return None
        return X, np.linalg.inv(A), res_max

    def _marco_torso(self, S):
        """(derecha, arriba, delante, lado) en el marco A, como en angulos_humano.py."""
        lado = 1.0 if self.brazo == "derecho" else -1.0
        S2 = self.kf["hombro2"].p
        arriba = None
        if self.par.arriba_ref == "calibrada" and self.par.arriba_fija is not None:
            arriba = np.asarray(self.par.arriba_fija, dtype=float)
        if self.par.arriba_ref == "caderas":
            C, C2 = self.kf["cadera"].p, self.kf["cadera2"].p
            if S2 is not None and C is not None and C2 is not None:
                a = (S + S2) / 2.0 - (C + C2) / 2.0
                if np.linalg.norm(a) > 0.1:
                    arriba = a / np.linalg.norm(a)
        if arriba is None:
            arriba = np.array([0.0, -1.0, 0.0])           # vertical de la cámara A (y de imagen ↓)
        if S2 is not None and np.linalg.norm(S - S2) > 0.1:
            derecha = (S - S2) if self.brazo == "derecho" else (S2 - S)
        else:
            derecha = np.array([-1.0, 0.0, 0.0]) * lado   # sin hombro opuesto: de cara a la cámara A
        derecha = derecha - float(derecha @ arriba) * arriba
        derecha /= max(np.linalg.norm(derecha), 1e-9)
        delante = np.cross(arriba, derecha)
        return derecha, arriba, delante, lado


def reconstruir_codo(S, W, L1, L2, codo_rel_prev=None, abajo=np.array([0.0, 1.0, 0.0])):
    """Codo en la circunferencia compatible con los huesos, del lado del codo anterior (o, sin
    él, 'abajo' = gravedad en ejes de cámara). None si no hay solución."""
    v = W - S
    d = float(np.linalg.norm(v))
    if d < 1e-6:
        return None
    u = v / d
    if d >= L1 + L2 - 1e-6:
        return S + u * L1
    if d <= abs(L1 - L2) + 1e-3:
        return None
    a = (L1 * L1 - L2 * L2 + d * d) / (2.0 * d)
    h = math.sqrt(max(L1 * L1 - a * a, 0.0))
    pref = codo_rel_prev if codo_rel_prev is not None else abajo
    n = pref - float(pref @ u) * u
    nn = float(np.linalg.norm(n))
    if nn < 1e-6:
        n = np.cross(u, np.array([1.0, 0.0, 0.0]))
        nn = float(np.linalg.norm(n))
        if nn < 1e-6:
            return None
    return S + u * a + (n / nn) * h


class CalibradorHuesos:
    """Aprende (brazo, antebrazo) con el brazo estirado en T-pose: mediana de n fotogramas en
    los que los dos huesos caen en bandas razonables. Igual que EstimadorLongitudesBrazo."""
    BANDA_BRAZO = (0.22, 0.38)
    BANDA_ANTEBRAZO = (0.18, 0.34)

    def __init__(self, n=30, manual=None):
        self.n = n
        self.manual = tuple(manual) if manual else None
        self.h1, self.h2 = [], []
        self.fijas = None
        self.calibrando = False
        self.rechazados, self.relajado = 0, False

    def iniciar(self):
        if self.manual is None:
            self.h1, self.h2, self.fijas, self.calibrando = [], [], None, True
            self.rechazados, self.relajado = 0, False

    @property
    def longitudes(self):
        return self.manual if self.manual is not None else self.fijas

    @property
    def progreso(self):
        return len(self.h1)

    FLEX_MAX_DEG = 25.0          # solo con el brazo (casi) recto: así el codo se ve bien
    RATIO_ANTEBRAZO = (0.70, 1.02)   # antebrazo / brazo de una persona (el registro del 28-oct dio
                                     # 1.03: brazo más corto que antebrazo = codo mal medido)

    N_RELAJAR = 90               # fotogramas rechazados tras los que se vuelve al criterio antiguo

    def alimentar(self, S, E, W, fiable=True):
        """fiable=False (p. ej. el codo no salió triangulado con las dos cámaras) → no cuenta.
        Si tras N_RELAJAR fotogramas los filtros nuevos lo rechazan todo (medida muy mala), se
        relajan (solo bandas de longitud, como antes) y se avisa: mejor calibrar algo que nada."""
        if not self.calibrando:
            return
        l1, l2 = float(np.linalg.norm(E - S)), float(np.linalg.norm(W - E))
        if l1 < 1e-6 or l2 < 1e-6:
            return
        flex = math.degrees(math.acos(float(np.clip((E - S) @ (W - E) / (l1 * l2), -1.0, 1.0))))
        estricto = fiable and flex <= self.FLEX_MAX_DEG and \
            self.RATIO_ANTEBRAZO[0] <= l2 / l1 <= self.RATIO_ANTEBRAZO[1]
        if not estricto and not self.relajado:
            self.rechazados += 1
            if self.rechazados >= self.N_RELAJAR:
                self.relajado = True
                print("⚠️  Huesos: la medida no da un brazo recto y proporcionado (¿extrínseca mal, ropa "
                      "oscura?); calibro con el criterio antiguo")
            return
        if self.BANDA_BRAZO[0] <= l1 <= self.BANDA_BRAZO[1] and self.BANDA_ANTEBRAZO[0] <= l2 <= self.BANDA_ANTEBRAZO[1]:
            self.h1.append(l1)
            self.h2.append(l2)
            if len(self.h1) >= self.n:
                self.fijas = (float(np.median(self.h1)), float(np.median(self.h2)))
                self.calibrando = False
