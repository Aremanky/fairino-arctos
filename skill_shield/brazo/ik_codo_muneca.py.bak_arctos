"""
Paso 3 — direcciones + IK con codo y muñeca (ideas del proyecto TRON 2).

(Extraído de Skill_Shield_Nivel_6.py — código sin cambios, solo reubicado.)
"""
import numpy as np
import mujoco
import math

from config import *

from skill_shield.brazo.angulos_humano import _lm_hand, _unitario, _wrap

# ── 7-oct: valores por defecto de los parámetros nuevos (por si config.py es antiguo) ──
for _k, _v in {
    "REUBICAR_OBJETIVOS": True, "COMPRIMIR_ABAJO": True, "PALMA_Z_MIN": 0.15,
    "CODO_Z_MIN": 0.12, "VERTICAL_HORIZ": (0.35, 0.60), "BASE_BLOQUEO_W": 0.3,
}.items():
    globals().setdefault(_k, _v)


# ══════════════════════════════════════════════════════════════════════════════
# PASO 3 — DIRECCIONES + IK CON CODO Y MUÑECA (ideas del proyecto TRON 2)
# ══════════════════════════════════════════════════════════════════════════════

class FiltroOneEuro:
    """
    PASO 5 — Filtro One Euro (Casiez et al., 2012), el mismo del TRON 2.
    Se adapta a la velocidad: en REPOSO filtra mucho (no tiembla) y cuando te
    mueves RÁPIDO filtra poco (casi sin retraso). Una media móvil, en cambio,
    retrasa siempre lo mismo. Funciona con vectores (np.array).
    """

    def __init__(self, min_cutoff=None, beta=None, d_cutoff=1.0):
        self.min_cutoff = FILTRO_MIN_CUTOFF if min_cutoff is None else min_cutoff
        self.beta       = FILTRO_BETA if beta is None else beta
        self.d_cutoff   = d_cutoff
        self.reset()

    def reset(self):
        self.x = self.dx = self.t = None

    @staticmethod
    def _alfa(corte, dt):
        tau = 1.0 / (2.0 * math.pi * corte)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, x, t):
        x = np.asarray(x, dtype=float)
        if self.x is None:
            self.x, self.dx, self.t = x.copy(), np.zeros_like(x), t
            return x.copy()
        dt = max(t - self.t, 1e-3)
        a_d = self._alfa(self.d_cutoff, dt)
        self.dx = a_d * (x - self.x) / dt + (1.0 - a_d) * self.dx
        a = self._alfa(self.min_cutoff + self.beta * float(np.linalg.norm(self.dx)), dt)
        self.x = a * x + (1.0 - a) * self.x
        self.t = t
        return self.x.copy()


def _eje_codo_humano_torso(az_rad):
    """Eje del codo del humano (horizontal, ⟂ al plano vertical de su brazo según
    el azimut) en componentes (fuera, arriba, delante). Solo importa su SIGNO
    relativo: se compara con el eje del codo del robot."""
    return np.array([math.cos(az_rad), 0.0, -math.sin(az_rad)])


def semilla_girada(q):
    """Misma q con la base girada 180° y el hombro inclinado hacia el otro lado:
    lleva a la solución con el codo doblado al OTRO lado ("codo arriba")."""
    q = np.asarray(q, dtype=float).copy()
    q[0] = _wrap(q[0] + math.pi)
    q[1] = -q[1]
    return q


def es_girada(q, q1_ref):
    """¿Está la base apuntando al lado contrario de tu brazo (configuración girada)?
    q1_ref = joint1 que pide el modo ángulos según tu azimut."""
    return abs(_wrap(float(q[0]) - float(q1_ref))) > math.pi / 2


def coste_ik(q, err, q_actual, q1_ref):
    """Error + penalizaciones. Evita que el rescate elija soluciones que exigen
    girar mucho la base (vueltas) o la configuración girada si no compensa."""
    c = err + IK_PENALIZA_BASE * abs(float(q[0]) - float(q_actual[0]))
    if es_girada(q, q1_ref):
        c += IK_PENALIZA_GIRADA
    return c


def rescatar_ik(modelo, datos, obj_c, obj_m, w_codo, hib, semillas, q_actual, q1_ref):
    """PASO 6 — resuelve desde varias posturas de partida y devuelve la de MENOR
    COSTE (error + penalizaciones): (q, error, coste, nombre)."""
    mejor = (None, float("inf"), float("inf"), "")
    for nombre, q0 in semillas:
        q, ec, em = resolver_ik_codo_muneca(modelo, datos, obj_c, obj_m, w_codo, hib, q0,
                                            n_iter=IK_RESCATE_ITER)
        e = math.hypot(w_codo * ec, em)
        c = coste_ik(q, e, q_actual, q1_ref)
        if c < mejor[2]:
            mejor = (q, e, c, nombre)
    return mejor


def _equivalente_en_rango(q, q_actual, lo, hi):
    """De q, q±2π, el que cae dentro de [lo, hi] más cerca de q_actual."""
    cands = [q + k * 2.0 * math.pi for k in (-1, 0, 1)]
    cands = [c for c in cands if lo <= c <= hi]
    if not cands:
        return float(np.clip(q, lo, hi))
    return float(min(cands, key=lambda c: abs(c - q_actual)))


def _perp(v, eje):
    """Componente de v perpendicular a eje (unitario), normalizada. None si no hay."""
    w = np.asarray(v, dtype=float) - float(np.dot(v, eje)) * np.asarray(eje)
    return _unitario(w) if np.linalg.norm(w) > 0.3 * max(np.linalg.norm(v), 1e-9) else None


def lateral_mano_humana_mundo(hand_wlm, torso, hib):
    """Dirección índice→meñique de TU mano, llevada a ejes del mundo del robot.
    La profundidad de MediaPipe Hands sale 'aplastada': se estira con PRONO_Z_ESCALA."""
    lat = _lm_hand(hand_wlm, LM_MENIQUE_MCP) - _lm_hand(hand_wlm, LM_INDICE_MCP)
    lat = np.array([lat[0], lat[1], lat[2] * PRONO_Z_ESCALA])
    derecha, arriba, delante, lado = torso
    t = np.array([lado * float(lat @ derecha), float(lat @ arriba), float(lat @ delante)])
    return _unitario(hib["ejes"] @ t)


def q6_para_orientacion(datos, hib, lat_local, lat_objetivo, q6_actual):
    """ORIENTACIÓN EN EL MUNDO (28-sep): qué joint6 deja la mano del robot con su
    índice→meñique apuntando igual que el tuyo, sea cual sea la postura del brazo
    que haya elegido la IK (normal, girada...). Devuelve None si no es fiable."""
    eje = _unitario(np.array(datos.xaxis[hib["j6"]]))
    if eje is None:
        return None
    lat_robot = np.array(datos.xmat[hib["b_mano"]]).reshape(3, 3) @ lat_local
    u, v = _perp(lat_robot, eje), _perp(lat_objetivo, eje)
    if u is None or v is None:
        return None
    delta = math.atan2(float(np.cross(u, v) @ eje), float(u @ v))
    return q6_actual + delta


def _suavizado_01(x, lo, hi):
    """0 por debajo de lo, 1 por encima de hi, curva suave (smoothstep) en medio."""
    t = float(np.clip((x - lo) / (hi - lo), 0.0, 1.0))
    return t * t * (3.0 - 2.0 * t)


def enderezar(d_b, d_a):
    """Zona muerta en la flexión del codo. Con el brazo casi estirado, el ruido del
    seguimiento se convierte en pequeñas flexiones en direcciones aleatorias
    (en tus T-pose el codo mide 17-25° estando recto). Por debajo de
    ZONA_MUERTA_CODO[0] el antebrazo se alinea con el brazo; hasta [1] se atenúa."""
    theta = float(np.arccos(np.clip(np.dot(d_b, d_a), -1.0, 1.0)))
    s = math.sin(theta)
    if s < 1e-6:
        return np.asarray(d_a, dtype=float)
    lo, hi = np.radians(ZONA_MUERTA_CODO)
    theta_n = theta * _suavizado_01(theta, lo, hi)
    return (math.sin(theta - theta_n) * np.asarray(d_b) + math.sin(theta_n) * np.asarray(d_a)) / s


def peso_codo_humano(d_b, d_a):
    """Con el brazo casi recto la posición del codo no aporta (y es ruido):
    su peso en la IK baja a 0. Doblado, vale HIBRIDO_PESO_CODO."""
    ang = math.degrees(math.acos(float(np.clip(np.dot(d_b, d_a), -1.0, 1.0))))
    return HIBRIDO_PESO_CODO * _suavizado_01(ang, *HIBRIDO_CODO_RECTO)


def objetivos_codo_muneca(datos, d_b, d_a, hib):
    """Tus direcciones (fuera, arriba, delante) → dónde deben estar el codo y la
    muñeca del ARCTOS: SU hombro + SUS longitudes en TUS direcciones."""
    hombro = np.array(datos.xanchor[hib["j_hombro"]])
    codo   = hombro + hib["L_brazo"] * (hib["ejes"] @ d_b)
    muneca = codo + hib["L_antebrazo"] * (hib["ejes"] @ d_a)
    if REUBICAR_OBJETIVOS:                                  # 7-oct: puntos 1 y 3
        v, h_ref, w = reubicar_palma(datos, hib, hombro, muneca - hombro)
        muneca = hombro + v
        d_codo, _ = reubicar_direccion(hib["ejes"] @ np.asarray(d_b), hib["L_brazo"],
                                       float(hombro[2]), CODO_Z_MIN, h_ref, w)
        codo = hombro + hib["L_brazo"] * d_codo
    muneca[2] = max(muneca[2], ARM_Z_MINIMA)
    return codo, muneca


def codo_compatible(hombro, palma, codo_humano, L1, L2):
    """
    PASO 7.4 — Dónde PUEDE estar el codo del ARCTOS si su palma va a 'palma':
    a L1 del hombro y a L2 de la palma (L2 = antebrazo + OrcaHand). Esos puntos forman
    una circunferencia; se elige el más cercano a 'codo_humano' (tu dirección de
    brazo). Así el codo solo decide la POSTURA y ya no tira de la palma: antes el
    objetivo del codo estaba en tu dirección con las longitudes del robot, y con el
    codo doblado era imposible cumplir los dos → la palma se pasaba hasta 3-4 cm.
    Si no hay solución (palma demasiado lejos o cerca), devuelve codo_humano.
    """
    v = np.asarray(palma, dtype=float) - hombro
    d = float(np.linalg.norm(v))
    if d < 1e-6 or d <= abs(L1 - L2) + 1e-4:
        return np.asarray(codo_humano, dtype=float)
    u = v / d
    if d >= L1 + L2:                                   # brazo estirado: codo en la línea
        return hombro + u * L1
    a = (L1 * L1 - L2 * L2 + d * d) / (2.0 * d)        # distancia del hombro al centro
    h = math.sqrt(max(L1 * L1 - a * a, 0.0))           # radio de la circunferencia
    w = np.asarray(codo_humano, dtype=float) - hombro
    n = w - float(w @ u) * u                           # hacia TU codo, ⟂ a hombro→palma
    nn = float(np.linalg.norm(n))
    if nn < 1e-6:
        return np.asarray(codo_humano, dtype=float)
    return hombro + a * u + (n / nn) * h


# ══════════════════════════════════════════════════════════════════════════════
# 7-oct — ESPACIO DE TRABAJO (puntos 1 y 3 de la revisión del vídeo del 7-oct)
# ══════════════════════════════════════════════════════════════════════════════
# 1) Tu hombro está a ~1,4 m del suelo; el del ARCTOS, a 0,29 m. Con el brazo por
#    debajo de la horizontal, el objetivo de la palma caía bajo el suelo y se
#    RECORTABA a ARM_Z_MINIMA: error de 50-80 cm, IK desesperada, codo plegado y
#    choques con el suelo. Ahora la mitad inferior se COMPRIME: de la horizontal
#    hacia arriba la elevación es 1:1; de la horizontal hacia abajo se reparte en el
#    rango que el robot sí puede alcanzar (palma ≥ PALMA_Z_MIN, codo ≥ CODO_Z_MIN).
#    Bajar el brazo sigue bajando la mano del robot, de forma continua.
# 3) Con el brazo casi VERTICAL (colgando o arriba) la dirección horizontal es ruido
#    y la base daba vueltas. Ahí se usa la dirección horizontal ACTUAL del robot
#    (transición suave según VERTICAL_HORIZ) y la IK bloquea joint1 (bloqueo_base).

def _horizontal_actual(datos, hib):
    """Dirección horizontal (unitaria, mundo) hombro→palma del robot AHORA."""
    bm = hib["b_mano"]
    p = datos.xpos[bm] + datos.xmat[bm].reshape(3, 3) @ hib.get("off_punta", np.zeros(3))
    v = np.array(p - datos.xanchor[hib["j_hombro"]], dtype=float)
    v[2] = 0.0
    n = float(np.linalg.norm(v))
    if n < 0.05:                        # el propio robot está casi vertical
        v = np.array(hib["ejes"][:, 2], dtype=float)     # su "delante"
        v[2] = 0.0
        n = float(np.linalg.norm(v))
    return v / n


def reubicar_direccion(u, dist, z_hombro, z_min, h_ref, w_h=None):
    """
    u: dirección unitaria (mundo) hombro→punto que está a 'dist' m del hombro.
    Devuelve (u', w): u' con la dirección horizontal estabilizada y la parte de
    abajo comprimida para que el punto no baje de z_min; w ∈ [0, 1] = cuánto se fía
    de TU dirección horizontal (0 = brazo vertical → se usa h_ref, la del robot).
    """
    u = np.asarray(u, dtype=float)
    uh = np.array([u[0], u[1], 0.0])
    nh = float(np.linalg.norm(uh))
    w = _suavizado_01(nh, *VERTICAL_HORIZ) if w_h is None else w_h
    h = uh / nh if nh > 1e-9 else h_ref
    h = _unitario(w * h + (1.0 - w) * h_ref)
    if h is None:
        h = h_ref
    e = math.acos(float(np.clip(-u[2], -1.0, 1.0)))       # 0 = abajo · π/2 = horizontal · π = arriba
    if COMPRIMIR_ABAJO and e < math.pi / 2:
        caida = z_hombro - z_min                           # lo que puede bajar el punto
        e_min = math.acos(float(np.clip(caida / dist, -1.0, 1.0))) if caida < dist else 0.0
        e = e_min + e * (math.pi / 2 - e_min) / (math.pi / 2)
    return math.sin(e) * h + np.array([0.0, 0.0, -math.cos(e)]), w


def reubicar_palma(datos, hib, hombro, v):
    """Vector hombro→palma (mundo) → (vector corregido, h_ref, w). Guarda w en
    hib["_w_horizontal"] para que el bucle sepa si debe bloquear la base."""
    d = float(np.linalg.norm(v))
    h_ref = _horizontal_actual(datos, hib)
    if d < 1e-6:
        hib["_w_horizontal"] = 0.0
        return np.asarray(v, dtype=float), h_ref, 0.0
    u, w = reubicar_direccion(np.asarray(v) / d, d, float(hombro[2]), PALMA_Z_MIN, h_ref)
    hib["_w_horizontal"] = w
    return u * d, h_ref, w


def bloqueo_base(hib):
    """[joint1, joint2, joint3] que la IK NO debe mover: joint1 con el brazo vertical."""
    return np.array([hib.get("_w_horizontal", 1.0) < BASE_BLOQUEO_W, False, False])


def codo_fiable(ang) -> bool:
    """
    7-oct (punto 2) — ¿El CODO de esta medida es de fiar? En el vídeo casi siempre
    salía reconstruido ("c:R", CODO ESTIMADO) y su objetivo quedaba a 22-34 cm del
    robot: con peso en la IK, tiraba de la postura hacia algo que no habías hecho.
    Fiable = medido con estéreo ('E' en su fuente) y ni reconstruido ni de MediaPipe.
    Sin información de fuentes (modo "mediapipe") se considera fiable.
    """
    if ang is None:
        return False
    if ang.get("codo_estimado"):
        return False
    fuentes = ang.get("fuentes")
    if not fuentes:
        return True
    for parte in str(fuentes).split():
        if parte.startswith("c:"):
            f = parte[2:].split("!")[0]                 # "E+E!A" → "E+E"
            return "E" in f and "R" not in f and "M" not in f
    return True


def objetivos_codo_palma(datos, d_b, d_a, hib, longitudes_humano):
    """
    La mano del robot va donde va TU PALMA, teniendo en cuenta que el ARCTOS
    con la OrcaHand no tiene tus proporciones:
      · tu palma respecto a tu hombro = L1·dir_brazo + (L2 + muñeca→palma)·dir_antebrazo
        (con TUS huesos calibrados)
      · se divide por tu alcance total (brazo + antebrazo + mano) → fracción ≤ 1
      · se multiplica por el alcance del ARCTOS hasta SU palma
    El codo sigue la dirección de tu brazo (postura), con peso en la IK.
    La mano se considera alineada con el antebrazo: la flexión de muñeca ya la
    hace la propia OrcaHand, y contarla aquí también la duplicaría.
    """
    L1, L2 = longitudes_humano
    L3 = HUMANO_MUNECA_PALMA
    palma_h = L1 * np.asarray(d_b) + (L2 + L3) * np.asarray(d_a)
    frac = palma_h / (L1 + L2 + L3)                      # |frac| ≤ 1 siempre
    hombro = np.array(datos.xanchor[hib["j_hombro"]])
    v = (hib["ejes"] @ frac) * hib["alcance_punta"] * HIBRIDO_ESCALA
    r, r_max = float(np.linalg.norm(v)), HIBRIDO_ALCANCE_MAX * hib["alcance_punta"]
    if r > r_max:
        v *= r_max / r
    d_codo = hib["ejes"] @ np.asarray(d_b)
    if REUBICAR_OBJETIVOS:                                  # 7-oct: puntos 1 y 3
        v, h_ref, w = reubicar_palma(datos, hib, hombro, v)
        d_codo, _ = reubicar_direccion(d_codo, hib["L_brazo"], float(hombro[2]),
                                       CODO_Z_MIN, h_ref, w)
    palma = hombro + v
    palma[2] = max(palma[2], ARM_Z_MINIMA)
    codo = hombro + hib["L_brazo"] * d_codo
    if CODO_COMPATIBLE and "L_antebrazo_punta" in hib:      # PASO 7.4
        codo = codo_compatible(hombro, palma, codo, hib["L_brazo"], hib["L_antebrazo_punta"])
    return codo, palma


def resolver_ik_codo_muneca(modelo, datos, obj_c, obj_m, w_codo, hib, q_inicial,
                            n_iter=None, lam=0.05, paso_max=None, q_ref=None, w_cont=0.0,
                            bloqueados=None):
    """
    IK de joint1..3 que persigue A LA VEZ el codo (con peso w_codo) y la muñeca
    del ARCTOS, por mínimos cuadrados amortiguados. Como el ARCTOS no puede
    reproducir cualquier postura humana (hombro de 2 ejes), busca el mejor
    compromiso. Tomado del TRON 2:
      · si una articulación se saldría de su límite, se BLOQUEA y las demás
        compensan (en vez de recortar y quedarse con error)
      · búsqueda lineal: solo se aceptan pasos que NO empeoran el error
    Solo usa cinemática (sin física) y deja la simulación como estaba.
    PASO 7.2 — con q_ref y w_cont > 0 (m/rad) también penaliza alejarse de q_ref
    (la solución anterior): cerca de una singularidad no gira la base media vuelta
    para ganar 1 cm. Donde el brazo está bien condicionado apenas influye y, con el
    objetivo quieto, converge igual al punto exacto (no deja error permanente).
    7-oct — 'bloqueados' (3 bool): joints que la IK no mueve (p.ej. joint1 con el
    brazo vertical, ver bloqueo_base). None = todos libres, como antes.
    Devuelve (q, error_codo_m, error_muneca_m) — errores geométricos, sin la penalización.
    """
    n_iter   = HIBRIDO_IK_ITER if n_iter is None else n_iter
    paso_max = HIBRIDO_IK_PASO if paso_max is None else paso_max
    qidx, didx, lo, hi = hib["qpos3"], hib["dof3"], hib["lo"], hib["hi"]
    jc, bc, bm = hib["j_codo"], hib["b_codo"], hib["b_mano"]
    guarda = datos.qpos[qidx].copy()
    jac = np.zeros((3, modelo.nv))
    off = hib.get("off_punta", np.zeros(3))

    def fijar(q):
        datos.qpos[qidx] = q
        mujoco.mj_kinematics(modelo, datos)
        mujoco.mj_comPos(modelo, datos)          # necesario para mj_jac

    def punta():
        """Punto final que se persigue: palma de la OrcaHand (o muñón)."""
        return datos.xpos[bm] + datos.xmat[bm].reshape(3, 3) @ off

    cont = q_ref is not None and w_cont > 0.0
    q_ref = np.asarray(q_ref, dtype=float) if cont else None
    n_filas = 9 if cont else 6

    def errores(qq):
        ec = float(np.linalg.norm(obj_c - datos.xanchor[jc]))
        em = float(np.linalg.norm(obj_m - punta()))
        tot = float(np.hypot(w_codo * ec, em))
        if cont:
            tot = float(np.hypot(tot, w_cont * np.linalg.norm(qq - q_ref)))
        return tot, ec, em

    q = np.clip(np.asarray(q_inicial, dtype=float), lo, hi)
    fijar(q)
    err = errores(q)[0]
    for _ in range(n_iter):
        p_c, p_m = datos.xanchor[jc].copy(), punta().copy()
        mujoco.mj_jac(modelo, datos, jac, None, p_c, bc)
        J_c = jac[:, didx].copy()
        mujoco.mj_jac(modelo, datos, jac, None, p_m, bm)
        J_m = jac[:, didx].copy()
        J = np.vstack([w_codo * J_c, J_m])
        e = np.concatenate([w_codo * (obj_c - p_c), obj_m - p_m])
        if cont:                                             # PASO 7.2: continuidad
            J = np.vstack([J, w_cont * np.eye(3)])
            e = np.concatenate([e, w_cont * (q_ref - q)])

        libres = np.ones(3, dtype=bool) if bloqueados is None \
            else ~np.asarray(bloqueados, dtype=bool)
        if not libres.any():
            break
        for _ in range(3):                                   # bloqueo de límites
            Jl = J * libres
            dq = Jl.T @ np.linalg.solve(Jl @ Jl.T + lam ** 2 * np.eye(n_filas), e)
            dq[~libres] = 0.0
            fuera = libres & (((q + dq) < lo) | ((q + dq) > hi))
            if not fuera.any():
                break
            libres &= ~fuera
        mayor = float(np.abs(dq).max())
        if mayor > paso_max:
            dq *= paso_max / mayor

        aceptado, alfa = False, 1.0                          # búsqueda lineal
        for _ in range(4):
            q_n = np.clip(q + alfa * dq, lo, hi)
            fijar(q_n)
            err_n = errores(q_n)[0]
            if err_n <= err + 1e-4:
                q, err, aceptado = q_n, err_n, True
                break
            alfa *= 0.5
        if not aceptado:
            fijar(q)
            break
        if err < 1e-3 or alfa * min(mayor, paso_max) < 1e-5:   # llegó / ya no avanza
            break

    fijar(q)
    _, ec, em = errores(q)
    fijar(guarda)                                            # la simulación, como estaba
    return q, ec, em


def objetivo_mano_hibrido(datos, v_humano, long_brazo, hib):
    """Tu vector hombro→muñeca (m) → punto (mundo) donde debe estar mano_mount."""
    frac = np.asarray(v_humano, dtype=float) / max(long_brazo, 1e-3)
    n = float(np.linalg.norm(frac))
    if n > 1.0:                       # ruido del estéreo: nunca "más que estirado"
        frac /= n
    v = hib["ejes"] @ frac * hib["alcance"] * HIBRIDO_ESCALA
    r, r_max = float(np.linalg.norm(v)), HIBRIDO_ALCANCE_MAX * hib["alcance"]
    if r > r_max:
        v *= r_max / r
    obj = np.array(datos.xanchor[hib["j_hombro"]]) + v
    obj[2] = max(obj[2], ARM_Z_MINIMA)
    return obj
