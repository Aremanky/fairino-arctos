"""
Brazo IZQUIERDO en la app (29-oct): ARCTOS + OrcaHand izquierda teleoperados con tu brazo y tu
mano IZQUIERDOS, a la vez que el FR5 con los derechos.

Skill_Shield_Nivel_6.py solo llama a esto en unos pocos sitios (marcados "BRAZO IZQUIERDO"):

    reposo_inicial(modelo, datos)                     # al cargar el modelo
    compensar_gravedad_mano_izq(modelo)               # (con HAND_GRAVCOMP)
    izq = BrazoIzquierdo(modelo, datos, teleop_v2, cfg, guante)   # tras crear la teleop v2 del FR5
    izq.antes_de_paso(m, d, dt)                       # en el hilo de física, antes de cada mj_step
    izq.lanzar_mano(pose_lm_prev, frame_limpio)       # al principio de la vuelta (Hands en paralelo)
    izq.fotograma(datos, t, obs_a, obs_b, pose, frame_bgr, activa)   # tras la v2 del FR5
    izq.activar(datos) / izq.standby(datos) / izq.cerrar()           # [A] / [S] / al salir

Lo comparte con el FR5 (sin duplicar trabajo):
  · las cámaras y su extrínseca A↔B (la calibra y la vigila la teleop del FR5; la del brazo
    izquierdo espera a que termine esa fase al pulsar [A] y luego calibra sus huesos);
  · la vertical del tronco que se mide en la T-pose (mismos Parametros de fusión);
  · el SDK del MANUS (los dedos de la mano izquierda llegan en el mismo callback).
Cada brazo es un obstáculo para el otro (TELEOP_COLISION_ENTRE_BRAZOS).
"""
import math
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np

from skill_shield.teleop_fr5.brazo_arctos import compartir_obstaculos, construir_teleop_arctos
from skill_shield.teleop_fr5.teleop_brazo import TeleopEnParalelo

_ARCTOS_ACT = ["act_arctos_joint1", "act_arctos_joint2", "act_arctos_joint3",
               "act_arctos_joint4", "act_arctos_joint5", "act_arctos_joint6"]
N_MANO = 17
# MediaPipe: puntos de la mano izquierda en Pose y en Hands
_POSE_MUNECA_IZQ, _POSE_MENIQUE_IZQ, _POSE_INDICE_IZQ = 15, 17, 19
_LM_MUNECA, _LM_INDICE_MCP, _LM_CORAZON_MCP, _LM_MENIQUE_MCP = 0, 5, 9, 17


def _cfg(cfg, nombre, defecto=None):
    if cfg is None:
        return defecto
    if isinstance(cfg, dict):
        return cfg.get(nombre, defecto)
    return getattr(cfg, nombre, defecto)


def _indices(modelo, cfg=None):
    """(índices de ctrl del ARCTOS, índice del primer actuador de la mano izquierda) o (None, None)."""
    import mujoco
    A = mujoco.mjtObj.mjOBJ_ACTUATOR
    nombres = _cfg(cfg, "ARCTOS_ACTUADORES", _ARCTOS_ACT)
    idx = [mujoco.mj_name2id(modelo, A, n) for n in nombres]
    i0 = mujoco.mj_name2id(modelo, A, _cfg(cfg, "MANO_IZQ_PRIMER_ACTUADOR", "left_wrist_actuator"))
    if min(idx) < 0 or i0 < 0 or i0 + N_MANO > modelo.nu:
        return None, None
    return idx, i0


def reposo_inicial(modelo, datos, cfg=None):
    """ARCTOS y mano izquierda a su reposo (al cargar el modelo). False si el modelo no los tiene."""
    idx, i0 = _indices(modelo, cfg)
    if idx is None:
        return False
    for i, v in zip(idx, _cfg(cfg, "POSICION_STANDBY_ARCTOS", [0.0, 0.0, 0.21, 0.0, 0.0, 0.0])):
        datos.ctrl[i] = float(v)
    for k, v in enumerate(_cfg(cfg, "POSICION_STANDBY", [0.0] * N_MANO)[:N_MANO]):
        datos.ctrl[i0 + k] = float(v)
    return True


def compensar_gravedad_mano_izq(modelo, nombre="mano_mount_izq"):
    """Como HAND_GRAVCOMP con la derecha: gravcomp en todos los cuerpos que cuelgan de mano_mount_izq."""
    import mujoco
    b0 = mujoco.mj_name2id(modelo, mujoco.mjtObj.mjOBJ_BODY, nombre)
    if b0 < 0:
        return 0
    n = 0
    for b in range(1, modelo.nbody):
        p = b
        while p > 0 and p != b0:
            p = modelo.body_parentid[p]
        if p == b0:
            modelo.body_gravcomp[b] = 1.0
            n += 1
    return n


class BrazoIzquierdo:
    def __init__(self, modelo, datos, teleop_der, cfg, guante_der=None, verbose=True):
        from skill_shield.brazo.control_suave import GeneradorSuave
        self.cfg = cfg
        self.idx, self.i0 = _indices(modelo, cfg)
        if self.idx is None:
            raise ValueError("el modelo no tiene el ARCTOS ni la mano izquierda (¿fairino_mjcf.xml antiguo?)")
        self.der = teleop_der
        g = lambda n, d=None: _cfg(cfg, n, d)
        suelo = g("SOPORTE_SUELO_Z", -0.975)
        self.teleop = construir_teleop_arctos(
            g("RUTA_XML"), teleop_der.fusor.cams, cfg=cfg, par_fusion=teleop_der.fusor.par,
            z_suelo=-0.975 if suelo is None else suelo, longitudes_manual=g("LONG_BRAZO_MANUAL"),
            n_calibracion=int(g("CALIB_HUESOS_FRAMES", 30)), verbose=verbose)
        self.paralelo = TeleopEnParalelo(self.teleop) if g("TELEOP_IZQ_EN_PARALELO", True) else None
        v, a = float(g("TELEOP_IZQ_SUAVE_VEL", 2.5)), float(g("TELEOP_IZQ_SUAVE_ACEL", 20.0))
        self.gen = GeneradorSuave(self.idx, (v,) * 6, (a,) * 6)
        self.gen.activar(datos)                       # sujeta el reposo desde el primer mj_step
        self.standby_q = list(g("POSICION_STANDBY_ARCTOS", [0.0, 0.0, 0.21, 0.0, 0.0, 0.0]))
        self.standby_mano = list(g("POSICION_STANDBY", [0.0] * N_MANO))[:N_MANO]
        self.colision = bool(g("TELEOP_COLISION_ENTRE_BRAZOS", True))
        self.debug = bool(g("TELEOP_IZQ_DEBUG", True))
        self.debug_cada = float(g("DEBUG_ANGULOS_CADA_S", 0.5))
        # dedos: guante MANUS izquierdo sobre el SDK del derecho
        self.guante = None
        if guante_der is not None and str(g("FUENTE_DEDOS_IZQ", "manus")) == "manus":
            from skill_shield.mano.manus_guante import ManoManusSecundaria
            self.guante = ManoManusSecundaria(guante_der, self.i0, g("MANUS_CALIBRACION_IZQ"))
        # pronosupinación y flexión de muñeca de la mano izquierda (MediaPipe Hands en su hilo)
        self.prono = str(g("TELEOP_IZQ_PRONO", "hands"))
        self.muneca_mano = str(g("TELEOP_IZQ_MUNECA_MANO", "camara"))
        self.signo_muneca = float(g("TELEOP_IZQ_MUNECA_SIGNO", -1.0))
        self.recorte_px = int(g("MANO_RECORTE_PX", 256))
        self._det = self._pool = self._fut = None
        if self.prono == "hands" or self.muneca_mano == "camara":
            import mediapipe as mp
            self._det = mp.solutions.hands.Hands(static_image_mode=False, max_num_hands=1,
                                                 min_detection_confidence=0.5, min_tracking_confidence=0.5)
            self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="HandsIzq")
        self._mano_res = None                         # último resultado de Hands (mano izquierda)
        self._caja = None
        self.vec_neutro = self.eje_lat = None
        self.ctrl_muneca = 0.0
        # estado
        self.activo = False
        self.esperando_camaras = False
        self.t_prev = None
        self.t_print = 0.0
        self.info = {"estado": "standby"}
        if verbose:
            print(f"🦾 BRAZO IZQUIERDO: ARCTOS en ctrl[{self.idx[0]}..{self.idx[-1]}], OrcaHand izquierda en "
                  f"ctrl[{self.i0}..{self.i0 + N_MANO - 1}] · alcance {self.teleop.retarget.alcance_robot:.2f} m · "
                  f"pronosupinación '{self.prono}' · dedos "
                  + (f"MANUS izquierdo ({self.guante.calibracion})" if self.guante else "quietos"))

    # ── hilo de física ──────────────────────────────────────────────────────
    def antes_de_paso(self, m, d, dt):
        self.gen.paso(m, d, dt)
        if self.guante is not None:
            self.guante.escribir_ctrl(d, dt)

    # ── teclas ──────────────────────────────────────────────────────────────
    def activar(self, datos):
        """[A]: calibración de huesos del brazo izquierdo (T-pose con LOS DOS brazos)."""
        if self.paralelo is not None:
            self.paralelo.recoger()
            self.paralelo.ultimo = (None, {"estado": "calibrando"})
        self.activo = True
        self.t_prev = None
        self.vec_neutro = self.eje_lat = None
        if not self.gen.activo:
            self.gen.activar(datos)
        # la extrínseca A↔B la calcula la teleop del FR5: los huesos, cuando acabe
        self.esperando_camaras = self.der.fase_cal == "camaras"
        if not self.esperando_camaras:
            self._iniciar_calibracion()

    def _iniciar_calibracion(self):
        self.esperando_camaras = False
        self.teleop.iniciar_calibracion()
        self.teleop.ik.q = None                       # sale de su postura actual

    def standby(self, datos):
        """[S]: ARCTOS y mano izquierda a su reposo (de forma continua)."""
        if self.paralelo is not None:
            self.paralelo.recoger()
        self.activo = False
        self.esperando_camaras = False
        if self.guante is not None:
            self.guante.aplicar = False
        self._enviar(datos, self.standby_q)
        for k, v in enumerate(self.standby_mano):
            datos.ctrl[self.i0 + k] = float(v)
        self.ctrl_muneca = float(self.standby_mano[0]) if self.standby_mano else 0.0
        self.info = {"estado": "standby"}

    def cerrar(self):
        if self.paralelo is not None:
            self.paralelo.cerrar()
            self.paralelo = None
        if self._pool is not None:
            self._pool.shutdown(wait=True)
            self._pool = None
        if self._det is not None:
            self._det.close()
            self._det = None
        self.gen.desactivar()

    # ── por fotograma ───────────────────────────────────────────────────────
    def _enviar(self, datos, q):
        if self.gen.activo:
            self.gen.enviar(datos, self.idx, [float(v) for v in q])
        else:
            for i, v in zip(self.idx, q):
                datos.ctrl[i] = float(v)

    def q_actual(self, datos):
        return [float(datos.ctrl[i]) for i in self.idx]

    def lanzar_mano(self, pose_lm, imagen):
        """MediaPipe Hands sobre el recorte de TU MANO IZQUIERDA (hilo propio; Pose del fotograma
        anterior). Se recoge en fotograma()."""
        if self._pool is None or pose_lm is None or imagen is None or self._fut is not None:
            return
        from skill_shield.vision.recorte_mano import caja_recorte_mano, detectar_mano_en_recorte
        caja = caja_recorte_mano(pose_lm, "izquierdo", imagen.shape[1], imagen.shape[0])
        if caja is not None:
            self._caja = (caja, imagen.shape[1], imagen.shape[0])
            self._fut = self._pool.submit(detectar_mano_en_recorte, self._det, imagen, caja, self.recorte_px)

    def _recoger_mano(self):
        if self._fut is None:
            return None
        try:
            r = self._fut.result()
        except Exception:
            r = None
        self._fut = None
        if r is not None and r.multi_hand_world_landmarks:
            return r
        return None

    @staticmethod
    def _wl(lms, i):
        l = lms.landmark[i]
        return np.array([l.x, l.y, l.z], dtype=float)

    def _mano_vectores(self, res_hands, pose):
        """(adelante, índice→meñique) de tu mano izquierda en ejes de la cámara A, o None."""
        if self.prono == "fija":
            return None
        if self.prono == "hands" and res_hands is not None:
            hw = res_hands.multi_hand_world_landmarks[0]
            return (self._wl(hw, _LM_CORAZON_MCP) - self._wl(hw, _LM_MUNECA),
                    self._wl(hw, _LM_MENIQUE_MCP) - self._wl(hw, _LM_INDICE_MCP))
        if pose is not None and getattr(pose, "pose_world_landmarks", None) is not None:
            pw = pose.pose_world_landmarks
            vis = [pw.landmark[i].visibility for i in (_POSE_MUNECA_IZQ, _POSE_MENIQUE_IZQ, _POSE_INDICE_IZQ)]
            if min(vis) > 0.5:
                w, me, ix = (self._wl(pw, i) for i in (_POSE_MUNECA_IZQ, _POSE_MENIQUE_IZQ, _POSE_INDICE_IZQ))
                return (0.5 * (me + ix) - w, me - ix)
        return None

    def _flexion_muneca(self, datos, res_hands):
        """Flexión de la OrcaHand izquierda: como la derecha (retargeting.retarget_a_ctrl), tu mano
        respecto a su pose al empezar, con zona muerta y ganancia, y el signo de la mano espejo."""
        if self.muneca_mano != "camara" or res_hands is None:
            return
        from skill_shield.mano.biomecanica import calcular_flexion_muneca
        hw = res_hands.multi_hand_world_landmarks[0]
        if self.vec_neutro is None:
            self.vec_neutro, self.eje_lat = calcular_flexion_muneca(hw)
            return
        crudo = float(calcular_flexion_muneca(hw, self.vec_neutro, self.eje_lat))
        zm, gan = 0.25, 1.40                           # los de la mano derecha
        obj = 0.0 if abs(crudo) < zm else math.copysign(abs(crudo) - zm, crudo) * gan
        obj = float(np.clip(self.signo_muneca * obj, -1.134, 0.610))
        self.ctrl_muneca = 0.85 * self.ctrl_muneca + 0.15 * obj
        datos.ctrl[self.i0] = self.ctrl_muneca

    def fotograma(self, datos, t, obs_a, obs_b, pose, frame_bgr, activa, q_der=None):
        """Una vuelta del bucle (con el candado de la física cogido, como el resto del bucle)."""
        res_hands = self._recoger_mano()
        if self.guante is not None:
            self.guante.aplicar = bool(activa and self.activo)
        if frame_bgr is not None and self._caja is not None and self._pool is not None:
            (x0, y0, lado), w, h = self._caja              # recuadro del recorte de la mano izquierda
            sx, sy = frame_bgr.shape[1] / w, frame_bgr.shape[0] / h
            cv2.rectangle(frame_bgr, (int(x0 * sx), int(y0 * sy)), (int((x0 + lado) * sx), int((y0 + lado) * sy)),
                          (255, 0, 200) if res_hands is not None else (0, 0, 255), 1)
        if not (activa and self.activo):
            return
        self._flexion_muneca(datos, res_hands)
        if self.esperando_camaras:
            if self.der.fase_cal == "camaras":
                self.info = {"estado": "calibrando", "texto": "IZQ: esperando a la calibracion de camaras"}
                self._hud(frame_bgr)
                return
            self._iniciar_calibracion()
        dt = None if self.t_prev is None else min(t - self.t_prev, 0.1)
        self.t_prev = t
        mano = self._mano_vectores(res_hands, pose)
        # La observación de la cámara A trae la silueta de tu mano DERECHA (la del FR5): para el
        # brazo izquierdo se rehace con la de tu mano izquierda (o sin ella: la fusión la estima)
        obs_a, obs_b = self._obs_izq(obs_a, res_hands), self._obs_izq(obs_b, None)
        q_act = self.q_actual(datos)
        if self.colision:
            compartir_obstaculos(self.der, q_der, self.teleop, q_act)
        if self.paralelo is not None:
            q, info = self.paralelo.recoger()
            self.paralelo.lanzar({"A": obs_a, "B": obs_b}, t, mano=mano, q_actual=q_act, dt=dt)
        else:
            q, info = self.teleop.actualizar({"A": obs_a, "B": obs_b}, t, mano=mano, q_actual=q_act, dt=dt)
        self.info = info
        est = info.get("estado", "?")
        if q is not None and est in ("exacta", "aprox", "transicion", "bloqueado"):
            self._enviar(datos, q)
        self._hud(frame_bgr)
        if self.debug and t - self.t_print >= self.debug_cada:
            self.t_print = t
            qtxt = "  ".join(f"{v:+.2f}" for v in q) if q is not None else "-"
            print(f"[ARCTOS izq] {est:10s} rama={info.get('rama')} err={100 * (info.get('error') or 0):4.1f}cm  "
                  f"q=[{qtxt}]  {info.get('motivo', '')}")

    @staticmethod
    def _obs_izq(obs, res_hands):
        if obs is None:
            return None
        from skill_shield.teleop_fr5.fusion_muneca_codo import ObservacionCamara
        mano_px = None
        if res_hands is not None and res_hands.multi_hand_landmarks:
            mano_px = np.array([[l_.x * obs.intr.ancho, l_.y * obs.intr.alto]
                                for l_ in res_hands.multi_hand_landmarks[0].landmark])
        return ObservacionCamara(obs.lm, obs.depth, obs.intr, mano_lm=mano_px, z_func=obs.z_func, t=obs.t)

    def _hud(self, frame_bgr):
        if frame_bgr is None:
            return
        info = self.info
        est = info.get("estado", "?")
        if est == "calibrando":
            txt = info.get("texto") or f"IZQ: CALIBRANDO HUESOS {info.get('progreso', 0)} - T-pose"
            col = (0, 200, 255)
        else:
            txt = f"ARCTOS (izq): {est.upper()} err={100 * (info.get('error') or 0):.1f}cm"
            col = {"exacta": (0, 255, 0), "aprox": (0, 200, 255), "transicion": (255, 200, 0)}.get(est, (0, 0, 255))
        x = max(10, frame_bgr.shape[1] - 10 - 8 * len(txt))
        cv2.putText(frame_bgr, txt, (x, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1)
        if self.guante is not None:
            tg = self.guante.estado_txt()
            cv2.putText(frame_bgr, tg, (max(10, frame_bgr.shape[1] - 290), 85), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                        (0, 255, 0) if "OK" in tg else (0, 165, 255), 1)
